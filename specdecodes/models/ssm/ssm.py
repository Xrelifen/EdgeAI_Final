import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_model
import math, os
from typing import List


from bigtree import Node

from .training_hooks import TrainingHook, NEFTuneHook
from .sampling_utils import topk_sampling, k_sampling, heuristic_k_sampling
from ..utils import invert_mask, keep_top_n_nodes

from ..llm.modeling_llama import ACT2FN, LlamaMLP
from ..llm import modeling_llama_no_init_weights as modeling_llama
from ..llm import modeling_llama_no_inout_norm as modeling_llama_eagle
from ..llm import modeling_llama_no_in_norm

     
     
def load_custom_model(model, model_path):
    # Load the model
    missing_keys, unexpected_keys = load_model(model, model_path, strict=False)
    
    # Remove embed_tokens if not found (for custom models that uses LLM's embed_tokens)
    for key in missing_keys:
        if 'embed_tokens' in key:
            print(f"embed_tokens not found. Use LLM's embed_tokens instead.")
            del model.model.embed_tokens
    missing_keys = [key for key in missing_keys if 'embed_tokens' not in key]
    
    # error handling
    assert len(missing_keys) == 0 and len(unexpected_keys) == 0, f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
    
    return model


class MergeLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size*2, config.hidden_size, bias=True)

    def forward(self, x, emb):
        # swapped (x, emb) to (emb, x) to match official implementation of Eagle
        return self.fc(torch.cat((emb, x), dim=-1))

class MergeFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, emb):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(emb, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(emb)) * self.up_proj(x))

        return x + down_proj
    
class FeatureSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.org_hidden_size = config.org_hidden_size
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.org_hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, emb):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(emb, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(emb)) * self.up_proj(x))

        return x + down_proj
    
class Extractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.org_hidden_size = config.org_hidden_size
        self.hidden_size = config.hidden_size # reduced hidden size
        self.intermediate_size = config.org_intermediate_size
        self.gate_proj = nn.Linear(self.org_hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.org_hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class Inserter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size # reduced hidden size
        self.org_hidden_size = config.org_hidden_size # reduced hidden size
        self.intermediate_size = config.org_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.org_hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, hidden_states):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(hidden_states, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(hidden_states, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

        return x + down_proj

    
class SSMBase(nn.Module):
    def __init__(self, model=None, config=None, eos_token_id=None, sampling_method='greedy', keep_embeddings=False):
        super().__init__()
        self.eos_token_id = eos_token_id
        
        # Set model and config
        if model is not None and config is not None:
            raise ValueError("Only one of model or config must be provided.")   
        elif model is not None:
            self.model = model
            self.config = model.config
        elif config is not None:
            self.model = self.init_custom_model(config)
            if not keep_embeddings:
                if hasattr(self.model, "embed_tokens"): 
                    del self.model.embed_tokens
                    self.model.embed_tokens = None
                    
            self.config = config
        else:
            raise ValueError("Either model or config must be provided.")
        self.init_additional_modules(config)
        
        # Draft parameters
        self.init_draft_parameters()
        self.init_sampling_method(sampling_method)
        
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path,
        *model_args,
        config = None,
        torch_dtype=torch.float32,
        **model_kwargs
    ):
        # Remove the following arguments from model_kwargs, cause AutoModelForCausalLM does not accept them
        eos_token_id = model_kwargs.pop("eos_token_id", None)
        sampling_method = model_kwargs.pop("sampling_method", "greedy")
        
        # Load HuggingFace model if config is not provided
        if config is not None: 
            draft_model_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            model = cls(None, config=config, eos_token_id=eos_token_id, sampling_method=sampling_method, *model_args, **model_kwargs)
            model = load_custom_model(model, draft_model_path)
        
        else:
            ssm = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                **model_kwargs
            )
            model = cls(ssm, config=config, eos_token_id=eos_token_id, sampling_method=sampling_method, *model_args, **model_kwargs).to(dtype=torch_dtype)
        
        # Convert the model to the desired dtype and return
        model.to(dtype=torch_dtype)
        return model
        
    def init_custom_model(self, config):
        return modeling_llama.LlamaModel(config)
    
    def init_additional_modules(self, config):
        pass

    def init_draft_parameters(self):
        # self.depth = 9 + 1
        # self.topk_len = 15
        # self.min_sample_prob = 1e-8
        # self.min_accept_prob = 1e-8
        
        self.depth = 6 + 1
        self.topk_len = 8
        self.min_sample_prob = 1e-2
        self.min_accept_prob = 1e-2
        self.max_tokens = -1 #64
    
    def init_sampling_method(self, sampling_method):
        if sampling_method == 'greedy':
            self.sample_nodes = topk_sampling
        elif sampling_method == 'stochastic':
            self.sample_nodes = k_sampling
        elif sampling_method == 'hstochastic':
            self.sample_nodes = heuristic_k_sampling
        else:
            raise ValueError("Sampling method not supported")
        
    @torch.no_grad()
    def forward(self, input_ids, *model_args, **kwargs):
        raise NotImplementedError
    
    @torch.no_grad()
    def speculate(self, input_ids, past_key_values, **kwargs):
        raise NotImplementedError
        
    # Currently not used. This may be used to match LLM's sampling behavior.
    @torch.no_grad()
    def _sample_probs(
        self,
        logits: torch.FloatTensor,
        logits_warper,
        do_sample: bool,
    ):
        if do_sample:
            batch, seq_len, vocab_size = logits.shape
            
            logits = logits.view(-1, vocab_size)
            next_token_scores = logits_warper(None, logits)
            probs = torch.softmax(next_token_scores, dim=-1)
            return probs.view(batch, seq_len, vocab_size) # preserve shape
        
        else:
            return torch.softmax(logits, dim=-1)


class SSMBaseNEFT(SSMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forward_hook_handles: List[TrainingHook] = []
            
    def activate_forward_hooks(self):
        """Activates/registers forward hooks for the model."""
        assert self.model is not None, "Model is not initialized."

        # Initialize forward hook handles
        if hasattr(self.config, "neftune_noise_alpha"):
            self._forward_hook_handles.append(
                NEFTuneHook(neftune_noise_alpha=self.config.neftune_noise_alpha)
            )

        # Activate forward hooks iteratively
        for hook in self._forward_hook_handles:
            # Update the model with the forward hooks in place
            self.model = hook.activate_hook(self.model)

    def deactivate_forward_hooks(self) -> None:
        """Deactivates/de-registers forward hooks for the model (if needed)."""
        for handle in self._forward_hook_handles:
            handle.deactivate_hook()


class SSM_Classic(SSMBaseNEFT):
    def forward(self, input_ids, embed_tokens=None, *model_args, **kwargs):
        # not using hidden_states
        _ = kwargs.pop("hidden_states", None)
        
        with torch.no_grad():
            if self.model.get_input_embeddings():
                embed_tokens = self.model.get_input_embeddings()
            inputs_embeds = embed_tokens(input_ids)

        return self.model(inputs_embeds=inputs_embeds, *model_args, **kwargs)
    
    @torch.no_grad()
    def _update_tree_attention_data(self, depth, nodes, tree_mask, position_offset, device):
        indices = torch.tensor([node.ind for node in nodes])
        
        input_ids = torch.tensor([node.id for node in nodes], device=device)[None]
        
        position_ids = torch.zeros(len(nodes), device=device)[None] + (position_offset + depth)
        
        # Generating tree masks for the new nodes, don't have to consider the old nodes
        tree_mask = tree_mask[:, :, indices]
        tree_mask = torch.concat((tree_mask, torch.eye(len(nodes), device=device, dtype=torch.bool)[None, None]), dim=3)

        return input_ids, position_ids, tree_mask
    
    @torch.no_grad()
    def speculate(self, input_ids, past_key_values, embed_tokens, lm_head, *model_args, **kwargs):
        """This method is used to draft/guess the next tokens that the LLM may generate.

        Args:
            inputs (list): A list of two tensors: hidden_states and input_ids.
            past_key_values (Cache): Cache object to store the past key-values generated by the model.
            embed_tokens (Module): embedding from LLM.
            lm_head (Module): lm_head from LLM.

        Returns:
            Node: The root node of the generated draft token tree.
        """
        
        device = input_ids.device
        if hasattr(self.model, "lm_head"):
            dtype = self.model.lm_head.weight.dtype
        else:
            dtype = lm_head.weight.dtype
        
        # take out last token as sample_token
        sample_token = input_ids[:, -1:]
        
        # keep original length of input_ids
        org_input_len = input_ids.shape[1] # offset of positon_id
        
        # initialize tree_mask and tree 
        tree_mask = torch.ones([1, 1, 1, org_input_len], device=device, dtype=torch.bool)
        root = Node("1", id=sample_token[0][0].item(), prob=1, global_prob=1, ind=-1)
        
        depth = 1 # depth starts from 1 in tree library
        prev_nodes = [root]
        while depth < self.depth:
            #* Decode previous nodes
            if depth == 1: # first iteration
                kv_len = past_key_values.get_seq_length()
                outputs = self(
                    input_ids[:, kv_len:],
                    embed_tokens=embed_tokens,
                    past_key_values=past_key_values,
                )
                if hasattr(self.model, "lm_head"):
                    logits = outputs.logits[:, -1:].clone()
                else:
                    logits = lm_head(outputs.last_hidden_state[:, -1:])
            else:
                input_ids, position_ids, tree_mask = self._update_tree_attention_data(depth, prev_nodes, tree_mask, org_input_len, device=logits.device)
                outputs = self(
                    input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids, 
                    embed_tokens=embed_tokens,
                    attention_mask=invert_mask(tree_mask, dtype=dtype)
                )
                if hasattr(self.model, "lm_head"):
                    logits = outputs.logits
                else:
                    logits = lm_head(outputs.last_hidden_state)

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            #* Get the probabilities of each token
            T = 1
            sampled_probs = torch.softmax(logits[0]/T, dim=-1)
            
            #* Sample/Select the next nodes
            next_nodes = self.sample_nodes(sampled_probs, prev_nodes, num_samples=self.topk_len, step=depth, min_accept_prob=self.min_accept_prob)

            #* Append nodes to their parent nodes
            for node in next_nodes:
                prev_nodes[node.ind].append(node)
 
            #* Get the nodes as input for next iteration
            next_nodes = [node for node in next_nodes if node.id != self.eos_token_id and node.global_prob > self.min_sample_prob] # don't sample nodes after eos_token_id
            prev_nodes = next_nodes
            
            #* Depth increment
            depth += 1
            
            #* Early stop if no nodes for next iteration
            # TODO: Also break if total_global_prob < threshold, where it does not benefit to continue
            if len(next_nodes) == 0:
                break
        
        #* Crop the tree to the max_candidate_tokens
        past_key_values.crop(org_input_len)
        
        return keep_top_n_nodes(root, self.max_tokens)


class SSM_Eagle(SSMBaseNEFT):
    def init_custom_model(self, config):
        config.org_hidden_size = config.hidden_size
        config.org_intermediate_size = config.intermediate_size
        if hasattr(config, "compress_hidden_ratio"):
            config.hidden_size = int(config.hidden_size * config.compress_hidden_ratio)
        if hasattr(config, "compress_intermediate_ratio"):
            config.intermediate_size = int(config.intermediate_size * config.compress_intermediate_ratio)
            
        return modeling_llama_eagle.LlamaModel(config)

    def init_additional_modules(self, config):
        self.fusion = MergeLinear(config)
        
    def forward(self, input_ids, hidden_states, embed_tokens=None, *model_args, **kwargs):
        with torch.no_grad():
            embed_tokens = self.model.get_input_embeddings() if embed_tokens is None else embed_tokens
            inputs_embeds = embed_tokens(input_ids)
        
        hidden_states = self.fusion(hidden_states, inputs_embeds)
        hidden_states = self.model(inputs_embeds=hidden_states, *model_args, **kwargs)[0]

        return hidden_states
    
    @torch.no_grad()
    def _update_tree_attention_data(self, depth, nodes, hidden_states, tree_mask, position_offset, device):
        indices = torch.tensor([node.ind for node in nodes])

        input_hidden = hidden_states[:, indices].to(device)
        
        input_ids = torch.tensor([node.id for node in nodes], device=device)[None]
        
        position_ids = torch.zeros(len(nodes), device=device)[None] + (position_offset + depth)
        
        # Generating tree masks for the new nodes, don't have to consider the old nodes
        tree_mask = tree_mask[:, :, indices]
        tree_mask = torch.concat((tree_mask, torch.eye(len(nodes), device=device, dtype=torch.bool)[None, None]), dim=3)

        return input_hidden, input_ids, position_ids, tree_mask
    
    @torch.no_grad()
    def speculate(self, input_ids, hidden_states, past_key_values, embed_tokens, lm_head, *model_args, **kwargs):
        """This method is used to draft/guess the next tokens that the LLM may generate.

        Args:
            input_ids (Tensor): The input token ids.
            hidden_states (Tensor): hidden_states from the previous iteration.
            past_key_values (Cache): Cache object to store the past key-values generated by the model.
            embed_tokens (Module): embedding from LLM.
            lm_head (Module): lm_head from LLM.

        Returns:
            Node: The root node of the generated draft token tree.
        """
        
        device = input_ids.device
        if hasattr(self.model, "lm_head"):
            dtype = self.model.lm_head.weight.dtype
        else:
            dtype = lm_head.weight.dtype
        
        # take out last token as sample_token
        sample_token = input_ids[:, -1:]
        
        # remove the first token from input_ids (input_ids is shifted by 1)
        input_ids = input_ids[:, 1:]
        
        # keep original length of input_ids
        org_input_len = input_ids.shape[1] # offset of positon_id
        
        # initialize tree_mask and tree 
        tree_mask = torch.ones([1, 1, 1, org_input_len], device=device, dtype=torch.bool)
        root = Node("1", id=sample_token[0][0].item(), prob=1, global_prob=1, ind=-1)
        
        depth = 1 # depth starts from 1 in tree library
        prev_nodes = [root]
        while depth < self.depth:
            #* Decode previous nodes
            if depth == 1: # first iteration
                kv_len = past_key_values.get_seq_length()
                hidden_states = self(
                    input_ids[:, kv_len:],
                    hidden_states=hidden_states,
                    embed_tokens=embed_tokens,
                    past_key_values=past_key_values,
                )[:, -1:]
                
            else:
                hidden_states, input_ids, position_ids, tree_mask = self._update_tree_attention_data(depth, prev_nodes, hidden_states, tree_mask, org_input_len, device=hidden_states.device)
                hidden_states = self(
                    input_ids,
                    hidden_states=hidden_states,
                    embed_tokens=embed_tokens, 
                    past_key_values=past_key_values,
                    position_ids=position_ids, 
                    attention_mask=invert_mask(tree_mask, dtype=dtype),
                )

            #* Get the probabilities of each token
            T = 1
            sampled_probs = torch.softmax(lm_head(hidden_states)[0]/T, dim=-1)
            
            #* Sample/Select the next nodes
            next_nodes = self.sample_nodes(sampled_probs, prev_nodes, num_samples=self.topk_len, step=depth, min_accept_prob=self.min_accept_prob)

            #* Append nodes to their parent nodes
            for node in next_nodes:
                prev_nodes[node.ind].append(node)
 
            #* Get the nodes as input for next iteration
            next_nodes = [node for node in next_nodes if node.id != self.eos_token_id and node.global_prob > self.min_sample_prob] # don't sample nodes after eos_token_id
            prev_nodes = next_nodes
            
            #* Depth increment
            depth += 1
            
            #* Early stop if no nodes for next iteration
            # TODO: Also break if total_global_prob < threshold, where it does not benefit to continue
            if len(next_nodes) == 0:
                break
        
        #* Crop the tree to the max_candidate_tokens
        past_key_values.crop(org_input_len)
        
        return keep_top_n_nodes(root, self.max_tokens)
    
class SSM_ShrinkClassic(SSM_Classic):
    def init_additional_modules(self, config):
        self.limited_vocab_size = 8192
        self.lm_head = nn.Linear(config.hidden_size, self.limited_vocab_size, bias=False)
        self.model.embed_tokens = nn.Embedding(self.limited_vocab_size, config.hidden_size, config.pad_token_id)
        
        self.id_ssm_to_llm = torch.load("/home/nctu/scott306lr/SpecDecodes/specdecodes/experiments/top_8192_id_map.pt", weights_only=True)
        self.id_llm_to_ssm = torch.load("/home/nctu/scott306lr/SpecDecodes/specdecodes/experiments/top_8192_id_map_inverse.pt", weights_only=True)
        self.id_llm_freq_map = torch.load('/home/nctu/scott306lr/SpecDecodes/specdecodes/experiments/top_8192_id_map_freq.pt', weights_only=True)
        
        
    def forward(self, input_ids, embed_tokens=None, return_logits=False, *model_args, **kwargs):
        # not using hidden_states
        _ = kwargs.pop("hidden_states", None)
        
        # convert input_ids to custom vocab
        self.id_llm_to_ssm = self.id_llm_to_ssm.to(input_ids.device)
        input_ids = self.id_llm_to_ssm[input_ids]
        
        inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = self.model(inputs_embeds=inputs_embeds, *model_args, **kwargs)[0]
    
        if not return_logits:
            return hidden_states
        else:
            return self.lm_head(hidden_states), hidden_states
        
    
class SSM_ShrinkEagle(SSM_Eagle):
    def init_additional_modules(self, config):
        self.limited_vocab_size = 8192
        self.model.embed_tokens = nn.Embedding(self.limited_vocab_size, config.hidden_size, config.pad_token_id)
        self.lm_head = nn.Linear(config.org_hidden_size, self.limited_vocab_size, bias=False)
        
        # self.extract = Extractor(config)
        self.extract = nn.Linear(config.org_hidden_size, config.hidden_size, bias=False)
        
        self.id_ssm_to_llm = torch.load("/home/nctu/scott306lr/SpecDecodes/specdecodes/experiments/top_8192_id_map.pt", weights_only=True)
        self.id_llm_to_ssm = torch.load("/home/nctu/scott306lr/SpecDecodes/specdecodes/experiments/top_8192_id_map_inverse.pt", weights_only=True)
        self.id_llm_freq_map = torch.load('/home/nctu/scott306lr/SpecDecodes/specdecodes/experiments/top_8192_id_map_freq.pt', weights_only=True)
        
        if config.hidden_size != config.org_hidden_size:
            self.resize = nn.Linear(config.hidden_size, config.org_hidden_size, bias=False)
        else:
            self.resize = nn.Identity()
        
    def forward(self, input_ids, hidden_states, embed_tokens=None, return_logits=False, *model_args, **kwargs):
        # convert input_ids to custom vocab
        self.id_llm_to_ssm = self.id_llm_to_ssm.to(hidden_states.device)
        input_ids = self.id_llm_to_ssm[input_ids]
        
        inputs_embeds = self.model.embed_tokens(input_ids).to(hidden_states.dtype)
        hidden_states = inputs_embeds + self.extract(hidden_states)
        hidden_states = self.model(inputs_embeds=hidden_states, *model_args, **kwargs)[0]
        hidden_states = self.resize(hidden_states)
        
        if not return_logits:
            return hidden_states
        else:
            return self.lm_head(hidden_states), hidden_states   
        
    @torch.no_grad()
    def speculate(self, input_ids, hidden_states, past_key_values, embed_tokens, lm_head, *model_args, **kwargs):
        """This method is used to draft/guess the next tokens that the LLM may generate.

        Args:
            input_ids (Tensor): The input token ids.
            hidden_states (Tensor): hidden_states from the previous iteration.
            past_key_values (Cache): Cache object to store the past key-values generated by the model.
            embed_tokens (Module): embedding from LLM.
            lm_head (Module): lm_head from LLM.

        Returns:
            Node: The root node of the generated draft token tree.
        """
        
        device = input_ids.device
        if hasattr(self.model, "lm_head"):
            dtype = self.model.lm_head.weight.dtype
        else:
            dtype = lm_head.weight.dtype
        
        # take out last token as sample_token
        sample_token = input_ids[:, -1:]
        
        # remove the first token from input_ids (input_ids is shifted by 1)
        input_ids = input_ids[:, 1:]
        
        # keep original length of input_ids
        org_input_len = input_ids.shape[1] # offset of positon_id
        
        # initialize tree_mask and tree 
        tree_mask = torch.ones([1, 1, 1, org_input_len], device=device, dtype=torch.bool)
        root = Node("1", id=sample_token[0][0].item(), prob=1, global_prob=1, ind=-1)
        
        # early exit if input token is limited_vocab_size-1
        self.id_llm_to_ssm = self.id_llm_to_ssm.to(input_ids.device)
        if self.id_llm_to_ssm[sample_token[0][0]] == self.limited_vocab_size-1:
            return root
        
        depth = 1 # depth starts from 1 in tree library
        prev_nodes = [root]
        while depth < self.depth:
            #* Decode previous nodes
            if depth == 1: # first iteration
                kv_len = past_key_values.get_seq_length()
                hidden_states = self(
                    input_ids[:, kv_len:],
                    hidden_states=hidden_states,
                    embed_tokens=embed_tokens,
                    past_key_values=past_key_values,
                )[:, -1:]
                
            else:
                hidden_states, input_ids, position_ids, tree_mask = self._update_tree_attention_data(depth, prev_nodes, hidden_states, tree_mask, org_input_len, device=hidden_states.device)
                hidden_states = self(
                    input_ids,
                    hidden_states=hidden_states,
                    embed_tokens=embed_tokens, 
                    past_key_values=past_key_values,
                    position_ids=position_ids, 
                    attention_mask=invert_mask(tree_mask, dtype=dtype),
                )
            
            logits = lm_head(hidden_states)[0]

            #* Get the probabilities of each token
            T = 1
            sampled_probs = torch.softmax(logits/T, dim=-1)
            
            #* Sample/Select the next nodes
            next_nodes = self.sample_nodes(sampled_probs, prev_nodes, num_samples=self.topk_len, step=depth, min_accept_prob=self.min_accept_prob)

            #* Append nodes to their parent nodes
            for node in next_nodes:
                # remap with self.id_ssm_to_llm
                node.id = self.id_ssm_to_llm[node.id]
                
                # ignore if node.id is limited_vocab_size-1
                if node.id != self.limited_vocab_size-1:
                    prev_nodes[node.ind].append(node)
 
            #* Get the nodes as input for next iteration
            next_nodes = [node for node in next_nodes if node.id != self.eos_token_id and node.global_prob > self.min_sample_prob] # don't sample nodes after eos_token_id
            prev_nodes = next_nodes
            
            #* Depth increment
            depth += 1
            
            #* Early stop if no nodes for next iteration
            # TODO: Also break if total_global_prob < threshold, where it does not benefit to continue
            if len(next_nodes) == 0:
                break
        
        #* Crop the tree to the max_candidate_tokens
        past_key_values.crop(org_input_len)
        
        return keep_top_n_nodes(root, self.max_tokens)