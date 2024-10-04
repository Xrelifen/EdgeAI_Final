import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_model
import os

from bigtree import Node

from .sampling_utils import topk_sampling, k_sampling, heuristic_k_sampling, mixed_k_sampling
from ..utils import invert_mask

from ..llm import modeling_llama_no_init_weights as modeling_llama
from ..llm import modeling_llama_no_layernorm
from ..llm import modeling_llama_shared_kv

class SSMBase(nn.Module):
    def __init__(self, model=None, config=None, eos_token_id=None, sampling_method='greedy', *model_args, **model_kwargs):
        super().__init__()
        
        # Set model and config
        if model is not None and config is not None:
            raise ValueError("Only one of model or config must be provided.")   
        
        elif model is not None:
            self.model = model
            self.config = model.config
            
        elif config is not None:
            self.model = self.init_custom_model(config)
            self.config = config
            
        else:
            raise ValueError("Either model or config must be provided.")
        
        # Set other attributes
        self.eos_token_id = eos_token_id
        self.init_sampling_method(sampling_method)
         
        self.depth = 9 + 1 # 6 + 1 # 9 + 1 
        self.topk_len = 15 # 5 # 15
        self.min_sample_prob = 1e-8
        
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
            load_model(model, draft_model_path, strict=True)
        
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
    
    def init_sampling_method(self, sampling_method):
        if sampling_method == 'greedy':
            self.sample_nodes = topk_sampling
        elif sampling_method == 'stochastic':
            self.sample_nodes = k_sampling
        elif sampling_method == 'hstochastic':
            self.sample_nodes = heuristic_k_sampling
        elif sampling_method == 'mixed':
            def mixed_sampling(sampling_probs, nodes, num_samples, step):
                SWITCH_STEP = 2
                if step <= SWITCH_STEP:
                    return topk_sampling(sampling_probs, nodes, num_samples, step)
                else:
                    return heuristic_k_sampling(sampling_probs, nodes, num_samples, step)
            self.sample_nodes = mixed_sampling
        else:
            raise ValueError("Sampling method not supported")
    
    @torch.no_grad()
    def forward(self, inputs, *model_args, **kwargs):
        raise NotImplementedError
    
    @torch.no_grad()
    def speculate(self, inputs, past_key_values, **kwargs):
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


class SSM_Classic(SSMBase):
    def __init__(self, model=None, config=None, eos_token_id=None, sampling_method='greedy', *model_args, **model_kwargs):
        super().__init__(model=model, config=config, eos_token_id=eos_token_id, sampling_method=sampling_method, *model_args, **model_kwargs)
    
    def forward(self, input_ids, *model_args, **kwargs):
        _ = kwargs.pop("embed_tokens", None)
        return self.model(input_ids, *model_args, **kwargs)
    
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
    def speculate(self, inputs, past_key_values, embed_tokens, lm_head):
        """This method is used to draft/guess the next tokens that the LLM may generate.

        Args:
            inputs (list): A list of two tensors: hidden_states and input_ids.
            past_key_values (Cache): Cache object to store the past key-values generated by the model.
            embed_tokens (Module): embedding from LLM.
            lm_head (Module): lm_head from LLM.

        Returns:
            Node: The root node of the generated draft token tree.
        """
        [_, input_ids] = inputs
        
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
                    past_key_values=past_key_values
                )
                if hasattr(self.model, "lm_head"):
                    logits = outputs.logits[:, -1:].clone()
                else:
                    logits = lm_head(outputs.last_hidden_state[:, -1:]).float()
            else:
                input_ids, position_ids, tree_mask = self._update_tree_attention_data(depth, prev_nodes, tree_mask, org_input_len, device=logits.device)
                outputs = self(
                    input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids, 
                    attention_mask=invert_mask(tree_mask, dtype=dtype)
                )
                if hasattr(self.model, "lm_head"):
                    logits = outputs.logits
                else:
                    logits = lm_head(outputs.last_hidden_state).float()

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            #* Get the probabilities of each token
            T = 1
            sampled_probs = torch.softmax(logits[0]/T, dim=-1)
            
            #* Sample/Select the next nodes
            next_nodes = self.sample_nodes(sampled_probs, prev_nodes, num_samples=self.topk_len, step=depth)

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
        
        return root
    

class SSM_Eagle(SSMBase):
    def __init__(self, model=None, config=None, eos_token_id=None, sampling_method='greedy', *model_args, **model_kwargs):
        super().__init__(model=model, config=config, eos_token_id=eos_token_id, sampling_method=sampling_method, *model_args, **model_kwargs)
        
        self.fc = nn.Linear(self.config.hidden_size*2, self.config.hidden_size, bias=True)
        if hasattr(self.model, "embed_tokens"):
            del self.model.embed_tokens
    
    def init_custom_model(self, config):
        return modeling_llama_no_layernorm.LlamaModel(config)
        
    def forward(self, inputs, *model_args, **kwargs):
        [hidden_states, input_ids] = inputs
        embed_tokens = kwargs.pop("embed_tokens", None)
        
        with torch.no_grad():
            inputs_embeds = embed_tokens(input_ids).to(hidden_states.dtype)
            
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        return self.model(inputs_embeds=hidden_states, *model_args, **kwargs)
    
    @torch.no_grad()
    def _update_tree_attention_data(self, depth, nodes, hidden_states, tree_mask, position_offset):
        device = hidden_states.device 
        indices = torch.tensor([node.ind for node in nodes])

        input_hidden = hidden_states[:, indices].to(device)
        
        input_ids = torch.tensor([node.id for node in nodes], device=device)[None]
        
        position_ids = torch.zeros(len(nodes), device=device)[None] + (position_offset + depth)
        
        # Generating tree masks for the new nodes, don't have to consider the old nodes
        tree_mask = tree_mask[:, :, indices]
        tree_mask = torch.concat((tree_mask, torch.eye(len(nodes), device=device, dtype=torch.bool)[None, None]), dim=3)

        return input_hidden, input_ids, position_ids, tree_mask
    
    @torch.no_grad()
    def speculate(self, inputs, past_key_values, embed_tokens, lm_head):
        """This method is used to draft/guess the next tokens that the LLM may generate.

        Args:
            inputs (list): A list of two tensors: hidden_states and input_ids.
            past_key_values (Cache): Cache object to store the past key-values generated by the model.
            embed_tokens (Module): embedding from LLM.
            lm_head (Module): lm_head from LLM.

        Returns:
            Node: The root node of the generated draft token tree.
        """
        [hidden_states, input_ids] = inputs
        device = hidden_states.device
        input_ids = input_ids.to(device)
        
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
                outputs = self(
                    inputs=[
                        hidden_states,
                        input_ids[:, kv_len:]
                        ],
                    embed_tokens=embed_tokens,
                    past_key_values=past_key_values
                )
                hidden_states = outputs.last_hidden_state[:, -1:].clone() # Only the last token's hidden state is needed.
            else:
                hidden_states, input_ids, position_ids, tree_mask = self._update_tree_attention_data(depth, prev_nodes, hidden_states, tree_mask, org_input_len)
                outputs = self(
                    inputs=[
                        hidden_states, 
                        input_ids
                        ],
                    embed_tokens=embed_tokens, 
                    past_key_values=past_key_values,
                    position_ids=position_ids, 
                    attention_mask=invert_mask(tree_mask, dtype=hidden_states.dtype)
                )
                hidden_states = outputs.last_hidden_state

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            #* Get the probabilities of each token
            T = 1
            sampled_probs = torch.softmax(lm_head(hidden_states)[0]/T, dim=-1)
            
            #* Sample/Select the next nodes
            next_nodes = self.sample_nodes(sampled_probs, prev_nodes, num_samples=self.topk_len, step=depth)

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
        
        return root


class SSM_SharedKV(SSM_Classic):
    def __init__(self, model=None, config=None, eos_token_id=None, sampling_method='greedy', *model_args, **model_kwargs):
        super().__init__(model=model, config=config, eos_token_id=eos_token_id, sampling_method=sampling_method, *model_args, **model_kwargs)
    
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
            load_model(model, draft_model_path, strict=True)
        
        else:
            ssm = modeling_llama_shared_kv.LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                **model_kwargs
            )
            model = cls(ssm, config=config, eos_token_id=eos_token_id, sampling_method=sampling_method, *model_args, **model_kwargs).to(dtype=torch_dtype)
        
        # Convert the model to the desired dtype and return
        model.to(dtype=torch_dtype)
        return model
        
    def init_custom_model(self, config):
        return modeling_llama_shared_kv.LlamaModel(config)
    
    @torch.no_grad()
    def speculate(self, inputs, past_key_values, past_key_values_llm, embed_tokens, lm_head):
        """This method is used to draft/guess the next tokens that the LLM may generate.

        Args:
            inputs (list): A list of two tensors: hidden_states and input_ids.
            past_key_values (Cache): Cache object to store the past key-values generated by the model.
            embed_tokens (Module): embedding from LLM.
            lm_head (Module): lm_head from LLM.

        Returns:
            Node: The root node of the generated draft token tree.
        """
        [_, input_ids] = inputs
        
        device = input_ids.device
        if hasattr(self.model, "lm_head"):
            dtype = self.model.lm_head.weight.dtype
        else:
            dtype = lm_head.weight.dtype
        
        # take out last token as sample_token
        sample_token = input_ids[:, -1:]
        
        # keep original length of input_ids
        org_llm_kv_len = past_key_values_llm.get_seq_length()
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
                    past_key_values=past_key_values,
                    past_key_values_llm=past_key_values_llm
                )
                if hasattr(self.model, "lm_head"):
                    logits = outputs.logits[:, -1:].clone()
                else:
                    logits = lm_head(outputs.last_hidden_state[:, -1:]).float()
            else:
                input_ids, position_ids, tree_mask = self._update_tree_attention_data(depth, prev_nodes, tree_mask, org_input_len, device=logits.device)
                outputs = self(
                    input_ids,
                    past_key_values=past_key_values,
                    past_key_values_llm=past_key_values_llm,
                    position_ids=position_ids, 
                    attention_mask=invert_mask(tree_mask, dtype=dtype)
                )
                if hasattr(self.model, "lm_head"):
                    logits = outputs.logits
                else:
                    logits = lm_head(outputs.last_hidden_state).float()

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            #* Get the probabilities of each token
            T = 1
            sampled_probs = torch.softmax(logits[0]/T, dim=-1)
            
            #* Sample/Select the next nodes
            next_nodes = self.sample_nodes(sampled_probs, prev_nodes, num_samples=self.topk_len, step=depth)

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
        past_key_values_llm.crop(org_llm_kv_len)
        
        return root

class SSM_SX(SSMBase):
    def __init__(self, ssm_path, config, eos_token_id=None, torch_dtype=torch.float16):
        super().__init__(config, eos_token_id)

        self.model = AutoModelForCausalLM.from_pretrained(ssm_path, torch_dtype=torch_dtype)
        self.verify_method = "stochastic"
        self.budget = 64
        self.depth = 10
        self.topk_len = 16

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path,
        *model_args,
        config,
        torch_dtype=torch.float16,
        **model_kwargs
    ):

        model = cls(pretrained_model_name_or_path, config, torch_dtype=torch_dtype, *model_args, **model_kwargs)
        return model

    @torch.no_grad()
    def _sample_probs(
        self,
        logits: torch.FloatTensor,
        logits_warper,
        do_sample: bool,
        T=1.0,
    ):
        if do_sample:
            batch, seq_len, vocab_size = logits.shape
            
            logits = logits.view(-1, vocab_size)
            next_token_scores = logits_warper(None, logits)
            probs = torch.softmax(next_token_scores/T, dim=-1)
            return probs.view(batch, seq_len, vocab_size) # preserve shape
        
        else:
            return torch.softmax(logits/T, dim=-1)    

    @torch.no_grad()
    def _update_tree_attention_data(self, depth, nodes, tree_mask, position_offset, device="cuda:0"):
        indices = torch.tensor([node.ind for node in nodes])
        input_ids = torch.tensor([node.id for node in nodes], device=device)[None]
        
        position_ids = torch.zeros(len(nodes), device=device)[None] + (position_offset + depth)
        
        # Generating tree masks for the new nodes, don't have to consider the old nodes
        tree_mask = tree_mask[:, :, indices]
        tree_mask = torch.concat((tree_mask, torch.eye(len(nodes), device=device, dtype=torch.bool)[None, None]), dim=3)

        return input_ids, position_ids, tree_mask

    @torch.no_grad()
    def forward(self, input_ids, past_key_values, attention_mask=None, position_ids=None):
        outputs = self.model(
            input_ids, 
            past_key_values=past_key_values, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
        )

        return outputs

    @torch.no_grad()
    def _sample_nodes(self, sampled_probs, prev_nodes, num_samples, step):
        next_nodes = k_sampling(sampled_probs, prev_nodes, num_samples, step)
        return next_nodes

    @torch.no_grad()
    def speculate(self, input_ids, past_key_values):
        """This method is used to draf/guess the next tokens that the LLM may generate.

        Args:
            inputs (list): A list of two tensors: hidden_states and input_ids.
            past_key_values (Cache): Cache object to store the past key-values generated by the model.

        Returns:
            Node: The root node of the generated draft token tree.
        """       

        device = self.model.device
        dtype = self.model.dtype
        input_ids = input_ids.to(device)

        # take out last token as sample_token
        sample_token = input_ids[:, -1:]

        # keep original length of input_ids
        org_input_len = input_ids.shape[1] # offset of position_id

        # initialize tree_mask and tree
        tree_mask = torch.ones([1, 1, 1, org_input_len], device=device, dtype=torch.bool)
        root = Node(str(sample_token[0][0].item()), id=sample_token[0][0].item(), prob=1, global_prob=1, ind=-1)

        depth = 1
        prev_nodes = [root]
        while depth < self.depth:
            # * Decode previous nodes
            if depth == 1:
                kv_len = past_key_values.get_seq_length()
                outputs = self(
                    input_ids[:, kv_len:],
                    past_key_values=past_key_values,
                )
            else:
                # TODO: update_tree_attention
                input_ids, position_ids, tree_mask = self._update_tree_attention_data(depth, prev_nodes, tree_mask, org_input_len, device=device)
                outputs = self(
                    input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=invert_mask(tree_mask, dtype=dtype),
                )
        
            # * Get probabilities of each token
            sampled_probs = torch.softmax(outputs.logits, dim=-1)
        
            if sampled_probs.dim() == 3:
                sampled_probs = sampled_probs.squeeze(0)

            del outputs

            # * Sample / Select the next nodes
            next_nodes = self._sample_nodes(sampled_probs, prev_nodes, num_samples=self.topk_len, step=depth)
            # root.show()

            # * depth increment
            depth += 1

            # * Append nodes to their parent nopdes
            for node in next_nodes:
                prev_nodes[node.ind].append(node)

            # * Get the nodes as input for next iteration
            next_nodes = [node for node in next_nodes if node.id != self.eos_token_id] # don't sample nodes after eos_token_id
            prev_nodes = next_nodes

            # TODO: Break if total_global_prob < threahold, where it does not benefit to continue
            if len(next_nodes) == 0:
                break

        #* Crop the tree to the max_candidate_tokens
        past_key_values.crop(org_input_len)

        return root