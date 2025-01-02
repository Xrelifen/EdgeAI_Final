import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_torchdynamo_compiling
from safetensors.torch import load_model
import math, os
from typing import List, Tuple
from torch import Tensor
import nvtx

from .training_hooks import TrainingHook, NEFTuneHook

from ..utils import invert_mask
from ..llm import modeling_llama_no_init_weights as modeling_llama
from ..llm import modeling_llama_no_inout_norm as modeling_llama_eagle
from ..cpu_tree import TreeBuilderWorker
     
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
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.fc = nn.Linear(in_shape, out_shape, bias=True)

    def forward(self, x, emb):
        # swapped (x, emb) to (emb, x) to match official implementation of Eagle
        return self.fc(torch.cat((emb, x), dim=-1))

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
            self.config = config
        else:
            raise ValueError("Either model or config must be provided.")
        
        # Initialize additional modules
        self.init_additional_modules(config)
        
        # Draft parameters
        self.init_draft_parameters()
        
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
        self.max_depth = 6 + 1
        self.topk_len = 10
        self.min_accept_prob = 1e-2 #! Not used
        self.max_tokens = 64
        
    def get_input_embeddings(self):
        # If the model has input embeddings, return it. Otherwise, return None
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        else:
            return None
        
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
        
    @torch.no_grad()
    def topk_sampling(
        self,
        sampled_probs: torch.Tensor, 
        parent_probs: torch.Tensor, 
        sample_k: int,
        sample_min_prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # batch_size, N_available_leaves = parent_probs.shape
        batch_size, N_available_leaves, vocab_size = sampled_probs.shape

        with nvtx.annotate("sampling_0"):
            # Ensure input tensors are contiguous (Not sure if this is needed)
            sampled_probs = sampled_probs.contiguous()
            parent_probs = parent_probs.contiguous()

        with nvtx.annotate("sampling_1"):
            # Expand the sampled_probs to [batch_size, N_available_leaves, vocab_size]
            global_probs = sampled_probs * parent_probs.unsqueeze(-1)

        with nvtx.annotate("sampling_2"):
            # Flatten the global_probs to [N_available_leaves * vocab_size]
            flattened_probs = global_probs.view(batch_size, -1)  # Shape: [N_available_leaves * vocab_size]

        with nvtx.annotate("sampling_3"):
            # Perform top-k sampling
            topk_probs, topk_indices = torch.topk(
                flattened_probs, sample_k, dim=1, sorted=True#False#largest=True, sorted=True
            )  # Both shape: [sample_k]
        
        with nvtx.annotate("sampling_4"):
            # Check if there is any probs above min_prob threshold. If not, valid_flag will be False
            # valid_flag = topk_probs.max() > sample_min_prob
            valid_flag = True

        with nvtx.annotate("sampling_5"):
            # Compute parent indices
            parent_indices = (topk_indices // vocab_size).long()  # Shape: [sample_k]
        
        with nvtx.annotate("sampling_6"):
            # Compute token ids
            token_ids = (topk_indices % vocab_size).long()  # Shape: [sample_k]

        return token_ids, topk_probs, parent_indices, valid_flag

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

class TreeMaskCache(nn.Module):
    def __init__(self, prefix_len: int, sample_len: int, max_sample_depth: int, dtype: str, device: str):
        super().__init__()
        self.prefix_len = prefix_len
        self.sample_len = sample_len
        self.max_sample_depth = max_sample_depth
        self.max_cache_len = prefix_len + sample_len * max_sample_depth
        self.dtype = dtype
        self.device = device
        
        self.tree_mask_cache = torch.zeros(
            [1, 1, self.sample_len, self.max_cache_len],
            device=device,
            dtype=torch.bool,
        )
        if not is_torchdynamo_compiling():
            # Mark the buffer's address as static for optimization purposes
            torch._dynamo.mark_static_address(self.tree_mask_cache)
        
        # set the first prefix_len elements to True
        self.tree_mask_cache[:, :, 0, :self.prefix_len] = True
        
        # set the current length to prefix_len
        self.current_len = prefix_len

        # create an eye block for later use
        self.eye_block = torch.eye(self.sample_len, device=device, dtype=torch.bool)[None, None]
        
        
    def update_tree_mask(self, parent_indices: torch.Tensor) -> torch.Tensor:
        self.tree_mask_cache[..., :self.current_len] = self.tree_mask_cache[..., parent_indices[0], :self.current_len]
        self.tree_mask_cache[..., self.current_len:self.current_len+self.sample_len] = self.eye_block
        self.current_len += self.sample_len
        
        return invert_mask(self.tree_mask_cache[..., :self.current_len], dtype=self.dtype)

    
class SSM_Classic(SSMBaseNEFT):
    def forward(self, input_ids, *model_args, **kwargs):
        # not using hidden_states
        _ = kwargs.pop("hidden_states", None) #! Refactor this

        logits = self.model(input_ids, *model_args, **kwargs).logits
        return logits
    
    @torch.no_grad()
    def speculate(self, input_ids, past_key_values, embed_tokens, lm_head, *model_args, **kwargs):
        # 1) Obtain necessary parameters
        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Currently only handling batch_size=1 for simplicity"
        
        # 2) Create Tree used for target model inference later
        root_id = input_ids[0, -1]
        thread_worker = TreeBuilderWorker(root_id=root_id, prob_dtype=dtype)
        thread_worker.start()
        
        # 3) Initialize tree mask cache for draft model inference
        tree_mask_cache = TreeMaskCache(
            prefix_len=org_input_len,
            sample_len=self.topk_len,
            max_sample_depth=self.max_depth,
            dtype=dtype,
            device=device,
        )

        # 4) Initialize parent probabilities & position ids
        parent_probs = torch.tensor([[1.0]], device=device, dtype=dtype)
        position_ids = torch.full((batch_size, self.topk_len), org_input_len, device=device, dtype=torch.long)

        # 5) First forward pass
        with nvtx.annotate("first forward", color="red"):
            kv_len = past_key_values.get_seq_length()
            logits = self(
                input_ids[:, kv_len:],
                past_key_values=past_key_values,
            )[:, -1:] # keep only the last hidden state

        with nvtx.annotate("softmax"):
            sampled_probs = torch.softmax(logits, dim=-1)
        
        # 6) Main loop
        for depth_i in range(1, self.max_depth):
            # --------------------------------------
            # A. Compute token distribution & Sample
            # --------------------------------------
            with nvtx.annotate("sample nodes", color="green"):
                token_ids, child_probs, parent_indices, valid_flag = self.topk_sampling(
                    sampled_probs,
                    parent_probs,
                    self.topk_len, 
                    self.min_accept_prob
                )
                parent_probs = child_probs
            
            # --------------------------------------
            # B. Early stop if all probs are below min_accept_prob (currently not used, introduces syncing stalls)
            # --------------------------------------
            # with nvtx.annotate("early stop"):
            #     # if depth_i > 3:
            #     valid_flag = sampled_probs.max() > self.min_sample_prob
            #     if not valid_flag:
            #         print(f"Early stop at depth {depth_i}/{self.max_depth}")
            #         break
            
            # --------------------------------------
            # C. Add new nodes to the CPU tree
            # --------------------------------------
            with nvtx.annotate("add nodes", color="green"):
                worker_expansion = (token_ids, child_probs, parent_indices)
                thread_worker.put_expansion(worker_expansion)
                
            with nvtx.annotate("position"):
                position_ids += 1
                
            with nvtx.annotate("tree mask"):
                tree_attention_mask = tree_mask_cache.update_tree_mask(parent_indices)
            
            with nvtx.annotate("forward", color="red"):
                logits = self(
                    token_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=tree_attention_mask,
                    *model_args,
                    **kwargs
                )

            with nvtx.annotate("softmax"):
                sampled_probs = torch.softmax(logits, dim=-1)
        
        # Discard new calcs in KV cache after original input length
        with nvtx.annotate("crop kv"):
            past_key_values.crop(org_input_len)

        # Obtain the final tree
        thread_worker.close_queue()
        thread_worker.join()
        tree = thread_worker.get_tree()
        
        # Prune to top-n
        tree.prune_to_top_n(self.max_tokens)
        
        return tree


class SSM_Eagle(SSMBaseNEFT):
    def init_custom_model(self, config):
        return modeling_llama_eagle.LlamaModel(config)

    def init_additional_modules(self, config):
        self.fusion = MergeLinear(config.hidden_size*2, config.hidden_size)

    def forward(self, input_ids, hidden_states, embed_tokens=None, *model_args, **kwargs):
        # with torch.no_grad():
        if self.get_input_embeddings():
            embed_tokens = self.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)
        
        hidden_states = self.fusion(hidden_states, inputs_embeds)
        hidden_states = self.model(inputs_embeds=hidden_states, *model_args, **kwargs)[0]
        return hidden_states
    
    @torch.no_grad()
    def speculate(self, input_ids, hidden_states, past_key_values, embed_tokens, lm_head, *model_args, **kwargs):
        # 1-1) Obtain necessary parameters
        device = input_ids.device
        if hasattr(self.model, "lm_head"):
            dtype = self.model.lm_head.weight.dtype
        else:
            dtype = lm_head.weight.dtype
        
        # 1-2) Remove the first token from input_ids (shift by 1)
        input_ids = input_ids[:, 1:]
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Currently only handling batch_size=1 for simplicity"
        
        # 2) Create Tree used for target model inference later
        root_id = input_ids[0, -1]
        thread_worker = TreeBuilderWorker(root_id=root_id, prob_dtype=dtype)
        thread_worker.start()
        
        # 3) Initialize tree mask cache for draft model inference
        tree_mask_cache = TreeMaskCache(
            prefix_len=org_input_len,
            sample_len=self.topk_len,
            max_sample_depth=self.max_depth,
            dtype=dtype,
            device=device,
        )

        # 4) Initialize parent probabilities & position ids
        parent_probs = torch.tensor([[1.0]], device=device, dtype=dtype)
        position_ids = torch.full((batch_size, self.topk_len), org_input_len, device=device, dtype=torch.long)

        # 5) First forward pass
        with nvtx.annotate("first forward", color="red"):
            kv_len = past_key_values.get_seq_length()
            hidden_states = self(
                input_ids[:, kv_len:],
                hidden_states=hidden_states,
                embed_tokens=embed_tokens,
                past_key_values=past_key_values,
            )[:, -1:] # keep only the last hidden state

        with nvtx.annotate("softmax"):
            sampled_probs = torch.softmax(lm_head(hidden_states), dim=-1)
        
        # 6) Main loop
        for depth_i in range(1, self.max_depth):
            # --------------------------------------
            # A. Compute token distribution & Sample
            # --------------------------------------
            with nvtx.annotate("sample nodes", color="green"):
                token_ids, child_probs, parent_indices, valid_flag = self.topk_sampling(
                    sampled_probs,
                    parent_probs,
                    self.topk_len, 
                    self.min_accept_prob
                )
                parent_probs = child_probs
            
            # with nvtx.annotate("early stop"):
            #     # if depth_i > 3:
            #     valid_flag = sampled_probs.max() > self.min_sample_prob
            #     if not valid_flag:
            #         print(f"Early stop at depth {depth_i}/{self.max_depth}")
            #         break
            
            # --------------------------------------
            # B. Add new nodes to the CPU tree
            # --------------------------------------
            with nvtx.annotate("add nodes", color="green"):
                worker_expansion = (token_ids, child_probs, parent_indices)
                thread_worker.put_expansion(worker_expansion)
                
            with nvtx.annotate("filter"):
                # Expand parent_indices to match hidden_states along the last dimension
                parent_indices_expanded = parent_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                hidden_states = torch.gather(hidden_states, dim=1, index=parent_indices_expanded)

            with nvtx.annotate("position"):
                position_ids += 1
            
            with nvtx.annotate("tree mask"):
                tree_attention_mask = tree_mask_cache.update_tree_mask(parent_indices)
            
            with nvtx.annotate("forward", color="red"):
                hidden_states = self(
                    token_ids,
                    hidden_states=hidden_states,
                    embed_tokens=embed_tokens,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=tree_attention_mask,
                    *model_args,
                    **kwargs
                )

            with nvtx.annotate("softmax"):
                sampled_probs = torch.softmax(lm_head(hidden_states), dim=-1)
        
        # Discard new calcs in KV cache after original input length
        with nvtx.annotate("crop kv"):
            past_key_values.crop(org_input_len)

        # Obtain the final tree
        thread_worker.close_queue()
        thread_worker.join()
        tree = thread_worker.get_tree()
        
        # Prune to top-n
        tree.prune_to_top_n(self.max_tokens)
        
        return tree