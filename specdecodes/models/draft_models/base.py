from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.utils import is_torchdynamo_compiling
from safetensors.torch import load_model
from typing import List, Tuple
import logging
import os
import nvtx

from ..utils.utils import invert_mask


def load_custom_model(model, model_path, remove_embeddings=False):
    # Load the model
    missing_keys, unexpected_keys = load_model(model, model_path, strict=False)
    
    # Remove embed_tokens if not found (for custom models that uses LLM's embed_tokens)
    for key in missing_keys:
        if 'embed_tokens' in key:
            print(f"embed_tokens not found. Use LLM's embed_tokens instead.")
            if remove_embeddings:
                del model.model.embed_tokens
    missing_keys = [key for key in missing_keys if 'embed_tokens' not in key]
    
    # error handling
    assert len(missing_keys) == 0 and len(unexpected_keys) == 0, f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
    
    return model

#TODO: Compare this implementation's speed using torch.compile
# class TreeData(nn.Module):
#     def __init__(self, root_id: int, sample_len: int, max_sample_depth: int, dtype: str, device: str):
#         super().__init__()
#         self.root_id = root_id
#         self.sample_len = sample_len
#         self.max_sample_depth = max_sample_depth
#         self.dtype = dtype
#         self.device = device
        
#         self.token_ids_data = torch.zeros(
#             [1, max_sample_depth, sample_len],
#             device=device,
#             dtype=torch.long,
#         )
#         self.child_probs_data = torch.zeros(
#             [1, max_sample_depth, sample_len],
#             device=device,
#             dtype=dtype,
#         )
#         self.parent_indices_data = torch.zeros(
#             [1, max_sample_depth, sample_len],
#             device=device,
#             dtype=torch.long,
#         )   
#         if not is_torchdynamo_compiling():
#             # Mark the buffer's address as static for optimization purposes
#             torch._dynamo.mark_static_address(self.token_ids_data)
#             torch._dynamo.mark_static_address(self.child_probs_data)
#             torch._dynamo.mark_static_address(self.parent_indices_data)
            
#         self.current_depth = 0
        
#     def update(self, token_ids: torch.Tensor, child_probs: torch.Tensor, parent_indices: torch.Tensor) -> torch.Tensor:
#         self.token_ids_data[:, self.current_depth].copy_(token_ids)
#         self.child_probs_data[:, self.current_depth].copy_(child_probs)
#         self.parent_indices_data[:, self.current_depth].copy_(parent_indices)
#         self.current_depth += 1
    
#     def get_data(self):
#         return (self.token_ids_data, self.child_probs_data, self.parent_indices_data)

class TreeData(nn.Module):
    def __init__(self, root_id: int, sample_len: int, max_sample_depth: int, dtype: str, device: str):
        super().__init__()
        self.root_id = root_id
        self.sample_len = sample_len
        self.max_sample_depth = max_sample_depth
        self.dtype = dtype
        self.device = device
        
        self.token_ids_data = []
        self.child_probs_data = []
        self.parent_indices_data = []
        
    def update(self, token_ids: torch.Tensor, child_probs: torch.Tensor, parent_indices: torch.Tensor) -> torch.Tensor:
        self.token_ids_data.append(token_ids)
        self.child_probs_data.append(child_probs)
        self.parent_indices_data.append(parent_indices)
    
    def get_data(self):
        self.token_ids_data = torch.cat(self.token_ids_data, dim=0).unsqueeze(0)
        self.child_probs_data = torch.cat(self.child_probs_data, dim=0).unsqueeze(0)
        self.parent_indices_data = torch.cat(self.parent_indices_data, dim=0).unsqueeze(0)
        
        return (self.token_ids_data, self.child_probs_data, self.parent_indices_data)


class TreeMaskCache:
    def __init__(self, prefix_len: int, sample_len: int, max_cache_len: int, dtype: str, device: str):
        self.prefix_len = prefix_len
        self.sample_len = sample_len
        self.max_cache_len = max_cache_len
        self.dtype = dtype
        self.device = device

        # build static tree_mask
        if self.max_cache_len is not None:
            self.tree_mask_update_method = 'static'
            self.tree_mask_cache = torch.zeros(
                (1, 1, self.sample_len, self.max_cache_len),
                device=self.device,
                dtype=torch.bool
            )
            if not is_torchdynamo_compiling():
                # Mark the buffer's address as static for optimization purposes
                torch._dynamo.mark_static_address(self.tree_mask_cache)
            
            # Initialize the first `prefix_len` elements to True
            self.tree_mask_cache[:, :, 0, :self.prefix_len] = True
            self.current_len = self.prefix_len
            
        # build dynamic tree_mask instead
        else:
            self.tree_mask_update_method = 'dynamic'
            self.tree_mask_cache = torch.ones(
                (1, 1, 1, self.prefix_len),
                device=self.device,
                dtype=torch.bool
            )

        # Create an identity block for later use
        self.eye_block = torch.eye(self.sample_len, device=self.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    
    def update_tree_mask(self, parent_indices: torch.Tensor) -> torch.Tensor:
        if self.tree_mask_update_method == 'static': # static tree mask update
            # Update existing mask based on parent indices
            self.tree_mask_cache[..., :self.current_len] = self.tree_mask_cache[..., parent_indices[0], :self.current_len]
            # Append the eye_block to the mask
            self.tree_mask_cache[..., self.current_len:self.current_len + self.sample_len] = self.eye_block
            # Update the current length
            self.current_len += self.sample_len
        else: 
            # Dynamically expand the mask by concatenating the eye_block
            tree_mask = self.tree_mask_cache[:, :, parent_indices[0]]
            self.tree_mask_cache = torch.concat((tree_mask, self.eye_block), dim=3)
        
        # Invert the mask and return
        return invert_mask(self.tree_mask_cache, dtype=self.dtype)

class DraftModelBase(nn.Module):
    def __init__(self, base_model=None, target_model=None, eos_token_id=None, *model_args, **model_kwargs):
        super().__init__()
        self.eos_token_id = eos_token_id
        
        # Set model and config
        if base_model is not None and target_model is not None:
            raise ValueError("Only one of model or config must be provided.")   
        elif base_model is not None:
            self.model = base_model
        elif target_model is not None:
            self.model = self.init_base_model(target_model)
        else:
            raise ValueError("Either model or config must be provided.")

        # Initialize additional modules if needed
        self.init_additional_modules()
        
        # Set prefill function same as forward. 
        # prefill_forward() is used for prefill phase that cannot torch.compile()
        self.prefill_forward = self.forward
        
    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device
    
    @property
    def config(self):
        return self.model.config
            
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        target_model=None,
        torch_dtype=torch.float32,
        use_hf_eagle:bool=False,
        remove_embeddings=False,
        **model_kwargs
    ):
        eos_token_id = model_kwargs.pop("eos_token_id", None)

        # load local safetensors
        if os.path.exists(pretrained_model_name_or_path):
            logging.info(f"Loading local Eagle model from {pretrained_model_name_or_path}")
            draft_model_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            model = cls(target_model=target_model, eos_token_id=eos_token_id, *model_args, **model_kwargs)
            load_custom_model(model, draft_model_path, remove_embeddings=remove_embeddings)

        # HF Eagle safetensors
        elif use_hf_eagle:
            logging.info(f"Downloading Eagle safetensors from HF repo {pretrained_model_name_or_path}")
            draft_model_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="model.safetensors",
                repo_type="model",
            )
            model = cls(target_model=target_model, eos_token_id=eos_token_id, *model_args, **model_kwargs)
            load_custom_model(model, draft_model_path, remove_embeddings=remove_embeddings)

        else:
            # AutoModelForCausalLM branch ex.llama3.2-1b
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                *model_args,
                **model_kwargs
            )
            model = cls(base_model, eos_token_id=eos_token_id, *model_args, **model_kwargs).to(dtype=torch_dtype)

        model.to(dtype=torch_dtype)
        return model
        
    def init_base_model(self, target_model):
        raise NotImplementedError
    
    def init_additional_modules(self):
        pass
    
    def update_modules(self, **kwargs):
        pass
        
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
        sample_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # batch_size, N_available_leaves = parent_probs.shape
        batch_size, N_available_leaves, vocab_size = sampled_probs.shape
        
        #! Test
        # if sample_k == 1:
        #     device = sampled_probs.device
        #     return sampled_probs.argmax(dim=-1), sampled_probs.max(dim=-1).values, torch.zeros(batch_size, dtype=torch.long, device=device)[None], True
        

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
                flattened_probs, sample_k, dim=1, sorted=True
            )  # Both shape: [sample_k]

        with nvtx.annotate("sampling_3"):
            # Compute parent indices
            parent_indices = (topk_indices // vocab_size).long()  # Shape: [sample_k]
        
        with nvtx.annotate("sampling_4"):
            # Compute token ids
            token_ids = (topk_indices % vocab_size).long()  # Shape: [sample_k]

        return token_ids, topk_probs, parent_indices