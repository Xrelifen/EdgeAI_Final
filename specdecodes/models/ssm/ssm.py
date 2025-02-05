import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_torchdynamo_compiling
from safetensors.torch import load_model
import os
from typing import List, Tuple
import nvtx

from .training_hooks import TrainingHook, NEFTuneHook
from ..llm import modeling_llama as modeling_llama
from ..llm import modeling_llama_no_inout_norm as modeling_llama_eagle
from ..utils.utils import invert_mask
from ..utils.cpu_tree import Tree
     
def load_custom_model(model, model_path, keep_embeddings=False):
    # Load the model
    missing_keys, unexpected_keys = load_model(model, model_path, strict=False)
    
    # Remove embed_tokens if not found (for custom models that uses LLM's embed_tokens)
    for key in missing_keys:
        if 'embed_tokens' in key:
            print(f"embed_tokens not found. Use LLM's embed_tokens instead.")
            if not keep_embeddings:
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
    def __init__(self, model=None, config=None, eos_token_id=None, keep_embeddings=False):
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
        
        # set prefill function same as forward so torch.compile() forward will not execute on prefill phase)
        #! Not needed after torch version=2.7, where torch.compiler.set_stance("force_eager") is introduced
        self.prefill_forward = self.forward
        
    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device
    
        
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path,
        *model_args,
        config = None,
        torch_dtype=torch.float32,
        keep_embeddings = False,
        **model_kwargs
    ):
        # Remove the following arguments from model_kwargs, cause AutoModelForCausalLM does not accept them
        eos_token_id = model_kwargs.pop("eos_token_id", None)
        
        # Load HuggingFace model if config is not provided
        if config is not None: 
            draft_model_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            model = cls(None, config=config, eos_token_id=eos_token_id, keep_embeddings=keep_embeddings, *model_args, **model_kwargs)
            model = load_custom_model(model, draft_model_path, keep_embeddings=keep_embeddings)
        
        else:
            ssm = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                *model_args, 
                **model_kwargs
            )
            model = cls(ssm, config=config, eos_token_id=eos_token_id, *model_args, **model_kwargs).to(dtype=torch_dtype)
        
        # Convert the model to the desired dtype and return
        model.to(dtype=torch_dtype)
        return model
        
    def init_custom_model(self, config):
        return modeling_llama.LlamaModel(config)
    
    def init_additional_modules(self, config):
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
                flattened_probs, sample_k, dim=1, sorted=False
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
    
    # @torch.no_grad()
    # def topk_sampling(
    #     self,
    #     sampled_probs: torch.Tensor,
    #     parent_probs: torch.Tensor,
    #     sample_k: int,
    #     sample_min_prob: float,
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    #     bsz, n_leaves, vocab = sampled_probs.shape
    #     sampled_probs = sampled_probs.contiguous()
    #     parent_probs  = parent_probs.contiguous()

    #     with nvtx.annotate("partial_topk"):
    #         # 1) Partial top-k over vocab dimension
    #         ptk_vals, ptk_ids = torch.topk(sampled_probs, k=sample_k, dim=-1)
    #         # 2) Multiply by parent_probs
    #         global_probs = ptk_vals * parent_probs.unsqueeze(-1)

    #     with nvtx.annotate("final_topk"):
    #         # 3) Flatten & final top-k
    #         flat_probs = global_probs.view(bsz, -1)
    #         top_probs, top_idx = torch.topk(flat_probs, sample_k, dim=1, sorted=False)
    #         parent_idx = (top_idx // sample_k).long()
    #         child_idx  = (top_idx % sample_k).long()

    #     with nvtx.annotate("gather_ids"):
    #         # 4) Gather final token_ids
    #         rows = torch.arange(bsz, device=sampled_probs.device).unsqueeze(-1)
    #         token_ids = ptk_ids[rows, parent_idx, child_idx].long()

    #     # Decide on valid_flag (simplified here)
    #     valid_flag = True  # or use top_probs.max(dim=1) > sample_min_prob, etc.

    #     return token_ids, top_probs, parent_idx, valid_flag


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

    
class SSM_Classic(SSMBaseNEFT):
    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits, dim=-1)
            
        return logits
    
    @torch.no_grad()
    def speculate(self, input_ids, past_key_values, **kwargs):
        # 1) Obtain necessary parameters
        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, org_input_len = input_ids.shape
        max_cache_len = getattr(past_key_values, "max_cache_len", None)
        kv_len = past_key_values.get_seq_length()
        assert batch_size == 1, "Only support batch_size=1 for now."
        
        # 2) Create Tree used for target model inference later
        root_id = input_ids[0, -1]
        tree_data = TreeData(
            root_id,
            sample_len=self.draft_params.topk_len,
            max_sample_depth=self.draft_params.max_depth,
            dtype=dtype,
            device=device,
        )
        
        # 3) Initialize tree mask cache for draft model inference
        with nvtx.annotate("init tree mask cache"):
            tree_mask_cache = TreeMaskCache(
                prefix_len=org_input_len,
                sample_len=self.draft_params.topk_len,
                max_cache_len=max_cache_len,
                dtype=dtype,
                device=device,
            )

        # 4) Initialize parent probabilities & position ids & cache_position
        with nvtx.annotate("init parent_probs & position_ids & cache_position"):
            parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            position_ids = torch.full((batch_size, self.draft_params.topk_len), org_input_len, device=device, dtype=torch.long)
            cache_position = torch.arange(kv_len, org_input_len, dtype=torch.long, device=device)

        # 5) First forward pass 
        with nvtx.annotate("ssm first forward", color="red"):
            sampled_probs = self.prefill_forward(
                input_ids[:, kv_len:],
                with_softmax=True,
                past_key_values=past_key_values,
                cache_position=cache_position,
                num_logits_to_keep=1,
            )
            kv_len = org_input_len
            
        with nvtx.annotate("update cache"):
            cache_position = torch.arange(org_input_len, org_input_len+self.draft_params.topk_len, dtype=torch.long, device=device)
        
        # 6) Main loop
        for depth_i in range(self.draft_params.max_depth):
            # --------------------------------------
            # A. Compute token distribution & Sample
            # --------------------------------------
            with nvtx.annotate("sample nodes", color="green"):
                token_ids, child_probs, parent_indices, valid_flag = self.topk_sampling(
                    sampled_probs,
                    parent_probs,
                    self.draft_params.topk_len, 
                    self.draft_params.min_accept_prob
                )
                parent_probs = child_probs
            
            # --------------------------------------
            # B. Early stop if all probs are below min_accept_prob (currently not used, introduces syncing stalls)
            # --------------------------------------
            # with nvtx.annotate("early stop"):
            #     if (depth_i % 3 == 0) and (depth_i > 0):
            #         valid_flag = sampled_probs.max() > self.draft_params.min_accept_prob
            #         if not valid_flag:
            #             print(f"Early stop at depth {depth_i}/{self.draft_params.max_depth}")
            #             break
            
            # --------------------------------------
            # C. Add new nodes to the CPU tree
            # --------------------------------------
            with nvtx.annotate("add nodes", color="green"):
                tree_data.update(token_ids, child_probs, parent_indices)
                
            with nvtx.annotate("position"):
                position_ids += 1
                
            with nvtx.annotate("tree mask"):
                tree_attention_mask = tree_mask_cache.update_tree_mask(parent_indices)
            
            with nvtx.annotate("ssm forward", color="red"):
                sampled_probs = self(
                    token_ids,
                    with_softmax=True,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=tree_attention_mask,
                    cache_position=cache_position,
                )
                kv_len += self.draft_params.topk_len
                
            with nvtx.annotate("update cache"):
                cache_position += self.draft_params.topk_len
        
        # Discard new calcs in KV cache after original input length
        with nvtx.annotate("crop kv"):
            past_key_values.crop(org_input_len, kv_len, dim=2)

        # Obtain the final tree
        with nvtx.annotate("tree related"):
            with nvtx.annotate("get tree data"):
                data = tree_data.get_data()
            with nvtx.annotate("build tree"):
                tree = Tree(root_id, dtype)
                tree.add_nodes(*data)
            with nvtx.annotate("prune tree"):
                tree.prune_to_top_n(self.draft_params.max_verify_tokens)
        
        return tree

# torch.set_float32_matmul_precision('high')

class SSM_Eagle(SSMBaseNEFT):
    def init_custom_model(self, config):
        return modeling_llama_eagle.LlamaModel(config)

    def init_additional_modules(self, config):
        self.fusion = MergeLinear(config.hidden_size*2, config.hidden_size)
        
    def set_modules(self, embed_tokens=None, lm_head=None):
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        if lm_head is not None:
            self.lm_head = lm_head
    
    def forward(self, input_ids, hidden_states, num_logits_to_keep=0, with_softmax=False, *model_args, **kwargs):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fusion(hidden_states, inputs_embeds)
        hidden_states = self.model(inputs_embeds=hidden_states, *model_args, **kwargs)[0][:, -num_logits_to_keep:]
        logits = self.lm_head(hidden_states)
        
        if with_softmax:
            logits = torch.softmax(logits, dim=-1)
            
        return logits, hidden_states
    
    @torch.no_grad()
    def speculate(self, input_ids, hidden_states, past_key_values, **kwargs):
        # 1-1) Remove the first token from input_ids (shift by 1)
        input_ids = input_ids[:, 1:]
        
        # 1-2) Obtain necessary parameters
        device = input_ids.device
        if hasattr(self.model, "lm_head"):
            dtype = self.model.lm_head.weight.dtype
        else:
            dtype = self.lm_head.weight.dtype
        batch_size, org_input_len = input_ids.shape
        max_cache_len = getattr(past_key_values, "max_cache_len", None)
        kv_len = past_key_values.get_seq_length()
        assert batch_size == 1, "Only support batch_size=1 for now."
        
        # 2) Create Tree used for target model inference later
        root_id = input_ids[0, -1]
        tree_data = TreeData(
            root_id,
            sample_len=self.draft_params.topk_len,
            max_sample_depth=self.draft_params.max_depth,
            dtype=dtype,
            device=device,
        )
        
        # 3) Initialize tree mask cache for draft model inference
        with nvtx.annotate("init tree mask cache"):
            tree_mask_cache = TreeMaskCache(
                prefix_len=org_input_len,
                sample_len=self.draft_params.topk_len,
                max_cache_len=max_cache_len,
                dtype=dtype,
                device=device,
            )

        # 4) Initialize parent probabilities & position ids & cache_position
        with nvtx.annotate("init parent_probs & position_ids & cache_position"):
            parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            position_ids = torch.full((batch_size, self.draft_params.topk_len), org_input_len, device=device, dtype=torch.long)
            cache_position = torch.arange(kv_len, org_input_len, dtype=torch.long, device=device)
            
        # 5) First forward pass
        with nvtx.annotate("ssm first forward", color="red"):
            sampled_probs, hidden_states = self.prefill_forward(
                input_ids[:, kv_len:],
                with_softmax=True,
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                num_logits_to_keep=1,
            )
            kv_len = org_input_len
        
        with nvtx.annotate("update cache"):
            cache_position = torch.arange(org_input_len, org_input_len+self.draft_params.topk_len, dtype=torch.long, device=device)

        # 6) Main loop
        for depth_i in range(self.draft_params.max_depth):
            # --------------------------------------
            # A. Compute token distribution & Sample
            # --------------------------------------
            with nvtx.annotate("sample nodes", color="green"):
                token_ids, child_probs, parent_indices, valid_flag = self.topk_sampling(
                    sampled_probs,
                    parent_probs,
                    self.draft_params.topk_len, 
                    self.draft_params.min_accept_prob
                )
                parent_probs = child_probs
            
            # --------------------------------------
            # B. Early stop if all probs are below min_accept_prob (currently not used, introduces syncing stalls)
            # --------------------------------------
            # with nvtx.annotate("early stop"):
            #     if (depth_i % 3 == 0) and (depth_i > 0):
            #         valid_flag = sampled_probs.max() > self.draft_params.min_accept_prob
            #         if not valid_flag:
            #             print(f"Early stop at depth {depth_i}/{self.draft_params.max_depth}")
            #             break
            
            # --------------------------------------
            # C. Add new nodes to the CPU tree
            # --------------------------------------
            with nvtx.annotate("add nodes", color="green"):
                tree_data.update(token_ids, child_probs, parent_indices)
                
            with nvtx.annotate("filter"):
                # Expand parent_indices to match hidden_states along the last dimension
                parent_indices_expanded = parent_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                hidden_states = torch.gather(hidden_states, dim=1, index=parent_indices_expanded)
                
            with nvtx.annotate("position"):
                position_ids += 1
            
            with nvtx.annotate("tree mask"):
                tree_attention_mask = tree_mask_cache.update_tree_mask(parent_indices)
            
            with nvtx.annotate("ssm forward", color="red"):
                sampled_probs, hidden_states = self(
                    token_ids,
                    with_softmax=True,
                    hidden_states=hidden_states,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=tree_attention_mask,
                    cache_position=cache_position,
                )
                kv_len += self.draft_params.topk_len
                
            with nvtx.annotate("update cache"):
                cache_position += self.draft_params.topk_len
        
        # Discard new calcs in KV cache after original input length
        with nvtx.annotate("crop kv"):
            past_key_values.crop(org_input_len, kv_len, dim=2)
             
        # Obtain the final tree   
        with nvtx.annotate("tree related"):
            with nvtx.annotate("get tree data"):
                data = tree_data.get_data()
            with nvtx.annotate("build tree"):
                tree = Tree(root_id, dtype)
                tree.add_nodes(*data)
            with nvtx.annotate("prune tree"):
                tree.prune_to_top_n(self.draft_params.max_verify_tokens)
        
        return tree

from .lib.utils.unsafe_import import model_from_hf_path
class SSM_QTIP(SSM_Classic):
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
        
        ssm, model_str = model_from_hf_path(pretrained_model_name_or_path, )
        model = cls(ssm, config=config, eos_token_id=eos_token_id, *model_args, **model_kwargs)
        
        return model
    
    
class SSM_ShareSD(SSMBaseNEFT):
    @classmethod
    def from_pretrained(
        cls, 
        base_model,
        *model_args,
        config = None,
        torch_dtype=torch.float32,
        keep_embeddings = False,
        **model_kwargs
    ):
        # Remove the following arguments from model_kwargs, cause AutoModelForCausalLM does not accept them
        eos_token_id = model_kwargs.pop("eos_token_id", None)

        model = cls(base_model, config=config, eos_token_id=eos_token_id, *model_args, **model_kwargs).to(dtype=torch_dtype)
        
        # Convert the model to the desired dtype and return
        model.to(dtype=torch_dtype)
        return model
    
    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits, dim=-1)
            
        return logits
    
    @torch.no_grad()
    def speculate(self, input_ids, past_key_values, *model_args, **kwargs):
        # 1) Obtain necessary parameters
        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, input_len = input_ids.shape
        max_cache_len = getattr(past_key_values, "max_cache_len", None)
        assert batch_size == 1, "Only support batch_size=1 for now."

        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("Initialize kv_len & cache_position"):
            kv_len = past_key_values.get_seq_length()
            # convert kv_len to int if it is a tensor
            if isinstance(kv_len, torch.Tensor):
                kv_len = kv_len.item()
            org_kv_len = kv_len
            cache_position = torch.arange(kv_len, kv_len+input_len, dtype=torch.long, device=device)
        
        # 3) First forward pass
        with nvtx.annotate("ssm first forward", color="red"):
            sampled_probs = self(
                input_ids,
                with_softmax=True,
                past_key_values=past_key_values,
                cache_position=cache_position,
                num_logits_to_keep=1,
            )
            kv_len += input_len

        with nvtx.annotate("update parent_probs & position_ids & cache_position"):
            parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            position_ids = torch.full((batch_size, self.draft_params.topk_len), kv_len+1, device=device, dtype=torch.long)
            cache_position = torch.arange(kv_len, kv_len+self.draft_params.topk_len, dtype=torch.long, device=device)
        
        # 4) Create TreeData & TreeMaskCache to manage tree structure and intermediate data.
        root_id = input_ids[0, -1]
        tree_data = TreeData(
            root_id,
            sample_len=self.draft_params.topk_len,
            max_sample_depth=self.draft_params.max_depth,
            dtype=dtype,
            device=device,
        )
        tree_mask_cache = TreeMaskCache(
            prefix_len=kv_len,
            sample_len=self.draft_params.topk_len,
            max_cache_len=max_cache_len,
            dtype=dtype,
            device=device,
        )

        # 5) Main loop
        for depth_i in range(self.draft_params.max_depth):
            # --------------------------------------
            # A. Compute token distribution & Sample
            # --------------------------------------
            with nvtx.annotate("sample nodes", color="green"):
                token_ids, child_probs, parent_indices, valid_flag = self.topk_sampling(
                    sampled_probs,
                    parent_probs,
                    self.draft_params.topk_len, 
                    self.draft_params.min_accept_prob
                )
                parent_probs = child_probs
            
            # --------------------------------------
            # B. Early stop if all probs are below min_accept_prob (currently not used, introduces syncing stalls)
            # --------------------------------------
            # with nvtx.annotate("early stop"):
            #     if (depth_i % 3 == 0) and (depth_i > 0):
            #         valid_flag = sampled_probs.max() > self.draft_params.min_accept_prob
            #         if not valid_flag:
            #             print(f"Early stop at depth {depth_i}/{self.draft_params.max_depth}")
            #             break
            
            # --------------------------------------
            # C. Add new nodes to the CPU tree
            # --------------------------------------
            with nvtx.annotate("add nodes", color="green"):
                tree_data.update(token_ids, child_probs, parent_indices)
                
            with nvtx.annotate("tree mask"):
                tree_attention_mask = tree_mask_cache.update_tree_mask(parent_indices)
            
            with nvtx.annotate("ssm forward", color="red"):
                sampled_probs = self(
                    token_ids,
                    with_softmax=True,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=tree_attention_mask,
                    cache_position=cache_position,
                )
                kv_len += self.draft_params.topk_len
                
            with nvtx.annotate("update position_ids & cache"):
                position_ids += 1
                cache_position += self.draft_params.topk_len
        
        # Discard new calcs in KV cache after original input length
        with nvtx.annotate("crop kv"):
            past_key_values.crop(org_kv_len, kv_len, dim=2)

        # Obtain the final tree
        with nvtx.annotate("tree related"):
            with nvtx.annotate("get tree data"):
                data = tree_data.get_data()
            with nvtx.annotate("build tree"):
                tree = Tree(root_id, dtype)
                tree.add_nodes(*data)
            with nvtx.annotate("prune tree"):
                tree.prune_to_top_n(self.draft_params.max_verify_tokens)
        
        return tree
