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
from ..cpu_tree import Tree
     
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
    def __init__(self, model=None, config=None, eos_token_id=None, sampling_method='greedy', keep_embeddings=False, tree_depth=6+1, topk_len=5, min_sample_prob=1e-2, min_accept_prob=1e-2):
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
        tree_depth = model_kwargs.pop("tree_depth", 6+1)
        topk_len = model_kwargs.pop("topk_len", 5)
        min_sample_prob = model_kwargs.pop("min_sample_prob", 1e-2)
        min_accept_prob = model_kwargs.pop("min_accept_prob", 1e-2)
        
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
            model = cls(
                model=ssm, 
                config=config, 
                eos_token_id=eos_token_id, 
                sampling_method=sampling_method, 
                tree_depth=tree_depth,
                topk_len=topk_len,
                min_sample_prob=min_sample_prob,
                min_accept_prob=min_accept_prob,
                *model_args, 
                **model_kwargs
            ).to(dtype=torch_dtype)
        
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
    #         parent_idx = top_idx // sample_k
    #         child_idx  = top_idx % sample_k

    #     with nvtx.annotate("gather_ids"):
    #         # 4) Gather final token_ids
    #         rows = torch.arange(bsz, device=sampled_probs.device).unsqueeze(-1)
    #         token_ids = ptk_ids[rows, parent_idx, child_idx]

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
        tree_data = TreeData(
            root_id,
            sample_len=self.topk_len,
            max_sample_depth=self.max_depth,
            dtype=dtype,
            device=device,
        )
        
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
                tree_data.update(token_ids, child_probs, parent_indices)
                
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
        tree = Tree(root_id, dtype)
        tree.add_nodes(*tree_data.get_data())
        
        # Prune to top-n
        tree.prune_to_top_n(self.max_tokens)
        
        return tree


# torch.set_float32_matmul_precision('high')

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
        tree_data = TreeData(
            root_id,
            sample_len=self.topk_len,
            max_sample_depth=self.max_depth,
            dtype=dtype,
            device=device,
        )
        
        # 3) Initialize tree mask cache for draft model inference
        with nvtx.annotate("init tree mask cache"):
            tree_mask_cache = TreeMaskCache(
                prefix_len=org_input_len,
                sample_len=self.topk_len,
                max_sample_depth=self.max_depth,
                dtype=dtype,
                device=device,
            )

        # 4) Initialize parent probabilities & position ids
        with nvtx.annotate("init parent_probs & position_ids"):
            parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
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
                tree_data.update(token_ids, child_probs, parent_indices)
                
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
        tree = Tree(root_id, dtype)
        tree.add_nodes(*tree_data.get_data())
        
        # Prune to top-n
        tree.prune_to_top_n(self.max_tokens)
        
    def forward(self, input_ids, hidden_states, embed_tokens=None, return_logits=False, *model_args, **kwargs):
        # convert input_ids to custom vocab
        self.id_llm_to_ssm = self.id_llm_to_ssm.to(hidden_states.device)
        input_ids = self.id_llm_to_ssm[input_ids]
        
        inputs_embeds = self.model.embed_tokens(input_ids).to(hidden_states.dtype)
        hidden_states = inputs_embeds + self.extract(hidden_states) 
        hidden_states = self.model(inputs_embeds=hidden_states, *model_args, **kwargs)[0]
        
        if not return_logits:
            return hidden_states
        else:
            return self.lm_head(hidden_states), hidden_states

class SSM_SQ(SSMBase):
    def __init__(self, ssm, config, eos_token_id=None, torch_dtype=torch.float16, sampling_method="greedy", *model_args, **model_kwargs):
        super().__init__(ssm, config, eos_token_id, sampling_method, *model_args, **model_kwargs)

        self.budget = 64
        self.depth = 12
        self.topk_len = 16

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        torch_dtype=torch.float16,
        **model_kwargs
    ):
        ssm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)
        model = cls(ssm, config, torch_dtype=torch_dtype, *model_args, **model_kwargs)
        return model

    def load_spectree_arch(self, spectree_arch:list[int]):
        self.spectree_arch = spectree_arch

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

            if depth == 1:
                sampled_probs = sampled_probs[-1:, :]
                
            # * Sample / Select the next nodes
            # next_nodes = self.sample_nodes(sampled_probs, prev_nodes, num_samples=self.topk_len, step=depth)

            n, vocab_dim = sampled_probs.shape
            # print(f'depth: {depth}')
            # print(f'child num should be: {len(self.spectree_arch[depth-1])}')
            # print(f'n: {n}')

            last_node_id = int(prev_nodes[-1].name)
            next_nodes = []
            for prev_ind, child_node_num in enumerate(self.spectree_arch[depth-1]):
                if prev_ind >= n:
                    break
                prev_node = prev_nodes[prev_ind]
                prev_node.sample_probs = sampled_probs[prev_ind]
                prev_node.verify_method = "greedy"
                
                topk_values, topk_indices = torch.topk(sampled_probs[prev_ind], child_node_num)

                for child_idx in range(child_node_num):
                    global_prob = topk_values[child_idx]
                    prob = global_prob / prev_node.global_prob
                    last_node_id += 1
                    new_node = Node(str(last_node_id), id=topk_indices[child_idx].item(), prob=prob, global_prob=global_prob, ind=prev_ind)
                    next_nodes.append(new_node)

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

        # print(f' --------------- Tree Info ---------------')

        # def count_nodes(node):
        #     return 1 + sum(count_nodes(child) for child in node.children)

        # # Get the size of the tree
        # tree_size = count_nodes(root)
        # print(f"Tree Size: {tree_size}")

        # def calculate_depth(node):
        #     if not node.children:
        #         return 1
        #     return 1 + max(calculate_depth(child) for child in node.children)

        # # Get the depth of the tree
        # tree_depth = calculate_depth(root)
        # print(f"Tree Depth: {tree_depth}")

        return root

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
        sampling_method = model_kwargs.pop("sampling_method", "greedy")
        tree_depth = model_kwargs.pop("tree_depth", 11+1)
        topk_len = model_kwargs.pop("topk_len", 5)
        min_sample_prob = model_kwargs.pop("min_sample_prob", 1e-2)
        min_accept_prob = model_kwargs.pop("min_accept_prob", 1e-2)
        
        ssm, model_str = model_from_hf_path(pretrained_model_name_or_path, )
        model = cls(
            model=ssm, 
            config=config, 
            eos_token_id=eos_token_id, 
            sampling_method=sampling_method, 
            tree_depth=tree_depth,
            topk_len=topk_len,
            min_sample_prob=min_sample_prob,
            min_accept_prob=min_accept_prob,
            *model_args, 
            **model_kwargs
        )
        
        # Convert the model to the desired dtype and return
        model
        return model
