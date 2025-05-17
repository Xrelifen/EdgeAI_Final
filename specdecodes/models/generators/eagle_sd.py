import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.utils import DraftParams, invert_mask


class EagleSDGeneratorBase(ClassicSDGeneratorBase):
    def _speculate(self, input_ids, hidden_states, past_key_values):
        return self.draft_model.speculate(
            input_ids,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
        )

    def _tree_decoding(self, tree, past_key_values, position_offset, cache_position, device):
        # Preparing target_model's tree decoding data, also updates each node's index (node.ind).
        with nvtx.annotate("create attn mask"):
            node_data = tree.get_node_data()
            tree_input_ids = node_data['token_ids']
            tree_position_ids = node_data['depths'] + position_offset
            tree_mask_partial = tree.create_attention_mask(position_offset)
        
        # Move to device
        with nvtx.annotate("mask to GPU"):
            tree_input_ids = tree_input_ids.to(device, non_blocking=True)
            tree_position_ids = tree_position_ids.to(device, non_blocking=True)
            tree_mask_partial = tree_mask_partial.to(device)
            torch.cuda.synchronize()
        
        # Assing to tree mask
        with nvtx.annotate("update mask"):
            tree_mask = self._update_tree_mask(tree_mask_partial)
            tree_mask = invert_mask(tree_mask, dtype=self.target_model.model.dtype)
        
        # llm forward
        #TODO: Remove unnecessary squeeze(0) and unsqueeze(0) operations
        with nvtx.annotate("llm forward", color="red"):
            outputs = self.target_model(
                tree_input_ids.unsqueeze(0),
                past_key_values=past_key_values,
                attention_mask=tree_mask,
                position_ids=tree_position_ids.unsqueeze(0),
                output_hidden_states=True,
                cache_position=cache_position
            )
        return outputs
    
    def _verify_step(self, p, q, token_ids, do_sample):
        sampled_token_id = p.argmax() if not do_sample else p.multinomial(1).squeeze(-1)
        if torch.any(sampled_token_id == token_ids):
            return sampled_token_id, None
        
        denom = 1.0 - p[token_ids].sum()
        p.div_(denom) if denom >= 1e-9 else p.zero_() # numerical stability
        p[token_ids].zero_()
        return None, p

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """
        Generate sequence of tokens with speculative decoding.

        This method consists of two main stages: prefill and decode.

        Prefill Stage:
        - Perform the model's initial forward pass.
        - Sample a token and append it to the input_ids.

        Decode Stage (with speculative decoding):
        - Iterate through the following steps:
            1. Perform SSM speculative sampling, returns sampled tokens in tree form.
            2. Decode the sampled tokens in parallel with the language model (LLM), generating probabilities for each token.
            3. Verify the sampled tokens by accepting or rejecting them, corresponding to the probabilities.
            4. Update the key-value cache, input_ids, and hidden_states accordingly.

        Args:
            input_ids (torch.LongTensor): The input token IDs. 
            stopping_criteria (StoppingCriteria): The criteria to stop the generation.
            logits_processor (LogitsProcessor): The processor to modify the logits.
            do_sample (bool): Whether to sample tokens during generation. If False, the generation will be deterministic.

        Returns:
            input_ids (torch.LongTensor): The generated token IDs.
        """
        assert self.target_model is not None, "target_model must be provided"
        assert self.draft_model is not None, "draft_model must be provided"
        assert self.tokenizer is not None, "tokenizer must be provided"

        # * clone input_ids 
        input_ids = input_ids.clone()
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # * prepare kv-cache
        # Raise error if max_length not set while using static cache
        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )
            
        if model_kwargs.get("past_key_values") is not None and model_kwargs.get("draft_past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values, "max_cache_len", None)
            
            draft_past_key_values = model_kwargs["draft_past_key_values"]
        else:
            raise ValueError("past_key_values and draft_past_key_values should both be provided")
        
        
        self._init_tree_mask(self.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device)
        cache_position = torch.arange(org_input_len, dtype=torch.long, device=input_ids.device)

        # * prefill stage
        with nvtx.annotate("prefill", color="orange"):
            outputs = self.target_model.prefill_forward(
                input_ids,
                past_key_values=past_key_values, 
                output_hidden_states=True, 
                cache_position=cache_position,
                logits_to_keep=1,
            )
            next_token_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            del outputs

        with nvtx.annotate("sample tokens"):
            sampled_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("update data"):
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            cache_position = torch.arange(org_input_len, org_input_len+self.draft_params.max_verify_tokens, dtype=torch.long, device=input_ids.device)

        with nvtx.annotate("decoding"):
            finished = False
            while not finished:
                # * speculate
                with nvtx.annotate("speculate", color="cyan"):
                    tree = self._speculate(input_ids, hidden_states, draft_past_key_values)

                # * tree decoding
                with nvtx.annotate("tree_decoding", color="orange"):
                    prev_kv_len = past_key_values.get_seq_length()
                    outputs = self._tree_decoding(tree, past_key_values, position_offset=input_ids.shape[1]-1, cache_position=cache_position, device=hidden_states.device)
                    next_token_logits = outputs.logits
                    hidden_states = outputs.hidden_states[-1]
                    del outputs

                # * verify
                with nvtx.annotate("verify"):
                    sampled_tokens, hidden_indices, (total_len, accept_len) = self._verify(
                                                        tree, next_token_logits, 
                                                        logits_processor,
                                                        do_sample
                                                    )
                    
                    sampled_tokens = sampled_tokens.to(input_ids.device, non_blocking=True)
                    hidden_indices = hidden_indices.to(hidden_states.device, non_blocking=True)
                
                with nvtx.annotate("reorder kv"):
                    past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, new_chunk_len=self.draft_params.max_verify_tokens, dim=2)

                # * update input_ids, hidden_states, and cache_position
                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    hidden_states = hidden_states[:, hidden_indices].clone()
                    cache_position += sampled_tokens.shape[1]
                
                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    finished = stopping_criteria(input_ids, None).item()
                
        return input_ids

    
class EagleSDGenerator(SDProfilingMixin, EagleSDGeneratorBase):
    def forward(self, *args, **kwargs):
        return self.target_model(*args, **kwargs)
    pass