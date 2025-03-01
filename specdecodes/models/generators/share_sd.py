import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin

class ShareSDGeneratorBase(ClassicSDGeneratorBase):
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
            4. Update the key-value cache and input_ids accordingly.

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
            
        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values, "max_cache_len", None)
        else:
            raise ValueError("past_key_values is not provided")
            
        
        self._init_tree_mask(self.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device)
        cache_position = torch.arange(org_input_len, dtype=torch.long, device=input_ids.device)

        # * prefill stage
        with nvtx.annotate("prefill", color="orange"):
            outputs = self.target_model.prefill_forward(
                input_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                logits_to_keep=1,
            )
            next_token_logits = outputs.logits
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
                    last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                    tree = self._speculate(last_token_id, past_key_values)

                # * tree decoding
                with nvtx.annotate("tree_decoding", color="orange"):
                    prev_kv_len = past_key_values.get_seq_length()
                    outputs = self._tree_decoding(tree, past_key_values, position_offset=input_ids.shape[1]-1, cache_position=cache_position, device=input_ids.device)
                    next_token_logits = outputs.logits
                    del outputs

                # * verify
                with nvtx.annotate("verify"):
                    sampled_tokens, hidden_indices, (total_len, accept_len) = self._verify(
                                                            tree, next_token_logits, 
                                                            logits_processor,
                                                            do_sample
                                                        )
                    sampled_tokens = sampled_tokens.to(input_ids.device, non_blocking=True)
                    
                    # if total_len < 6:
                    #     tree.print_tree_structure(tokenizer=self.tokenizer)
                
                with nvtx.annotate("reorder kv"):
                    past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, new_chunk_len=self.draft_params.max_verify_tokens, dim=2)

                # * update input_ids and cache_position
                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    cache_position += sampled_tokens.shape[1]
                
                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    finished = stopping_criteria(input_ids, None).item()
                
        return input_ids
    
class ShareSDGenerator(SDProfilingMixin, ShareSDGeneratorBase):
    pass