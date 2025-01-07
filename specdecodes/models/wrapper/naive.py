import logging
import time
import torch
from .base import WrapperBase

from transformers.generation.logits_process import LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteria

from transformers.cache_utils import StaticCache, DynamicCache

class NaiveWrapper(WrapperBase):
    def __init__(self, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
 
    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
    ):
        assert self.llm is not None, "LLM model must be provided"
        
        # * clone input_ids
        input_ids = input_ids.clone()

        # * prepare kv-cache and cache position
        llm_past_key_values = self.create_kv_cache(
            max_cache_len=stopping_criteria.max_length,
            max_batch_size=1,
            config=self.llm.model.config,
            device=input_ids.device,
            dtype=self.llm.model.dtype,
        )
        cache_position = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        
        # * prefill stage
        outputs = self.llm(input_ids, past_key_values=llm_past_key_values, return_dict=True, cache_position=cache_position)
        cache_position = cache_position[-1:] + 1
        
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        # We keep the seq_len axis considering cases of multiple tokens.
        next_token_logits = outputs.logits[:, -1:, :].clone() # hf uses outputs.logits[:, -1, :].clone() here

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs
        
        next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        finished = False
        while not finished:
            outputs = self.llm(input_ids[:, -1:], past_key_values=llm_past_key_values, return_dict=True, cache_position=cache_position)
            cache_position += 1
        
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # We keep the seq_len axis considering cases of multiple tokens.
            next_token_logits = outputs.logits.clone()
            
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
            
            next_tokens = self._sample_token(next_token_logits, logits_warper, do_sample)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Stopping criteria
            finished = stopping_criteria(input_ids, None)
            
        return input_ids
    
class ProfileNaiveWrapper(NaiveWrapper):
    def __init__(self, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.exp_log = {}

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_warper: LogitsWarper,
        do_sample: bool,
    ):
        # run generation
        org_input_len = len(input_ids[0])
        start_time = time.perf_counter()
        input_ids = super()._generate(input_ids, stopping_criteria, logits_warper, do_sample)
        end_time = time.perf_counter()
        
        self.exp_log['n_tokens'] = len(input_ids[0][org_input_len:])
        self.exp_log['tput'] = len(input_ids[0][org_input_len:]) / (end_time-start_time)
        logging.info(
            f"Generated {self.exp_log['n_tokens']} tokens in {end_time-start_time:.2f}s, throughput: {self.exp_log['tput']:.2f} tokens/s"
        )
            
        return input_ids