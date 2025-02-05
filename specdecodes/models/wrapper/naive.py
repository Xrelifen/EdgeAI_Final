import logging
import torch
from .base import WrapperBase

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria

import nvtx

class NaiveWrapper(WrapperBase):
    def __init__(self, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
 
    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
    ):
        assert self.llm is not None, "LLM model must be provided"
        
        # * clone input_ids
        input_ids = input_ids.clone()
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # * prepare kv-cache and cache position
        # Raise error if max_length not set while using static cache
        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )
                
        if self.cache_implementation == "dynamic":
            llm_max_cache_len = None
            llm_past_key_values = self.create_kv_cache("dynamic")
            
        elif self.cache_implementation == "static":
            llm_max_cache_len = stopping_criteria.max_length
            llm_past_key_values = self.create_kv_cache(
                "static",
                max_cache_len=llm_max_cache_len,
                max_batch_size=batch_size,
                config=self.llm.model.config,
                device=input_ids.device,
                dtype=self.llm.model.dtype,
            )
            
        cache_position = torch.arange(org_input_len, dtype=torch.long, device=input_ids.device)
        
        # * prefill stage
        with nvtx.annotate("prefill", color="orange"):
            outputs = self.llm.prefill_forward(
                input_ids,
                max_cache_len=llm_max_cache_len,
                past_key_values=llm_past_key_values,
                cache_position=cache_position,
                num_logits_to_keep=1,
            )
            next_token_logits = outputs.logits
        
        with nvtx.annotate("sample tokens"):
            next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)
        
        with nvtx.annotate("update data"):
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            cache_position = cache_position[-1:] + 1

        with nvtx.annotate("decoding"):
            finished = False
            while not finished:
                with nvtx.annotate("llm forward", color="orange"):
                    outputs = self.llm(
                        next_tokens, 
                        past_key_values=llm_past_key_values,
                        position_ids=cache_position.unsqueeze(0), 
                        cache_position=cache_position,
                    )
                    next_token_logits = outputs.logits
                
                # * update input_ids and cache_position
                with nvtx.annotate("sample tokens"):
                    next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)
                
                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    cache_position += 1
                
                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    finished = stopping_criteria(input_ids, None)
            
        return input_ids
    
class ProfileNaiveWrapper(NaiveWrapper):
    def __init__(self, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.exp_log = {}
        
        self.disable_logging = False

    def _generate(self, input_ids: torch.LongTensor, *model_args, **kwargs):
        if self.disable_logging:
            return super()._generate(input_ids, *model_args, **kwargs)
        
        # run generation
        org_input_len = len(input_ids[0])
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        input_ids = super()._generate(input_ids, *model_args, **kwargs)
        end_event.record()
        
        # Make sure all CUDA ops have finished before measuring
        torch.cuda.synchronize()
        
        # Elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = elapsed_time_ms / 1000.0
        
        self.exp_log['n_tokens'] = len(input_ids[0][org_input_len:])
        self.exp_log['tput'] = len(input_ids[0][org_input_len:]) / elapsed_time_s
        logging.info(
            f"Generated {self.exp_log['n_tokens']} tokens in {elapsed_time_s:.2f}s, throughput: {self.exp_log['tput']:.2f} tokens/s"
        )
            
        return input_ids