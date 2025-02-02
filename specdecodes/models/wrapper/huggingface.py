import logging
import torch
from .base import WrapperBase

class HuggingFaceWrapper(WrapperBase):
    def __init__(self, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
    
    def generate(
        self, 
        input_ids: torch.LongTensor, 
        temperature=None, top_p=None, top_k=None, 
        max_length=2048, do_sample=True, 
        *args,
        **kwargs
    ):
        assert self.llm is not None, "LLM model must be provided"
        
        if self.cache_implementation == "dynamic":
            self.llm.generation_config.cache_implementation = None
        else:
            self.llm.generation_config.cache_implementation = self.cache_implementation
        return self.llm.generate(
            input_ids=input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=max_length,
            do_sample=do_sample,
            *args,
            **kwargs,
        )
        
class ProfileHuggingFaceWrapper(HuggingFaceWrapper):
    def __init__(self, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.exp_log = {}
        self.disable_logging = False

    def generate(self, input_ids, *model_args, **kwargs):
        if self.disable_logging:
            return super().generate(input_ids, *model_args, **kwargs)
        
        # run generation
        org_input_len = len(input_ids[0])
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        input_ids = super().generate(input_ids, *model_args, **kwargs)
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