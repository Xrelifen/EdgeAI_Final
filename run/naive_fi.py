from .app_router import run_app
from .base import BaseBuilder

import torch
from specdecodes.models.generators.naive import NaiveGenerator
from specdecodes.models.generators.naive_fi import NaiveFIGenerator


class NaiveBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Load model configurations
        self.llm_path = "meta-llama/Llama-2-7b-chat-hf"
        self.generator_class = NaiveFIGenerator
       
        
        # Sample configurations
        self.do_sample = False
        self.temperature = 0
        
        # Speed up inference using torch.compile
        self.cache_implementation = "flashinfer"
        self.warmup_iter = 5
        self.compile_mode = None
        
        # Profiling
        self.generator_profiling = True
        
if __name__ == "__main__":
    run_app(NaiveBuilder())