from .app_router import run_app
from .base import BaseBuilder

import torch
from specdecodes.models.generators.naive import NaiveGenerator

class NaiveBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Load model configurations
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.generator_class = NaiveGenerator
        
        # Sample configurations
        self.do_sample = False
        self.temperature = 0
        
        # Speed up inference using torch.compile
        self.cache_implementation = "static"
        self.warmup_iter = 10
        self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
        
if __name__ == "__main__":
    run_app(NaiveBuilder())