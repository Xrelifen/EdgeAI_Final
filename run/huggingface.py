from .app_router import run_app
from .base import BaseBuilder

import torch
from specdecodes.models.generators.huggingface import HuggingFaceGenerator

class NaiveBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Load model configurations
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.generator_class = HuggingFaceGenerator
        
        # Sample configurations
        self.do_sample = False
        self.temperature = 0
        
        # Speed up inference using torch.compile
        self.cache_implementation = "dynamic"
        self.warmup_iter = 0
        self.compile_mode = None
        
        # Profiling
        self.generator_profiling = True
        self.nvtx_profiling = False
        
if __name__ == "__main__":
    run_app(NaiveBuilder())