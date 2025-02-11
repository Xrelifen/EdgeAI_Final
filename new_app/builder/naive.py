from ..app import run_app

import torch
from specdecodes.models import ProfileNaiveWrapper

class NaiveBuilder:
    def __init__(self):
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Load model configurations
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.generator_class = ProfileNaiveWrapper
        
        # Sample configurations
        self.do_sample = False
        self.temperature = 0
        
        # Speed up inference using torch.compile
        self.cache_implementation = "dynamic"
        self.warmup_iter = 0
        self.compile_mode = None
        
if __name__ == "__main__":
  run_app(NaiveBuilder())