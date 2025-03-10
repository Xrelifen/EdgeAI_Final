import logging
from ..app_router import run_app
from ..base import BaseBuilder

import torch
from specdecodes.models.generators.naive import NaiveGenerator

from specdecodes.helpers.recipes.offload.recipe_llama_8b_offload_8gb import recipe
from specdecodes.helpers.offloaders.prefetch_offloader_v3 import PrefetchOffloader

class NaiveBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16
        # self.max_length = 256
        
        # Load model configurations
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.generator_class = NaiveGenerator
        
        # Sample configurations
        self.do_sample = False
        self.temperature = 0
        
        # Quantization and offloading
        self.recipe = recipe
        self.offloader = PrefetchOffloader
        
        # Speed up inference using torch.compile
        self.cache_implementation = "static"
        # self.warmup_iter = 2
        # self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
        
    def _compile_generator(self, generator, compile_mode):
        logging.info(f"Compiling the generator with mode: {compile_mode}")
        generator.target_model.forward = torch.compile(generator.target_model.forward, mode=compile_mode, dynamic=False, fullgraph=True)
        
if __name__ == "__main__":
    run_app(NaiveBuilder())