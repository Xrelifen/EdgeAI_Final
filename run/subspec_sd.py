import logging
from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.utils.cache_utils import create_kv_cache
from specdecodes.models.draft_models.subspec_sd import SubSpecSDDraftModel
from specdecodes.models.generators.subspec_sd import SubSpecSDGenerator

from specdecodes.helpers.recipes.subspec.higgs_4bit_attn_4bit_mlp import Recipe
# from specdecodes.helpers.recipes.subspec.higgs_no_offload import Recipe


class SubSpecSDBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Device and precision settings.
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.max_length = 1024
        
        # Model paths.
        # self.llm_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        # self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.llm_path = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_class = SubSpecSDGenerator
        self.draft_params = DraftParams(
            temperature=0.3,
            max_depth=32,
            topk_len=8,
            max_verify_tokens=1024
        )
        
        # Recipe for quantization and offloading.
        self.recipe = Recipe()
        self.cpu_offload_gb = None
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 3
        # self.compile_mode = "max-autotune"
        
        # Profiling.
        self.generator_profiling = False
    
    def load_draft_model(self, target_model, tokenizer, draft_model_path):
        draft_model = SubSpecSDDraftModel.from_pretrained(
            draft_model_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id
        )
        return draft_model
    
    def load_kv_cache(self, target_model, draft_model):
        if self.cache_implementation == "static":
            if self.max_length is not None:
                # Additional sample tokens may cause KV-Cache tp exceed max_length, share with draft model.
                max_cache_len = self.max_length + self.draft_params.max_sample_tokens
            else:
                raise ValueError("max_length should be set for static cache.")
            
            # Create static kv-cache
            past_key_values = create_kv_cache(
                "static",
                max_cache_len=max_cache_len,
                max_batch_size=1,
                config=target_model.model.config,
                device=self.device,
                dtype=target_model.model.dtype,
            )
        else:
            # Create dynamic kv-cache
            past_key_values = create_kv_cache("dynamic")
            
        # Target model shares cache with draft model.
        draft_past_key_values = None
        
        return past_key_values, draft_past_key_values
    
    def compile_generator(self, generator):
        logging.info(f"Compiling generator with mode: {self.compile_mode}")
        # generator.target_model.forward = torch.compile(generator.target_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
        if getattr(generator, 'draft_model', None) is not None:
            generator.draft_model.forward = torch.compile(generator.draft_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
    
if __name__ == "__main__":
    run_app(SubSpecSDBuilder())