import logging
from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.subspec_sd import SubSpecSDDraftModel
from specdecodes.models.generators.subspec_sd import SubSpecSDGenerator

from specdecodes.helpers.recipes.subspec.hqq_4bit_attn_4bit_mlp import Recipe

class SubSpecSDBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Device and precision settings.
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Model paths.
        # self.llm_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        
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
        self.warmup_iter = 5
        self.compile_mode = "max-autotune"
        
        # Profiling.
        self.generator_profiling = True
    
    def load_draft_model(self, target_model, tokenizer, draft_model_path):
        draft_model = SubSpecSDDraftModel.from_pretrained(
            draft_model_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id
        )
        return draft_model
    
    def compile_generator(self, generator):
        logging.info(f"Compiling generator with mode: {self.compile_mode}")
        # generator.target_model.forward = torch.compile(generator.target_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
        if getattr(generator, 'draft_model', None) is not None:
            generator.draft_model.forward = torch.compile(generator.draft_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
    
if __name__ == "__main__":
    run_app(SubSpecSDBuilder())