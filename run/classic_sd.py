from .app_router import run_app
from .base import BaseBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.classic_sd import ClassicSDDraftModel
from specdecodes.models.generators.classic_sd import ClassicSDGenerator

from specdecodes.helpers.recipes.recipe_4bit_mlp import recipe

class ClassicSDBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_model_path = "meta-llama/Llama-3.2-1B-Instruct"
        
        self.dtype = torch.float16
        self.device = "cuda:0"
        
        # Generator configurations
        self.generator_class = ClassicSDGenerator
        self.draft_params = DraftParams(
            max_depth=12,
            topk_len=16,
            max_verify_tokens=256,
            min_accept_prob=1e-8,
        )
        
        # Offloading
        # self.recipe = recipe
        # self.vram_limit = None # in GB
        
        # Speed up inference using torch.compile
        self.cache_implementation = "static"
        self.warmup_iter = 10
        self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
    
    def _load_draft_model(self, target_model=None, tokenizer=None, draft_path=None):
        draft_model = ClassicSDDraftModel.from_pretrained(
            draft_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            device_map=self.device,
            eos_token_id=tokenizer.eos_token_id
        )
        return draft_model
    
    
if __name__ == "__main__":
    run_app(ClassicSDBuilder())