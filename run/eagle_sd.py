from .app_router import run_app
from .base import BaseBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.eagle_sd import EagleSDDraftModel
from specdecodes.models.generators.eagle_sd import EagleSDGenerator

class EagleSDBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        self.llm_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.draft_model_path = "/home/scott306lr_l/checkpoints/eagle/official/EAGLE-LLaMA3-Instruct-8B/"
        
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Generator configurations
        self.generator_class = EagleSDGenerator
        self.draft_params = DraftParams(
            max_depth=6,
            topk_len=10,
            max_verify_tokens=64,
            min_accept_prob=1e-8,
        )
        
        # Offloading
        # self.recipe = temp_recipe
        
        # Speed up inference using torch.compile
        self.cache_implementation = "static"
        self.warmup_iter = 10
        self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
    
    def _load_draft_model(self, target_model=None, tokenizer=None, draft_path=None):
        draft_model = EagleSDDraftModel.from_pretrained(
            draft_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id
        ).to(target_model.lm_head.weight.device)
        draft_model.update_modules(embed_tokens=target_model.get_input_embeddings(), lm_head=target_model.lm_head)
        return draft_model
    
    
if __name__ == "__main__":
    run_app(EagleSDBuilder())