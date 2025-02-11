
from ..app import run_app
from .base import BaseBuilder

import torch
from specdecodes.models.utils.modeling_utils import DraftParams
from specdecodes.models import SSM_ShareSD, ProfileShareSDWrapper

class MyBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Generator configurations
        self.generator_class = ProfileShareSDWrapper
        self.draft_params = DraftParams(
            max_depth=15,
            topk_len=4,
            max_verify_tokens=64,
            min_accept_prob=1e-8,
        )
        
        # Offloading
        self.offload_recipe = None
        self.vram_limit = None # in GB
        
        # Speed up inference using torch.compile
        # self.warmup_iter = 5 # 10
        # self.compile_mode = "max-autotune"
        # self.cache_implementation = "static"
    
    def _load_draft_model(self, model, tokenizer, draft_path):
        draft_model = SSM_ShareSD.from_pretrained(
            model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id,
        )
        return draft_model
    
    
if __name__ == "__main__":
    run_app(MyBuilder())