import logging
from .app_router import run_app
from .base import BaseBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.share_sd import ShareSDDraftModel
from specdecodes.models.generators.share_sd import ShareSDGenerator
from specdecodes.helpers.offloaders.prefetch_offloader import PrefetchOffloader
from specdecodes.helpers.recipes.recipe_4bit_mlp import recipe

class ShareSDBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Load model configurations
        # self.llm_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.generator_class = ShareSDGenerator
        self.draft_params = DraftParams(
            max_depth=50,
            topk_len=16,
            max_verify_tokens=1024,
            min_accept_prob=1e-8,
        )
        
        # Sample configurations
        self.do_sample = False
        self.temperature = 0
        
        # Quantization and offloading
        self.recipe = recipe
        self.offloader = PrefetchOffloader
        
        # Speed up inference using torch.compile
        self.cache_implementation = "static"
        self.warmup_iter = 0
        self.compile_mode = "max-autotune"
        
        # self.offloader = None
        
        # Profiling
        self.generator_profiling = True
    
    def _load_draft_model(self, target_model=None, tokenizer=None, draft_path=None):
        draft_model = ShareSDDraftModel.from_pretrained(
            draft_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id
        )
        return draft_model
    
    def _compile_generator(self, generator, compile_mode):
        logging.info(f"Compiling the generator with mode: {compile_mode}")
        torch.set_float32_matmul_precision('high')
        # generator.target_model.forward = torch.compile(generator.target_model.forward, mode=compile_mode, dynamic=False, fullgraph=True)
        if getattr(generator, 'draft_model', None) is not None:
            generator.draft_model.forward = torch.compile(generator.draft_model.forward, mode=compile_mode, dynamic=False, fullgraph=True)
    
if __name__ == "__main__":
    run_app(ShareSDBuilder())