from ..app import run_app
from .base import BaseRunner

import torch
from specdecodes.models.utils.modeling_utils import DraftParams
from specdecodes.models import SSM_Eagle, ProfileSDWrapper
from copy import deepcopy

class MyRunner(BaseRunner):
    def __init__(self):
        super().__init__()
        self.llm_path = "meta-llama/Llama-2-7b-chat-hf"
        self.draft_path = "/home/scott306lr_l/checkpoints/eagle/sl1-ce/model_5"
        self.draft_params = DraftParams(
            max_depth=6,
            topk_len=10,
            max_verify_tokens=64,
            min_accept_prob=1e-8,
        )
        self.offload_recipe = None
        self.vram_limit = None # in GB
        
        # Speed up inference using torch.compile
        self.warmup_iter = 10
        self.compile_mode = "max-autotune"
        self.cache_implementation = "static"
    
    def _load_draft_model(self, model, tokenizer, draft_path):
        draft_config = deepcopy(model.config)
        draft_config.num_hidden_layers = 1
        draft_model = SSM_Eagle.from_pretrained(
            draft_path,
            config=draft_config,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id,
            keep_embeddings=False
        ).to(model.lm_head.weight.device)
        draft_model.set_modules(embed_tokens=model.get_input_embeddings(), lm_head=model.lm_head)
        return draft_model
    
    def _load_pipeline_method(self, *args):
        return ProfileSDWrapper(*args)
    
    
if __name__ == "__main__":
    run_app(MyRunner())