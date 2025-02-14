
from .app_router import run_app
from .base import BaseBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.share_sd import ShareSDDraftModel
from specdecodes.models.generators.share_sd import ShareSDGenerator

from hqq.core.quantize import *

def temp_recipe(model, draft_model, vram_limit):  
    # Quantization
    nbits = 2
    group_size = 32
    base_quant_config_a = BaseQuantizeConfig(nbits=4, group_size=32, axis=1)
    base_quant_config_b = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
    
    quant_config = {}
    layer_cnt = len(model.model.layers)
    quant_start = 0 + 5 
    quant_end = layer_cnt - 1 - 5
    for i in range(quant_start, quant_end+1):
        quant_config[f"layers.{i}.self_attn.q_proj"] = base_quant_config_b
        quant_config[f"layers.{i}.self_attn.k_proj"] = base_quant_config_b
        quant_config[f"layers.{i}.self_attn.v_proj"] = base_quant_config_b
        quant_config[f"layers.{i}.self_attn.o_proj"] = base_quant_config_b
        quant_config[f"layers.{i}.mlp.gate_proj"] = base_quant_config_b
        quant_config[f"layers.{i}.mlp.up_proj"] = base_quant_config_b 
        quant_config[f"layers.{i}.mlp.down_proj"] = base_quant_config_a
        # quant to 2 bits (4 bits for the others):
        # u 12.50 # g 14.19 # d 11.93
        # g+u 11.93 # g+d 11.17 # u+d 10.29 
        # q+k+v+o+g 11.17

    target_config = None
    draft_config = {
        "quant_config": {
            "config": quant_config,
            "backend": "gemlite", #"torchao_int4",
        },
    }
    
    return target_config, draft_config

class ShareSDBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Load model configurations
        # self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.llm_path = "meta-llama/Llama-2-7b-chat-hf"
        self.generator_class = ShareSDGenerator
        self.draft_params = DraftParams(
            max_depth=15,
            topk_len=8,
            max_verify_tokens=128,
            min_accept_prob=1e-8,
        )
        
        # Sample configurations
        self.do_sample = False
        self.temperature = 0
        
        # Quantization and offloading
        self.recipe = temp_recipe
        
        # Speed up inference using torch.compile
        self.cache_implementation = "static"
        self.warmup_iter = 10
        self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
    
    def _load_draft_model(self, target_model=None, tokenizer=None, draft_path=None):
        draft_model = ShareSDDraftModel.from_pretrained(
            draft_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id
        ).to(target_model.lm_head.weight.device)
        draft_model.update_modules(embed_tokens=target_model.get_input_embeddings(), lm_head=target_model.lm_head)
        return draft_model
    
if __name__ == "__main__":
    run_app(ShareSDBuilder())