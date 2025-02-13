from .app_router import run_app
from .base import BaseBuilder

from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.classic_sd import ClassicSDDraftModel
from specdecodes.models.generators.classic_sd import ClassicSDGenerator

from hqq.core.quantize import *

def temp_recipe(model, draft_model, vram_limit):  
    # Quantization
    base_quant_config_a = BaseQuantizeConfig(nbits=4, group_size=32, axis=1)
    
    quant_config = {}
    for i in range(0, 31+1):
        quant_config[f"layers.{i}.self_attn.q_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.self_attn.k_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.self_attn.v_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.self_attn.o_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.mlp.gate_proj"] = base_quant_config_a
        quant_config[f"layers.{i}.mlp.up_proj"] = base_quant_config_a 
        quant_config[f"layers.{i}.mlp.down_proj"] = base_quant_config_a

    target_config = None
    draft_config = {
        "quant_config": {
            "config": quant_config,
            "backend": "gemlite",
        },
    }
    
    return target_config, draft_config

class ClassicSDBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_model_path = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Generator configurations
        self.generator_class = ClassicSDGenerator
        self.draft_params = DraftParams(
            max_depth=15,
            topk_len=8,
            max_verify_tokens=128,
            min_accept_prob=1e-8,
        )
        
        # Offloading
        # self.recipe = temp_recipe
        # self.vram_limit = None # in GB
        
        # Speed up inference using torch.compile
        # self.cache_implementation = "static"
        # self.warmup_iter = 10
        # self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
        self.nvtx_profiling = False
    
    def _load_draft_model(self, target_model=None, tokenizer=None, draft_path=None):
        draft_model = ClassicSDDraftModel.from_pretrained(
            draft_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id
        ).to(target_model.lm_head.weight.device)
        draft_model.update_modules(embed_tokens=target_model.get_input_embeddings(), lm_head=target_model.lm_head)
        return draft_model
    
    
if __name__ == "__main__":
    run_app(ClassicSDBuilder())