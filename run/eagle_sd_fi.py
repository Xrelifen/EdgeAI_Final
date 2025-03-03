from .app_router import run_app
from .base import BaseBuilder

from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.eagle_sd import EagleSDDraftModel
from specdecodes.models.generators.eagle_sd import EagleSDGenerator
from specdecodes.models.generators.eagle_sd_fi import EagleSDFIGenerator
from specdecodes.models.utils.flashinfer.monkey_patch import apply_flashinfer_kernel_to_llama
from hqq.core.quantize import *

def temp_recipe(model, draft_model, vram_limit):  
    # Quantization
    base_quant_config_a = BaseQuantizeConfig(nbits=4, group_size=32, axis=1)
    
    quant_config = {}
    for i in range(0, 1+1):
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

class EagleSDBuilder(BaseBuilder):
    def __init__(self):
        super().__init__()
        self.llm_path = "meta-llama/Llama-2-7b-chat-hf"
        self.draft_model_path = "/home/scott306lr_l/checkpoints/eagle/sl1-ce/model_5"
        
        self.dtype = torch.float16
        
        # Generator configurations
        self.generator_class = EagleSDFIGenerator
        self.draft_params = DraftParams(
            max_depth=6,
            topk_len=10,
            max_verify_tokens=64,
            min_accept_prob=1e-8,
        )
        
        # Offloading
        # self.recipe = temp_recipe
        # self.vram_limit = None # in GB
        
        # Speed up inference using torch.compile
        self.cache_implementation = "flashinfer"
        self.warmup_iter = 5
        self.compile_mode = None
        
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
        apply_flashinfer_kernel_to_llama(attention=True, rms_norm=True, swiglu=False, model=target_model)
        return draft_model
    
    
if __name__ == "__main__":
    run_app(EagleSDBuilder())