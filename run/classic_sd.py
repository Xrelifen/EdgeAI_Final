from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.classic_sd import ClassicSDDraftModel
from specdecodes.models.generators.classic_sd import ClassicSDGenerator

class ClassicSDBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Device and precision settings.
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Model paths.
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_model_path = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_class = ClassicSDGenerator
        self.draft_params = DraftParams(
            max_depth=12,
            topk_len=16,
            max_verify_tokens=1024,
        )
        
        # Recipe for quantization and offloading.
        self.recipe = None
        self.cpu_offload_gb = None
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 10
        self.compile_mode = "max-autotune"
        
        # Profiling.
        self.generator_profiling = True
    
    def load_draft_model(self, target_model, tokenizer, draft_model_path):
        draft_model = ClassicSDDraftModel.from_pretrained(
            draft_model_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            device_map=self.device,
            eos_token_id=tokenizer.eos_token_id
        )
        return draft_model

if __name__ == "__main__":
    run_app(ClassicSDBuilder())