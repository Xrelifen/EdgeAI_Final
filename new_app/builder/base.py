from ..app import run_app

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from specdecodes.models.utils.modeling_utils import DraftParams
from specdecodes.models import ProfileNaiveWrapper

class BaseBuilder:
    def __init__(self):
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.seed = 0
        
        # Load model configurations
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_path = None
        
        # Sample configurations
        self.max_length = 1024
        self.do_sample = False
        self.temperature = 0
        
        # Generator configurations
        self.generator_class = ProfileNaiveWrapper
        self.draft_params = None
        
        # Additional configurations
        self.warmup_iter = 0
        self.compile_mode = None
        self.cache_implementation = "dynamic"

        # Offloading
        self.offload_recipe = None
        self.target_config = None
        self.draft_config = None
        
        # Print Results
        self.print_time = True
        self.print_message = True
        
        # Logging and profiling
        self.logging = False
        self.nvtx_profiling = False
        
        
    def _load_model_and_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=self.device,
            _attn_implementation="sdpa",
        )
        return model, tokenizer
    
    def _load_draft_model(self, model, tokenizer, draft_path):
        # No draft model is needed for the base runner
        return None
        
    def _load_generator(self, **kwargs):
        return ProfileNaiveWrapper(**kwargs)
    
    def _compile_generator(self, generator, compile_mode):
        if compile_mode is not None:
            torch.set_float32_matmul_precision('high')
            generator.llm.forward = torch.compile(generator.llm.forward, mode=compile_mode, dynamic=False, fullgraph=True)
            if generator.ssm is not None:
                generator.ssm.forward = torch.compile(generator.ssm.forward, mode=compile_mode, dynamic=False, fullgraph=True)
    
    def build_generator(self):
        # 1. Load model and tokenizer
        model, tokenizer = self._load_model_and_tokenizer(self.llm_path)
        draft_model = self._load_draft_model(model, tokenizer, self.draft_path)
        
        # 2. Obtain configs either by recipe or manually.
        if self.offload_recipe:
            target_config, draft_config = self.recipe(model, vram_limit=self.vram_limit)
        else:
            target_config = getattr(self, "target_config", None)
            draft_config = getattr(self, "draft_config", None)
        
        # # 3. Quantize if needed before offloading
        # if draft_config is not None and draft_config.get("quant_config"):
        #     draft_model.model = quantize_model(draft_model.model, draft_config["quant_config"])
        # if target_config is not None and target_config.get("quant_config"):
        #     model = quantize_model(model, target_config["quant_config"])
        
        #  # 4. Apply hook and offload llm
        # if target_config is not None and target_config.get("device_map"):
        #     Offloader.dispatch_model(model, target_config["device_map"], compute_device=self.device)

        # 5. Build up the pipeline
        if draft_model is not None:
            generator = self.generator_class(draft_params=self.draft_params)
        else:
            generator = self._load_generator()
            
        generator.cache_implementation = self.cache_implementation
        generator.set_llm(model)
        generator.set_tokenizer(tokenizer)
        if draft_model is not None:
            generator.set_ssm(draft_model)
        generator.eval()
        
        # 6. Compile pipeline if needed
        self._compile_generator(generator, self.compile_mode)
        
        return generator, tokenizer
            
        
if __name__ == "__main__":
  run_app(BaseBuilder())