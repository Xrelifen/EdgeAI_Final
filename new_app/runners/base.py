from ..app import run_app

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from specdecodes.models.utils.modeling_utils import DraftParams
from specdecodes.models import ProfileNaiveWrapper

class BaseRunner:
    def __init__(self):
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.seed = 0
        
        # Load model paths
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_path = None
        
        # Sample configurations
        self.max_length = 1024
        self.do_sample = False
        self.temperature = 0
        
        # Additional configurations
        self.warmup_iter = 0
        self.compile_mode = None
        self.cache_implementation = "dynamic"
        
        self.draft_params = DraftParams(
            max_depth=12,
            topk_len=1,
            max_verify_tokens=64,
            min_accept_prob=1e-8,
        )

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
        
    def _load_pipeline(self, **kwargs):
        return ProfileNaiveWrapper(**kwargs)
    
    def _compile_pipeline(self, pipeline, compile_mode):
        if compile_mode is not None:
            torch.set_float32_matmul_precision('high')
            pipeline.llm.forward = torch.compile(pipeline.llm.forward, mode=compile_mode, dynamic=False, fullgraph=True)
            if pipeline.ssm is not None:
                pipeline.ssm.forward = torch.compile(pipeline.ssm.forward, mode=compile_mode, dynamic=False, fullgraph=True)
    
    def build_pipeline(self):
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
            pipeline = self._load_pipeline(draft_params=self.draft_params)
        else:
            pipeline = self._load_pipeline()
            
        pipeline.cache_implementation = self.cache_implementation
        pipeline.set_llm(model)
        pipeline.set_tokenizer(tokenizer)
        if draft_model is not None:
            pipeline.set_ssm(draft_model)
        pipeline.eval()
        
        # 6. Compile pipeline if needed
        self._compile_pipeline(pipeline, self.compile_mode)
        
        return pipeline, tokenizer
            
        
if __name__ == "__main__":
  run_app(BaseRunner())