import logging
import os
from .app_router import run_app

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from specdecodes.models.generators.naive import NaiveGenerator

from hqq.core.quantize import *
from hqq.utils.patching import prepare_for_inference
from specdecodes.models.utils.hqq.hf.base import AutoHQQHFModel

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)

def quantize_model(model, quant_config, dtype, device):
    logging.info(f"Quantizing model and applying backend: {quant_config['backend']}")
    AutoHQQHFModel.quantize_model(model, quant_config["config"], compute_dtype=dtype, device=device)
    HQQLinear.set_backend(HQQBackend.PYTORCH)
    prepare_for_inference(model, backend=quant_config["backend"])

class BaseBuilder:
    def __init__(self):
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Load model configurations
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_model_path = None
        
        # Sample configurations
        self.max_length = 1024
        self.do_sample = False
        self.temperature = 0
        
        # Generator configurations
        self.generator_class = NaiveGenerator
        self.draft_params = None
        
        # Additional configurations
        self.cache_implementation = "dynamic"
        self.warmup_iter = 0
        self.compile_mode = None

        # Quantization and offloading
        self.recipe = None
        self.vram_limit = None
        self.target_config = None
        self.draft_config = None
        
        # Profiling
        self.generator_profiling = True
        self.nvtx_profiling = False
        
        # Pipeline print Results
        self.print_time = True
        self.print_message = True
        
        # For benchmarking
        self.out_dir = None
        self.log_dir = "experiments"
        
        
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
    
    def _load_draft_model(self, model, tokenizer, draft_model_path):
        # No draft model is needed for the base runner
        return None
    
    def _compile_generator(self, generator, compile_mode):
        if compile_mode is not None:
            logging.info(f"Compiling the generator with mode: {compile_mode}")
            torch.set_float32_matmul_precision('high')
            generator.target_model.forward = torch.compile(generator.target_model.forward, mode=compile_mode, dynamic=False, fullgraph=True)
            if getattr(generator, 'draft_model', None) is not None:
                generator.draft_model.forward = torch.compile(generator.draft_model.forward, mode=compile_mode, dynamic=False, fullgraph=True)
    
    def build_generator(self):
        # 1. Load model and tokenizer
        model, tokenizer = self._load_model_and_tokenizer(self.llm_path)
        draft_model = self._load_draft_model(model, tokenizer, self.draft_model_path)
        
        # 2. Obtain configs either by recipe or manually.
        if self.recipe:
            target_config, draft_config = self.recipe(model, draft_model, vram_limit=self.vram_limit)
        else:
            target_config = getattr(self, "target_config", None)
            draft_config = getattr(self, "draft_config", None)
        
        # # 3. Quantize if needed before offloading
        if draft_config is not None and draft_config.get("quant_config"):
            quantize_model(draft_model.model, draft_config["quant_config"], self.dtype, self.device)
        if target_config is not None and target_config.get("quant_config"):
            quantize_model(model, target_config["quant_config"], self.dtype, self.device)
        
        #  # 4. Apply hook and offload llm
        # if target_config is not None and target_config.get("device_map"):
        #     Offloader.dispatch_model(model, target_config["device_map"], compute_device=self.device)

        # 5. Build up the pipeline
        generator = self.generator_class(
                target_model=model,
                tokenizer=tokenizer,
                draft_model=draft_model,
                draft_params=self.draft_params,
                cache_implementation=self.cache_implementation,
                profiling=self.generator_profiling,
            )
        generator.eval()
        
        # 6. Compile pipeline if needed
        self._compile_generator(generator, self.compile_mode)
        
        return generator, tokenizer
            
        
if __name__ == "__main__":
    run_app(BaseBuilder())