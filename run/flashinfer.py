from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.generators.naive_fi import NaiveFIGenerator
from specdecodes.models.utils.flashinfer.cache_manager import FlashInferCache
from specdecodes.models.utils.flashinfer.monkey_patch import (
    apply_flashinfer_kernel_to_llama,
)


class FlashinferBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16

        # Model paths.
        self.llm_path = "meta-llama/Llama-3.2-3B-Instruct"
        self.generator_class = NaiveFIGenerator

        # Generation parameters.
        self.do_sample = False
        self.temperature = 0

        # Recipe for quantization and offloading.
        self.recipe = None
        self.cpu_offload_gb = None

        # Additional configurations.
        self.cache_implementation = "static"
        self.page_len = 16
        self.warmup_iter = 5
        # self.compile_mode = "max-autotune"

        # Profiling
        self.generator_profiling = True

    def load_kv_cache(self, target_model, draft_model):

        if self.max_length is not None:
            if draft_model is not None:
                # Additional sample tokens may cause KV-Cache tp exceed max_length
                max_cache_len = (
                    self.max_length
                    + self.draft_params.max_verify_tokens
                    + self.page_len
                )
            else:
                max_cache_len = self.max_length
        else:
            raise ValueError("max_length should be set for static cache.")

        past_key_values = FlashInferCache(
            target_model.config, max_tokens=max_cache_len, PAGE_LEN=self.page_len
        ).kvCachePool
        draft_past_key_values = None
        apply_flashinfer_kernel_to_llama(
            attention=True, rms_norm=True, silu=False, model=target_model
        )

        return past_key_values, draft_past_key_values


if __name__ == "__main__":
    run_app(FlashinferBuilder())
