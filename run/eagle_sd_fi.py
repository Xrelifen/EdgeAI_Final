from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.eagle_sd import EagleSDDraftModel
from specdecodes.models.generators.eagle_sd_fi import EagleSDFIGenerator
from specdecodes.models.utils.flashinfer.cache_manager import FlashInferCache
from specdecodes.models.utils.flashinfer.monkey_patch import (
    apply_flashinfer_kernel_to_llama,
)
from specdecodes.models.utils.cache_utils import create_kv_cache


class EagleSDFIBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Device and precision settings.
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16

        # Model paths.
        self.llm_path = "meta-llama/Llama-3.2-3B-Instruct"
        self.draft_model_path = "JKroller/llama3.2-3b-eagle"

        # Generation parameters.
        self.do_sample = False
        self.temperature = 0

        # Generator-specific configurations.
        self.generator_class = EagleSDFIGenerator
        self.draft_params = DraftParams(
            temperature=1,
            max_depth=2,
            topk_len=10,
            max_verify_tokens=256,
        )

        # Recipe for quantization and offloading.
        self.recipe = None
        self.cpu_offload_gb = None

        # Additional configurations.
        self.cache_implementation = "static"
        self.page_len = 16
        self.warmup_iter = 5
        # self.compile_mode = "max-autotune"

        # Profiling.
        # self.generator_profiling = True

    def load_draft_model(self, target_model, tokenizer, draft_model_path):
        draft_model = EagleSDDraftModel.from_pretrained(
            draft_model_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            use_hf_eagle=True,
            eos_token_id=tokenizer.eos_token_id,
        ).to(self.device)
        draft_model.update_modules(
            embed_tokens=target_model.get_input_embeddings(),
            lm_head=target_model.lm_head,
        )
        return draft_model

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
        draft_past_key_values = create_kv_cache(
            "static",
            max_cache_len=max_cache_len,
            max_batch_size=1,
            config=draft_model.model.config,
            device=self.device,
            dtype=draft_model.model.dtype,
        )
        apply_flashinfer_kernel_to_llama(
            attention=True, rms_norm=True, silu=False, model=target_model
        )

        return past_key_values, draft_past_key_values


if __name__ == "__main__":
    run_app(EagleSDFIBuilder())
