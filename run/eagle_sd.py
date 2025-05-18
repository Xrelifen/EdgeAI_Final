from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.eagle_sd import EagleSDDraftModel
from specdecodes.models.generators.eagle_sd import EagleSDGenerator


class EagleSDBuilder(GeneratorPipelineBuilder):
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
        self.generator_class = EagleSDGenerator
        self.draft_params = DraftParams(
            temperature=1,
            max_depth=6,
            topk_len=10,
            max_verify_tokens=1024,
        )

        # Recipe for quantization and offloading.
        self.recipe = None
        self.cpu_offload_gb = None

        # Additional configurations.
        self.cache_implementation = "static"
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


if __name__ == "__main__":
    run_app(EagleSDBuilder())
