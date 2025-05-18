from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.generators.naive import NaiveGenerator


class NaiveBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations
        self.device = "cuda:0"
        self.dtype = torch.float16

        # Model paths.
        self.llm_path = "meta-llama/Llama-3.2-3B-Instruct"
        self.generator_class = NaiveGenerator

        # Generation parameters.
        self.do_sample = False
        self.temperature = 0

        # Recipe for quantization and offloading.
        self.recipe = None
        self.cpu_offload_gb = None

        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 5
        # self.compile_mode = "max-autotune"

        # Profiling
        self.generator_profiling = True


if __name__ == "__main__":
    run_app(NaiveBuilder())
