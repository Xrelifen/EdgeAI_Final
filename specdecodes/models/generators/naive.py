import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .base import GeneratorBase
from ..utils.mixin import ProfilingMixin


class NaiveGeneratorBase(GeneratorBase):
    def __init__(self, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        assert self.target_model is not None, "target_model must be provided"

        # Clone input_ids
        input_ids = input_ids.clone()
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # Prepare kv-cache and cache position
        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )

        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values, "max_cache_len", None)
        else:
            raise ValueError("past_key_values should be provided")

        cache_position = torch.arange(org_input_len, dtype=torch.long, device=input_ids.device)

        # Prefill stage
        with nvtx.annotate("prefill", color="orange"):
            outputs = self.target_model.prefill_forward(
                input_ids,
                max_cache_len=max_cache_len,
                past_key_values=past_key_values,
                cache_position=cache_position,
                num_logits_to_keep=1,
            )
            next_token_logits = outputs.logits

        with nvtx.annotate("sample tokens"):
            next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("update data"):
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            cache_position = cache_position[-1:] + 1

        # Decoding loop
        with nvtx.annotate("decoding"):
            finished = False
            while not finished:
                with nvtx.annotate("llm forward", color="orange"):
                    outputs = self.target_model(
                        next_tokens,
                        past_key_values=past_key_values,
                        position_ids=cache_position.unsqueeze(0),
                        cache_position=cache_position,
                    )
                    next_token_logits = outputs.logits

                with nvtx.annotate("sample tokens"):
                    next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    cache_position += 1

                with nvtx.annotate("stopping criteria"):
                    finished = stopping_criteria(input_ids, None)

        return input_ids

class NaiveGenerator(ProfilingMixin, NaiveGeneratorBase):
    pass