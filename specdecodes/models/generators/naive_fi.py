import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx
import os

from dataclasses import dataclass
from enum import Enum
from .base import GeneratorBase
from ..utils.mixin import ProfilingMixin
from ..utils.flashinfer.monkey_patch import apply_flashinfer_kernel_to_llama
from ..utils.flashinfer.cache_manager import (
    KvCachePool,
    KvCacheBatchPosition,
    RequestKvCache,
    getKvCacheBatchPosition,
    FlashInferCache
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper

class POS_ENCODING_MODE(Enum):
    ROPE_LLAMA = "ROPE_LLAMA"
    ALIBI = "ALIBI"
    NONE = "NONE"

@dataclass(frozen=True)
class AttentionRotaryParams:
    causal: bool = True
    pos_encoding_mode: POS_ENCODING_MODE = POS_ENCODING_MODE.ROPE_LLAMA
    rope_scale: float = 1.0
    rope_theta: float = 1.0e4
    

class NaiveFIGeneratorBase(GeneratorBase):
    def __init__(self, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        apply_flashinfer_kernel_to_llama(attention=True, rms_norm=True, swiglu=False, model=self.target_model)
        print("using flashinfer backend")

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
        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values, "max_cache_len", None)
        else:
            raise ValueError("past_key_values should be provided")
        
        
        kvCachePool = past_key_values
        seq_init_len = input_ids.shape[1]
        currentDevice = torch.device(f'cuda:{torch.cuda.current_device()}')

        PAGE_LEN = kvCachePool.page_len
        # Create a RequestKvCache instance
        request_kv_cache = RequestKvCache(
            kvCachePool=kvCachePool,
            page_len=PAGE_LEN,
            seq_init_len=seq_init_len
        )

        # Generate the KvCacheBatchPosition
        batch_position = getKvCacheBatchPosition(
            request_kv_caches=[request_kv_cache],
            mode='prefill',  # Set to False if you're doing incremental decoding
            device=currentDevice
        )
        
        # Prefill stage
        with nvtx.annotate("prefill", color="orange"):
            flashinferWrapper = FlashinferAttentionWrapper(
                self.target_model.config.num_attention_heads, self.target_model.config.num_key_value_heads, self.target_model.config.hidden_size,PAGE_LEN
            )
            flashinferWrapper.prepareAttention(
                'prefill',
                batch_position,
                kvCachePool.page_len,
                POS_ENCODING_MODE.NONE,
                kvCachePool.cache_data[0].dtype,
            )
            
            outputs = self.target_model(
                input_ids=input_ids,
                return_dict=True,
                past_key_values=None,
                use_cache=False,
                kvCachePool=kvCachePool,
                batch_position=batch_position,
                mode='prefill', 
                flashinferWrapper = flashinferWrapper
            )
        
            flashinferWrapper.endBatchAttention('prefill')
            next_token_logits = outputs.logits[:, -1:, :]

        with nvtx.annotate("sample tokens"):
            next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)
            # print("next_tokens")
            # print(next_token_logits.shape)

        with nvtx.annotate("update data"):
            # generated_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            # print("Generated text:", generated_text)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # generated_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            # print("Generated text:", generated_text)
            # cache_position = cache_position[-1:] + 1

        # Decoding loop
        with nvtx.annotate("decoding"):
            finished = False
            while not finished:

                # Update the KvCacheBatchPosition
                request_kv_cache.increment()
                batch_position = getKvCacheBatchPosition(
                    request_kv_caches=[request_kv_cache],
                    mode='decode',
                    device=currentDevice
                )
                seq_len = input_ids.shape[1]
                last_position = seq_len - 1
                position_ids = torch.tensor([[last_position]], device=currentDevice, dtype=torch.long)

                with nvtx.annotate("llm forward", color="orange"):
                    flashinferWrapper.prepareAttention(
                        'decode',
                        batch_position,
                        kvCachePool.page_len,
                        POS_ENCODING_MODE.NONE,
                        kvCachePool.cache_data[0].dtype,
                    )
         
                    outputs = self.target_model(
                        input_ids=input_ids[:, -1:],
                        position_ids = position_ids,
                        return_dict=True,
                        past_key_values=None,
                        use_cache=False,
                        kvCachePool=kvCachePool,
                        batch_position=batch_position,
                        mode='decode', 
                        flashinferWrapper = flashinferWrapper,
                        
                    )
                    
                    flashinferWrapper.endBatchAttention('decode')

                    
                    next_token_logits = outputs.logits

                with nvtx.annotate("sample tokens"):
                    next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)
                    # print("decoding logits shape")
                    # print(next_token_logits.shape)

                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    # cache_position += 1

                with nvtx.annotate("stopping criteria"):
                    finished = stopping_criteria(input_ids, None)

        request_kv_cache.release() 
        return input_ids

class NaiveFIGenerator(ProfilingMixin, NaiveFIGeneratorBase):
    pass