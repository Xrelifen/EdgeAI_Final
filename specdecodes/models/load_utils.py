
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .wrapper.huggingface import HuggingFaceWrapper
from .wrapper.naive import NaiveWrapper, ProfileNaiveWrapper
from .wrapper.sd import SDWrapper, ProfileSDWrapper
from .wrapper.share_sd import ShareSDWrapper, ProfileShareSDWrapper
from .ssm import SSM_Classic, SSM_Eagle, SSM_ShareSD

import os
from copy import deepcopy
from dataclasses import dataclass

from hqq.core.quantize import *
from hqq.utils.patching import prepare_for_inference
from .cache_utils import TreeDynamicCache, TreeStaticCache
from .utils import DraftParams
from .hqq.hf.base import AutoHQQHFModel

        
def create_kv_cache(
    cache_implementation = "dynamic",
    max_cache_len=None,
    max_batch_size=None,
    config=None,
    device='cpu',
    dtype='float16',
):
    if cache_implementation == "dynamic":
        return TreeDynamicCache()
    
    elif cache_implementation == "static":
        return TreeStaticCache(
            max_cache_len=max_cache_len,
            max_batch_size=max_batch_size,
            config=config,
            device=device,
            dtype=dtype,
        )

def share_param_deepcopy(model):
    # Build the memo dictionary from the model's parameters (and optionally buffers)
    model_memo = {}
    for _, param in model.named_parameters():
        model_memo[id(param)] = param
    for _, buf in model.named_buffers():
        model_memo[id(buf)] = buf

    # Clone the model using the memo dictionary.
    qmodule = deepcopy(model, memo=model_memo)
    return qmodule

def load_model(
    llm_path: str,
    ssm_path: str = None,
    mode: str = "naive",
    cache_impl: str = "dynamic",
    compile_mode: str = "eager",
    logging: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = "auto",
    **kwargs
):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    
    # load LLM
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device,
        _attn_implementation="sdpa",
    )

    ssm = None
    model_map = {
        "naive": lambda: ProfileNaiveWrapper() if logging else NaiveWrapper(),
        "hf": lambda: HuggingFaceWrapper(),
    }
    
    if mode.startswith("sd"):
        draft_config = deepcopy(llm.config) if os.path.exists(ssm_path) else None
        if draft_config:
            draft_config.num_hidden_layers = 1
        draft_params = kwargs.get("draft_params", DraftParams())
        
        if mode == "sd-share":
            # Clone the model using the memo dictionary.
            qmodule = share_param_deepcopy(llm)
            
            # quantize
            print("Quantizing model...")
            nbits = kwargs.get("nbits", 4)
            group_size = kwargs.get("group_size", 64)
            quant_range = kwargs.get("quant_range", (-1, -1)) # quantize all layers by default
            backend = "torchao_int4" if nbits == 4 else "gemlite"
            if backend == "torchao_int4":
                assert dtype == torch.bfloat16, "torchao_int4 only supports bfloat16."
            if backend == "gemlite":
                assert dtype != torch.bfloat16, "GemLite does not support bfloat16."
            
            base_quant_config_a = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
            base_quant_config_b = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
            quant_config = {}
            # for i in range(quant_start, quant_end+1):
            #get llm layers, ensure quant_range is in valid range
            layer_cnt = len(qmodule.model.layers)
            quant_start = quant_range[0] if quant_range[0] >= 0 else 0
            quant_end = quant_range[1] if (quant_range[1] > 0 and quant_range[1] < layer_cnt) else layer_cnt - 1
            for i in range(quant_start, quant_end+1):
                quant_config[f"layers.{i}.self_attn.q_proj"] = base_quant_config_a
                quant_config[f"layers.{i}.self_attn.k_proj"] = base_quant_config_a
                quant_config[f"layers.{i}.self_attn.v_proj"] = base_quant_config_a
                quant_config[f"layers.{i}.self_attn.o_proj"] = base_quant_config_a
                quant_config[f"layers.{i}.mlp.gate_proj"] = base_quant_config_b
                quant_config[f"layers.{i}.mlp.up_proj"] = base_quant_config_b
                quant_config[f"layers.{i}.mlp.down_proj"] = base_quant_config_b

            AutoHQQHFModel.quantize_model(qmodule, quant_config=quant_config, compute_dtype=dtype, device="cuda")
            if compile_mode != 'eager':
                HQQLinear.set_backend(HQQBackend.PYTORCH)
            else:
                HQQLinear.set_backend(HQQBackend.ATEN)

            prepare_for_inference(qmodule, backend=backend)
        
        ssm_map = {
            "sd-classic": lambda: SSM_Classic.from_pretrained(
                ssm_path, config=draft_config, eos_token_id=tokenizer.eos_token_id, torch_dtype=dtype
            ),
            "sd-eagle": lambda: SSM_Eagle.from_pretrained(
                ssm_path, config=draft_config, eos_token_id=tokenizer.eos_token_id, 
                torch_dtype=dtype, keep_embeddings=False
            ),
            "sd-share": lambda: SSM_ShareSD.from_pretrained(
                qmodule, eos_token_id=tokenizer.eos_token_id, torch_dtype=dtype
            ),
        }
        if mode not in ssm_map:
            raise ValueError("Invalid sd mode.")
        
        ssm = ssm_map[mode]().to(llm.model.layers[-1].self_attn.q_proj.weight.device)
        if mode == "sd-eagle":
            ssm.set_modules(embed_tokens=llm.get_input_embeddings(), lm_head=llm.lm_head)
        
        if mode == "sd-share":
            model = ProfileShareSDWrapper(draft_params=draft_params, out_dir=None) if logging else ShareSDWrapper(draft_params=draft_params)
        else:
            model = ProfileSDWrapper(draft_params=draft_params, out_dir=None) if logging else SDWrapper(draft_params=draft_params)
        model.set_ssm(ssm)
    else:
        model = model_map.get(mode, lambda: None)()
        if model is None:
            raise ValueError("Invalid mode.")
        
    model.cache_implementation = cache_impl
    model.set_tokenizer(tokenizer)
    model.set_llm(llm)
    model.eval()
        
    if compile_mode != 'eager':
        print("Running with Torch Inductor...")
        torch.set_float32_matmul_precision('high')

        llm.forward = torch.compile(llm.forward, mode=compile_mode, dynamic=False, fullgraph=True)
        if ssm is not None:
            ssm.forward = torch.compile(ssm.forward, mode=compile_mode, dynamic=False, fullgraph=True)
            #TODO: Modify sd/ssm to support prefill_forward compilation on sd-eagle and sd-classic
            # if mode != "sd-share": # sd-share does not require prefill_forward
            #     ssm.prefill_forward = torch.compile(ssm.prefill_forward, mode=compile_mode, dynamic=True, fullgraph=True)

    return model, tokenizer