from hqq.core.quantize import *
from ...utils import estimate_quantized_size


def recipe(model, draft_model, max_length, cpu_offload_gb, dtype, device):
    # Quantization
    quant_config = {}
    attn_quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
    mlp_quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
    
    layer_cnt = len(model.model.layers)
    quant_start = 0
    quant_end = layer_cnt - 1
    for i in range(quant_start, quant_end+1):
        quant_config[f"model.layers.{i}.self_attn.q_proj"] = attn_quant_config
        quant_config[f"model.layers.{i}.self_attn.k_proj"] = attn_quant_config
        quant_config[f"model.layers.{i}.self_attn.v_proj"] = attn_quant_config
        quant_config[f"model.layers.{i}.self_attn.o_proj"] = attn_quant_config
        quant_config[f"model.layers.{i}.mlp.gate_proj"] = mlp_quant_config
        quant_config[f"model.layers.{i}.mlp.up_proj"] = mlp_quant_config
        quant_config[f"model.layers.{i}.mlp.down_proj"] = mlp_quant_config
        
    estimate_qmodel_size = estimate_quantized_size(model, quant_config)
    print(f"Model required VRAM: {estimate_qmodel_size / 1024**3:.2f} GiB")

    # Configs
    target_config = {
        "quant_config": {
            "config": quant_config,
            "backend": "gemlite", #"torchao_int4",
        },
    }
    draft_config = None
    
    return target_config, draft_config