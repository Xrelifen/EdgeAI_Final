from hqq.core.quantize import *

def recipe(model, draft_model, max_length, cpu_offload_gb, dtype, device):
    layer_cnt = len(model.model.layers)
    
    # Offloading
    device_config = {}
    start = 0
    end = layer_cnt - 12 # Last 12 layers are not offloaded
    for i in range(start, end):
        device_config[f"model.layers.{i}.self_attn.q_proj"] = 'cpu'
        device_config[f"model.layers.{i}.self_attn.k_proj"] = 'cpu'
        device_config[f"model.layers.{i}.self_attn.v_proj"] = 'cpu'
        device_config[f"model.layers.{i}.self_attn.o_proj"] = 'cpu'
        device_config[f"model.layers.{i}.mlp.gate_proj"] = 'cpu'
        device_config[f"model.layers.{i}.mlp.up_proj"] = 'cpu'
        device_config[f"model.layers.{i}.mlp.down_proj"] = 'cpu'
   
    # Set device map
    device_map = {}
    for name, _ in model.named_parameters():
        layer_name = ".".join(name.split(".")[:-1])
        if layer_name in device_config:
            device_map[layer_name] = 'cpu'
        else:
            device_map[layer_name] = device
    for name, _ in model.named_buffers():
        layer_name = ".".join(name.split(".")[:-1])
        device_map[layer_name] = device

    # Configs
    target_config = {
        "device_map": device_map,
        "quant_config": None,
    }
    draft_config = None
    
    return target_config, draft_config