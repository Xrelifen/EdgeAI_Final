from hqq.core.quantize import *


def estimate_quantized_size(model, quant_config, max_input_len=0):
    weight_bytes = 0
    for name, param in model.named_parameters():
        layer_name = ".".join(name.split(".")[:-1])
        if layer_name in quant_config:
            nbits = quant_config[layer_name]['weight_quant_params']['nbits']
            group_size = quant_config[layer_name]['weight_quant_params']['group_size']
            weight_bytes += param.numel() * param.element_size() * nbits / 16 # quantized weight
            weight_bytes += param.numel() * param.element_size() / group_size * 2 # scale and zero
        else:
            weight_bytes += param.numel() * param.element_size()
    
    for name, param in model.named_buffers():
        weight_bytes += param.numel() * param.element_size()
        
    element_size = next(iter(model.parameters())).element_size() # assume activation has same element size as first param
    head_size = model.config.hidden_size // model.config.num_attention_heads
    activation_bytes = 2 * max_input_len * model.config.num_hidden_layers * model.config.num_key_value_heads * head_size * element_size # key and value cache
    return weight_bytes + activation_bytes

def recipe(model, draft_model, max_length, vram_limit, dtype, device):
    # Quantization
    quant_config = {}
    base_quant_config = BaseQuantizeConfig(nbits=4, group_size=32, axis=1)
    
    layer_cnt = len(model.model.layers)
    quant_start = 0
    quant_end = layer_cnt - 1
    for i in range(quant_start, quant_end+1):
        # quant_config[f"layers.{i}.self_attn.q_proj"] = base_quant_config_a
        # quant_config[f"layers.{i}.self_attn.k_proj"] = base_quant_config_a
        # quant_config[f"layers.{i}.self_attn.v_proj"] = base_quant_config_a
        # quant_config[f"layers.{i}.self_attn.o_proj"] = base_quant_config_a
        quant_config[f"model.layers.{i}.mlp.gate_proj"] = base_quant_config
        quant_config[f"model.layers.{i}.mlp.up_proj"] = base_quant_config
        quant_config[f"model.layers.{i}.mlp.down_proj"] = base_quant_config
        
    max_input_len = 1024
    estimate_qmodel_size = estimate_quantized_size(model, quant_config, max_length)
    print(f"Estimated required VRAM for {max_input_len} input tokens: {estimate_qmodel_size / 1024**3:.2f} GiB")
        
    # Device map
    device_map = {}
    for name, _ in model.named_parameters():
        layer_name = ".".join(name.split(".")[:-1])
        if layer_name in quant_config:
            device_map[layer_name] = 'cpu'
        else:
            device_map[layer_name] = device
    for name, _ in model.named_buffers():
        layer_name = ".".join(name.split(".")[:-1])
        device_map[layer_name] = device

    # Configs
    target_config = {
        "device_map": device_map,
    }
    draft_config = {
        "quant_config": {
            "config": quant_config,
            "backend": "gemlite", #"torchao_int4",
        },
    }
    
    return target_config, draft_config