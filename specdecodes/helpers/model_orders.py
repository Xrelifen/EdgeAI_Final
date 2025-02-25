
# order of layers in the model
def get_llama_layer_order(model_config):
    layers = []
    layers.append('model.embed_tokens')
    for i in range(model_config.num_hidden_layers):
        layers.append(f'model.layers.{i}.input_layernorm')
        layers.append(f'model.layers.{i}.self_attn.q_proj')
        layers.append(f'model.layers.{i}.self_attn.k_proj')
        layers.append(f'model.layers.{i}.self_attn.v_proj')
        layers.append(f'model.layers.{i}.self_attn.o_proj')
        layers.append(f'model.layers.{i}.mlp.gate_proj')
        layers.append(f'model.layers.{i}.mlp.up_proj')
        layers.append(f'model.layers.{i}.mlp.down_proj')
        layers.append(f'model.layers.{i}.post_attention_layernorm')
    layers.append('model.norm')
    layers.append('lm_head')
    return layers
        



MODEL_TYPE_GET_LAYER_ORDER = {
    "gemma": None,
    "gemma2": None,
    "llama": get_llama_layer_order,
    "granite": None,
    "mllama": None,
    "mllama_text_model": None,
    "mistral": None,
    "mixtral": None,
    "qwen2": None,
    "qwen2_vl": None,
    "phi3": None,
}