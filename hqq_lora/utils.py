import torch
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *
from hqq.utils.patching import recommended_inductor_config_setter, prepare_for_inference

def get_quantized_model(model, device):
    recommended_inductor_config_setter()
    quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

def apply_lora(model, lora_model_path="./results_lora/v0/checkpoint-19644/", device='cuda'):
    model.load_adapter(lora_model_path)

def prepare_model(model, device):
    prepare_for_inference(model) # optional: backend = 'gemlite'
    torch.cuda.empty_cache()