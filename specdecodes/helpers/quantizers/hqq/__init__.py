from hqq.core.quantize import *
from hqq.utils.patching import prepare_for_inference
from .hf.base import AutoHQQHFModel

class HqqQuantizer:
    @classmethod
    def quantize_model(cls, model, quant_config, compute_dtype, device):
        AutoHQQHFModel.quantize_model(model, quant_config, compute_dtype=compute_dtype, device=device)
        HQQLinear.set_backend(HQQBackend.PYTORCH)
        prepare_for_inference(model, backend="gemlite")