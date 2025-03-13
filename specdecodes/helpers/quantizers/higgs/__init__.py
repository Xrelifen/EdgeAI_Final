from hqq.core.quantize import *
from hqq.utils.patching import prepare_for_inference
from .hf.base import AutoHiggsHFModel

class HiggsQuantizer:
    @classmethod
    def quantize_model(cls, model, quant_config, compute_dtype, device):
        AutoHiggsHFModel.quantize_model(model, quant_config, compute_dtype=compute_dtype, device=device)