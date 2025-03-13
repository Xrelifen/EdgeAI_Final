import logging
from .hf.base import AutoHiggsHFModel

class HiggsQuantizer:
    @classmethod
    def quantize_model(cls, model, quant_config, compute_dtype, device):
        logging.info("Quantizing model with HiggsQuantizer. First few iterations may be very slow.")
        AutoHiggsHFModel.quantize_model(model, quant_config, compute_dtype=compute_dtype, device=device)