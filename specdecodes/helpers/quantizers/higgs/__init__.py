import logging
from .hf.base import AutoHiggsHFModel

class HiggsQuantizer:
    @classmethod
    def quantize_model(cls, model, quant_config, compute_dtype, device):
        logging.info("Quantizing model with HiggsQuantizer. First few iterations may be much slower.")
        AutoHiggsHFModel.quantize_model(model, quant_config, compute_dtype=compute_dtype, device=device)