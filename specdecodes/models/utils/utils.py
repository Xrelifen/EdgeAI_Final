import torch
from dataclasses import dataclass

def invert_mask(mask, dtype): 
    # Inversion using bitwise NOT and multiplication
    return (~mask).to(dtype) * torch.finfo(dtype).min

@dataclass
class DraftParams:
    max_depth: int = 6
    topk_len: int = 10
    max_verify_tokens: int = 64
    min_accept_prob: float = 1e-2
    
    def __post_init__(self):
        self.max_sample_tokens = self.max_depth * self.topk_len + 1
        self.max_verify_tokens = min(self.max_sample_tokens, self.max_verify_tokens)