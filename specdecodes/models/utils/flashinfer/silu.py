import torch
import torch.nn as nn

from flashinfer.activation import silu_and_mul


class FiLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        if config.hidden_act not in ["silu"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        concat = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(silu_and_mul(concat, enable_pdl=True))
