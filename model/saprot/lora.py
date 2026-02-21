import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.lora_a = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def inject_last3_qv_lora(model, rank: int = 8, alpha: int = 16, dropout: float = 0.1):
    layers = model.esm.encoder.layer
    for layer in layers[-3:]:
        self_attn = layer.attention.self
        self_attn.query = LoRALinear(self_attn.query, rank=rank, alpha=alpha, dropout=dropout)
        self_attn.value = LoRALinear(self_attn.value, rank=rank, alpha=alpha, dropout=dropout)
    return model
