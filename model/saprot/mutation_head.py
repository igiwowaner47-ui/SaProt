import torch
import torch.nn as nn


class MutationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 20):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden_states))


def compute_delta_score(logits: torch.Tensor, positions: torch.Tensor, wt_idx: torch.Tensor, mut_idx: torch.Tensor):
    batch = torch.arange(logits.size(0), device=logits.device)
    selected = logits[batch, positions]
    wt_score = selected.gather(-1, wt_idx.unsqueeze(-1)).squeeze(-1)
    mut_score = selected.gather(-1, mut_idx.unsqueeze(-1)).squeeze(-1)
    return mut_score - wt_score
