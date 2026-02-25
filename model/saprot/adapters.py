import torch
import torch.nn as nn


class AdapterBlock(nn.Module):
    """PreNorm adapter block with zero-init cross-attention output projection."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_proj = None

        self.query_norm = nn.LayerNorm(hidden_size)
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.zeros_(self.cross_attention.out_proj.weight)
        nn.init.zeros_(self.cross_attention.out_proj.bias)

        self.dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp_fc1 = nn.Linear(hidden_size, hidden_size * mlp_ratio)
        self.mlp_fc2 = nn.Linear(hidden_size * mlp_ratio, hidden_size)
        self.mlp = nn.Sequential(
            self.mlp_fc1,
            nn.GELU(),
            nn.Dropout(dropout),
            self.mlp_fc2,
            nn.Dropout(dropout),
        )
        # Keep adapter path as an exact no-op at step 0.
        # This mirrors: H_adapter = W2 * GELU(W1*(Out_attn + H_in) + b1) + b2 with W2=0.
        nn.init.zeros_(self.mlp_fc2.weight)
        nn.init.zeros_(self.mlp_fc2.bias)

    def set_feature_projector(self, projector: nn.Module):
        self.feature_proj = projector

    def forward(self, hidden_states: torch.Tensor, feature_states: torch.Tensor) -> torch.Tensor:
        if self.feature_proj is not None:
            feature_states = self.feature_proj(feature_states)

        q = self.query_norm(hidden_states)
        kv = self.feature_norm(feature_states)
        attn_out, _ = self.cross_attention(q, kv, kv, need_weights=False)
        hidden_states = hidden_states + self.dropout(attn_out)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states
