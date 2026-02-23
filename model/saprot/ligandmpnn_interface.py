from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn


@dataclass
class LigandMPNNBatch:
    """Canonical interface input for LigandMPNN/ProteinMPNN geometry extraction.

    Args:
        aa_seq: token indices or one-hot AA tensor, shape [B, L] or [B, L, A].
        wt_name: optional WT token indices, shape [B, L]. If None, aa_seq is used.
        coords: optional backbone coordinates [B, L, 4, 3] for N/CA/C/O.
        pdb_path: optional source path string or list[str] (for bookkeeping / upstream parser).
    """

    aa_seq: torch.Tensor
    wt_name: Optional[torch.Tensor] = None
    coords: Optional[torch.Tensor] = None
    pdb_path: Optional[Union[str, list[str]]] = None


class LigandMPNNGeometryAdapter(nn.Module):
    """Build token-level Z_geo features from MPNN decoder + WT AA embeddings.

    Required logic:
    - Full-context (non-autoregressive) decoder mask.
    - Use deep decoder node representations (e.g., Vdec).
    - Gather WT amino-acid embeddings (EAA).
    - Fuse and map to geometry features `features = Z_geo`, shape [B, L, D_geo_raw].
    """

    def __init__(
        self,
        decoder: nn.Module,
        aa_embedding: nn.Embedding,
        dec_dim: int,
        aa_dim: int,
        d_geo_raw: int = 256,
        fuse_mode: str = "concat",
    ):
        super().__init__()
        if fuse_mode not in {"concat", "sum"}:
            raise ValueError("fuse_mode must be one of {'concat', 'sum'}")

        self.decoder = decoder
        self.aa_embedding = aa_embedding
        self.d_geo_raw = d_geo_raw
        self.fuse_mode = fuse_mode

        if fuse_mode == "concat":
            self.geo_proj = nn.Linear(dec_dim + aa_dim, d_geo_raw)
        else:
            hidden = max(dec_dim, aa_dim)
            self.dec_align = nn.Linear(dec_dim, hidden)
            self.aa_align = nn.Linear(aa_dim, hidden)
            self.geo_proj = nn.Linear(hidden, d_geo_raw)

    @staticmethod
    def build_full_context_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return non-autoregressive full-context mask. Shape [B, L, L]."""

        return torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)

    def _normalize_token_tensor(self, tensor: torch.Tensor, field_name: str) -> torch.Tensor:
        if tensor.dim() not in (2, 3):
            raise ValueError(f"{field_name} must have shape [B, L] or [B, L, A]")
        if tensor.dim() == 3:
            return tensor.argmax(dim=-1)
        return tensor

    def _decode_vdec(self, aa_idx: torch.Tensor, coords: Optional[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        decoder_out = self.decoder(aa_idx=aa_idx, coords=coords, full_context_mask=mask)
        if isinstance(decoder_out, dict):
            if "Vdec" in decoder_out:
                return decoder_out["Vdec"]
            if "node_repr" in decoder_out:
                return decoder_out["node_repr"]
            raise KeyError("decoder output dict must include key 'Vdec' or 'node_repr'")
        return decoder_out

    def forward(self, batch: LigandMPNNBatch) -> Dict[str, torch.Tensor]:
        """Return a dict with geometry feature aliases for downstream compatibility.

        Returns:
            {
              "features": Z_geo,
              "Z_geo": Z_geo,
            }
            where Z_geo has shape [B, L, D_geo_raw].
        """

        if batch.coords is None and batch.pdb_path is None:
            raise ValueError("At least one of `coords` or `pdb_path` should be provided")

        aa_idx = self._normalize_token_tensor(batch.aa_seq, "aa_seq")
        wt_idx = self._normalize_token_tensor(batch.wt_name, "wt_name") if batch.wt_name is not None else aa_idx

        bsz, seqlen = aa_idx.shape
        if wt_idx.shape != aa_idx.shape:
            raise ValueError("wt_name and aa_seq must have the same [B, L] shape after normalization")
        mask = self.build_full_context_mask(bsz, seqlen, aa_idx.device)

        vdec = self._decode_vdec(aa_idx=aa_idx, coords=batch.coords, mask=mask)
        if vdec.shape[:2] != (bsz, seqlen):
            raise ValueError("Decoder output must have leading shape [B, L, D]")

        eaa = self.aa_embedding(wt_idx)

        if self.fuse_mode == "concat":
            fused = torch.cat([vdec, eaa], dim=-1)
        else:
            fused = self.dec_align(vdec) + self.aa_align(eaa)

        z_geo = self.geo_proj(fused)
        return {"features": z_geo, "Z_geo": z_geo}
