from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import importlib
import sys

import torch
import torch.nn as nn


@dataclass
class WrapperIOConfig:
    """IO contract for one external feature stream."""

    feature_key: str
    path_key: str
    input_key: str


class ExternalFeatureWrapper(nn.Module):
    """Lightweight wrapper to fetch external features for adapter injection.

    Priority:
    1) Read precomputed tensor from `batch[feature_key]`
    2) Read tensor file(s) from `batch[path_key]`
    3) Run frozen external encoder on `batch[input_key]`
    """

    def __init__(
        self,
        io_cfg: WrapperIOConfig,
        encoder: Optional[nn.Module] = None,
        dtype: torch.dtype = torch.float32,
        repo_path: Optional[str] = None,
        entrypoint: Optional[str] = None,
    ):
        super().__init__()
        self.io_cfg = io_cfg
        self.encoder = encoder
        self.dtype = dtype
        self.repo_path = repo_path
        self.entrypoint = entrypoint
        self.repo_extractor = None

        if self.encoder is not None:
            self.encoder.eval()
        elif self.repo_path and self.entrypoint:
            self.repo_extractor = self._load_repo_extractor(self.repo_path, self.entrypoint)

    def _load_repo_extractor(self, repo_path: str, entrypoint: str):
        root = Path(repo_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"repo_path does not exist: {root}")

        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

        module_name, fn_name = entrypoint.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        fn = getattr(mod, fn_name)
        if not callable(fn):
            raise TypeError(f"Entrypoint '{entrypoint}' is not callable")
        return fn

    def _run_repo_extractor(self, encoder_inputs: Any) -> torch.Tensor:
        if self.repo_extractor is None:
            raise KeyError(
                f"No repo extractor configured for '{self.io_cfg.feature_key}'."
            )
        with torch.no_grad():
            outputs = self.repo_extractor(encoder_inputs)

        if isinstance(outputs, dict):
            if "features" in outputs:
                outputs = outputs["features"]
            else:
                raise KeyError("Repo extractor dict output must include key 'features'.")

        if not torch.is_tensor(outputs):
            raise TypeError(f"Repo extractor output must be tensor or dict['features'], got {type(outputs)}")
        return outputs

    def _load_from_path(self, path_or_paths: Any) -> torch.Tensor:
        if isinstance(path_or_paths, (str, bytes)):
            return torch.load(path_or_paths, map_location="cpu")

        if isinstance(path_or_paths, Sequence):
            tensors = [torch.load(p, map_location="cpu") for p in path_or_paths]
            return torch.stack(tensors, dim=0)

        raise TypeError(f"Unsupported path container type: {type(path_or_paths)}")

    def _run_encoder(self, encoder_inputs: Any) -> torch.Tensor:
        if self.encoder is None and self.repo_extractor is None:
            raise KeyError(
                f"No encoder attached and no precomputed features found for '{self.io_cfg.feature_key}'."
            )

        if self.encoder is None and self.repo_extractor is not None:
            return self._run_repo_extractor(encoder_inputs)

        with torch.no_grad():
            if isinstance(encoder_inputs, dict):
                feats = self.encoder(**encoder_inputs)
            else:
                feats = self.encoder(encoder_inputs)

        if isinstance(feats, dict):
            if "features" in feats:
                feats = feats["features"]
            elif "last_hidden_state" in feats:
                feats = feats["last_hidden_state"]
            else:
                raise KeyError(
                    f"Encoder output dict for '{self.io_cfg.feature_key}' must include "
                    "'features' or 'last_hidden_state'."
                )

        if not torch.is_tensor(feats):
            raise TypeError(f"Encoder output must be a tensor or dict of tensors, got {type(feats)}")

        return feats

    def extract(self, batch: Dict[str, Any]) -> torch.Tensor:
        if self.io_cfg.feature_key in batch and batch[self.io_cfg.feature_key] is not None:
            feats = batch[self.io_cfg.feature_key]
        elif self.io_cfg.path_key in batch and batch[self.io_cfg.path_key] is not None:
            feats = self._load_from_path(batch[self.io_cfg.path_key])
        elif self.io_cfg.input_key in batch:
            feats = self._run_encoder(batch[self.io_cfg.input_key])
        else:
            raise KeyError(
                f"Missing external feature inputs for '{self.io_cfg.feature_key}'. Provide one of: "
                f"[{self.io_cfg.feature_key}, {self.io_cfg.path_key}, {self.io_cfg.input_key}]"
            )

        if not torch.is_tensor(feats):
            raise TypeError(f"Extracted feature for '{self.io_cfg.feature_key}' must be a tensor.")

        return feats.to(dtype=self.dtype)


class Boltz2Wrapper(ExternalFeatureWrapper):
    """Boltz-2 feature wrapper.

    Expected output is token-level features with shape [B, L, D].
    For online inference, we try common Boltz keys first.
    """

    _BOLTZ_CANDIDATE_KEYS = (
        "delta",
        "single_representation",
        "single",
        "features",
        "last_hidden_state",
    )

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        repo_path: Optional[str] = None,
        entrypoint: Optional[str] = None,
    ):
        super().__init__(
            io_cfg=WrapperIOConfig(
                feature_key="boltz_feat",
                path_key="boltz_feat_path",
                input_key="boltz_inputs",
            ),
            encoder=encoder,
            repo_path=repo_path,
            entrypoint=entrypoint,
        )

    def _run_encoder(self, encoder_inputs: Any) -> torch.Tensor:
        if self.encoder is None and self.repo_extractor is None:
            raise KeyError(
                f"No encoder attached and no precomputed features found for '{self.io_cfg.feature_key}'."
            )

        if self.encoder is None and self.repo_extractor is not None:
            feats = self._run_repo_extractor(encoder_inputs)
            if feats.dim() == 2:
                feats = feats.unsqueeze(0)
            if feats.dim() != 3:
                raise ValueError(
                    f"Boltz features must have shape [B, L, D] (or [L, D]). Got shape={tuple(feats.shape)}"
                )
            return feats

        with torch.no_grad():
            if isinstance(encoder_inputs, dict):
                outputs = self.encoder(**encoder_inputs)
            else:
                outputs = self.encoder(encoder_inputs)

        feats = outputs
        if isinstance(outputs, dict):
            feats = None
            for k in self._BOLTZ_CANDIDATE_KEYS:
                if k in outputs:
                    feats = outputs[k]
                    break
            if feats is None:
                raise KeyError(
                    "Boltz encoder output dict must include one of "
                    f"{self._BOLTZ_CANDIDATE_KEYS}, got keys={list(outputs.keys())}"
                )
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            feats = outputs[0]

        if not torch.is_tensor(feats):
            raise TypeError(f"Boltz features must be a tensor, got {type(feats)}")

        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        if feats.dim() != 3:
            raise ValueError(
                f"Boltz features must have shape [B, L, D] (or [L, D]). Got shape={tuple(feats.shape)}"
            )

        return feats


class ProFamWrapper(ExternalFeatureWrapper):
    """ProFam feature wrapper.

    Expected output is token-level features with shape [B, L, D].
    For online inference, we try common ProFam/LM output keys first.
    """

    _PROFAM_CANDIDATE_KEYS = (
        "features",
        "hidden_states",
        "last_hidden_state",
        "representations",
        "embeddings",
    )

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        repo_path: Optional[str] = None,
        entrypoint: Optional[str] = None,
    ):
        super().__init__(
            io_cfg=WrapperIOConfig(
                feature_key="profam_feat",
                path_key="profam_feat_path",
                input_key="profam_inputs",
            ),
            encoder=encoder,
            repo_path=repo_path,
            entrypoint=entrypoint,
        )

    def _run_encoder(self, encoder_inputs: Any) -> torch.Tensor:
        if self.encoder is None and self.repo_extractor is None:
            raise KeyError(
                f"No encoder attached and no precomputed features found for '{self.io_cfg.feature_key}'."
            )

        if self.encoder is None and self.repo_extractor is not None:
            feats = self._run_repo_extractor(encoder_inputs)
            if feats.dim() == 2:
                feats = feats.unsqueeze(0)
            if feats.dim() != 3:
                raise ValueError(
                    f"ProFam features must have shape [B, L, D] (or [L, D]). Got shape={tuple(feats.shape)}"
                )
            return feats

        with torch.no_grad():
            if isinstance(encoder_inputs, dict):
                outputs = self.encoder(**encoder_inputs)
            else:
                outputs = self.encoder(encoder_inputs)

        feats = outputs
        if isinstance(outputs, dict):
            feats = None
            for k in self._PROFAM_CANDIDATE_KEYS:
                if k in outputs:
                    feats = outputs[k]
                    break

            if isinstance(feats, dict) and -1 in feats:
                feats = feats[-1]

            if feats is None:
                raise KeyError(
                    "ProFam encoder output dict must include one of "
                    f"{self._PROFAM_CANDIDATE_KEYS}, got keys={list(outputs.keys())}"
                )
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            feats = outputs[0]

        if not torch.is_tensor(feats):
            raise TypeError(f"ProFam features must be a tensor, got {type(feats)}")

        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        if feats.dim() != 3:
            raise ValueError(
                f"ProFam features must have shape [B, L, D] (or [L, D]). Got shape={tuple(feats.shape)}"
            )

        return feats


class LigandMPNNWrapper(ExternalFeatureWrapper):
    """LigandMPNN feature wrapper.

    Expected output is token-level features with shape [B, L, D].
    For online inference, we try common MPNN-style keys first.
    """

    _LIGAND_MPNN_CANDIDATE_KEYS = (
        "features",
        "node_features",
        "decoder_hidden",
        "last_hidden_state",
        "hidden_states",
    )

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        repo_path: Optional[str] = None,
        entrypoint: Optional[str] = None,
    ):
        super().__init__(
            io_cfg=WrapperIOConfig(
                feature_key="ligandmpnn_feat",
                path_key="ligandmpnn_feat_path",
                input_key="ligandmpnn_inputs",
            ),
            encoder=encoder,
            repo_path=repo_path,
            entrypoint=entrypoint,
        )

    def _run_encoder(self, encoder_inputs: Any) -> torch.Tensor:
        if self.encoder is None and self.repo_extractor is None:
            raise KeyError(
                f"No encoder attached and no precomputed features found for '{self.io_cfg.feature_key}'."
            )

        if self.encoder is None and self.repo_extractor is not None:
            feats = self._run_repo_extractor(encoder_inputs)
            if feats.dim() == 2:
                feats = feats.unsqueeze(0)
            if feats.dim() != 3:
                raise ValueError(
                    "LigandMPNN features must have shape [B, L, D] (or [L, D]). "
                    f"Got shape={tuple(feats.shape)}"
                )
            return feats

        with torch.no_grad():
            if isinstance(encoder_inputs, dict):
                outputs = self.encoder(**encoder_inputs)
            else:
                outputs = self.encoder(encoder_inputs)

        feats = outputs
        if isinstance(outputs, dict):
            feats = None
            for k in self._LIGAND_MPNN_CANDIDATE_KEYS:
                if k in outputs:
                    feats = outputs[k]
                    break
            if feats is None:
                raise KeyError(
                    "LigandMPNN encoder output dict must include one of "
                    f"{self._LIGAND_MPNN_CANDIDATE_KEYS}, got keys={list(outputs.keys())}"
                )
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            feats = outputs[0]

        if not torch.is_tensor(feats):
            raise TypeError(f"LigandMPNN features must be a tensor, got {type(feats)}")

        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        if feats.dim() != 3:
            raise ValueError(
                "LigandMPNN features must have shape [B, L, D] (or [L, D]). "
                f"Got shape={tuple(feats.shape)}"
            )

        return feats
