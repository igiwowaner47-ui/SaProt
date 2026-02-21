import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from ..model_interface import register_model
from ..abstract_model import AbstractModel
from .adapters import AdapterBlock
from .backbone import build_saprot_backbone, freeze_module
from .lora import inject_last3_qv_lora
from .mutation_head import MutationHead, compute_delta_score


class HierarchicalInjectedLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, adapter: AdapterBlock, feature_name: str):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter
        self.feature_name = feature_name
        self.external_features = None

    def set_external_features(self, external_features: dict):
        self.external_features = external_features

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.base_layer(hidden_states, *args, **kwargs)
        if isinstance(outputs, tuple):
            layer_hidden = outputs[0]
        else:
            layer_hidden = outputs

        if self.external_features is not None and self.feature_name in self.external_features:
            layer_hidden = self.adapter(layer_hidden, self.external_features[self.feature_name])

        if isinstance(outputs, tuple):
            return (layer_hidden,) + outputs[1:]
        return layer_hidden


@register_model
class SaprotHierarchicalMutationModel(AbstractModel):
    def __init__(
        self,
        config_path: str,
        extra_config: dict = None,
        load_pretrained: bool = True,
        freeze_backbone: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        profam_dim: int = 256,
        boltz_dim: int = 256,
        mpnn_dim: int = 256,
        gradient_checkpointing: bool = True,
        profam_encoder: nn.Module = None,
        boltz_encoder: nn.Module = None,
        mpnn_encoder: nn.Module = None,
        **kwargs,
    ):
        self.config_path = config_path
        self.extra_config = extra_config or {}
        self.load_pretrained = load_pretrained
        self.freeze_backbone_flag = freeze_backbone
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.external_dims = {"profam": profam_dim, "boltz": boltz_dim, "mpnn": mpnn_dim}
        self.gradient_checkpointing = gradient_checkpointing
        self.external_encoders = {
            "profam": profam_encoder,
            "boltz": boltz_encoder,
            "mpnn": mpnn_encoder,
        }
        super().__init__(**kwargs)

    def initialize_model(self):
        self.tokenizer, self.model = build_saprot_backbone(
            config_path=self.config_path,
            extra_config=self.extra_config,
            load_pretrained=self.load_pretrained,
        )
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads

        self.feature_aligners = nn.ModuleDict(
            {k: nn.Linear(v, self.hidden_size) for k, v in self.external_dims.items()}
        )

        self.adapters = nn.ModuleDict(
            {
                "profam": AdapterBlock(self.hidden_size, self.num_heads),
                "boltz": AdapterBlock(self.hidden_size, self.num_heads),
                "mpnn": AdapterBlock(self.hidden_size, self.num_heads),
            }
        )
        for name, adapter in self.adapters.items():
            adapter.set_feature_projector(self.feature_aligners[name])

        if self.use_lora:
            self.model = inject_last3_qv_lora(
                self.model,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )

        encoder_layers = self.model.esm.encoder.layer
        n_layers = len(encoder_layers)
        targets = [(n_layers - 3, "profam"), (n_layers - 2, "boltz"), (n_layers - 1, "mpnn")]
        self.injected_layers = []
        for idx, feature_name in targets:
            wrapped = HierarchicalInjectedLayer(encoder_layers[idx], self.adapters[feature_name], feature_name)
            encoder_layers[idx] = wrapped
            self.injected_layers.append(wrapped)

        self.mutation_head = MutationHead(self.hidden_size, num_labels=20)

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.freeze_backbone_flag:
            freeze_module(self.model)
            for encoder in self.external_encoders.values():
                if encoder is not None:
                    freeze_module(encoder)

        for name, p in self.model.named_parameters():
            if "lora_a" in name or "lora_b" in name:
                p.requires_grad = True

        for p in self.feature_aligners.parameters():
            p.requires_grad = True
        for p in self.adapters.parameters():
            p.requires_grad = True
        for p in self.mutation_head.parameters():
            p.requires_grad = True

    def initialize_metrics(self, stage):
        return {
            f"{stage}_loss": torchmetrics.MeanMetric(),
            f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
        }

    def forward(self, inputs, external_features=None, mutation_info=None):
        external_features = external_features or {}
        for layer in self.injected_layers:
            layer.set_external_features(external_features)

        outputs = self.model.esm(**inputs, return_dict=True)
        logits20 = self.mutation_head(outputs.last_hidden_state)

        if mutation_info is None:
            return {"scores": logits20}

        delta = compute_delta_score(
            logits20,
            mutation_info["positions"],
            mutation_info["wt_idx"],
            mutation_info["mut_idx"],
        )
        return {"scores": logits20, "delta_score": delta}

    def loss_func(self, stage, outputs, labels):
        target = labels.get("labels")
        if target is None:
            raise KeyError("labels must include key 'labels' for mutation regression targets")

        pred = outputs["delta_score"]
        loss = F.mse_loss(pred, target.to(pred))

        self.metrics[stage][f"{stage}_loss"].update(loss.detach())
        if pred.shape[0] > 1:
            self.metrics[stage][f"{stage}_spearman"].update(pred.detach().float(), target.float())

        if stage == "train":
            self.log_info(self.get_log_dict(stage))
            self.reset_metrics(stage)
        return loss

    def validation_epoch_end(self, outputs):
        self.log_info(self.get_log_dict("valid"))
        self.reset_metrics("valid")

    def test_epoch_end(self, outputs):
        self.log_info(self.get_log_dict("test"))
        self.reset_metrics("test")
