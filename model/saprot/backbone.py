import torch.nn as nn
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer


def build_saprot_backbone(config_path: str, extra_config: dict = None, load_pretrained: bool = True):
    tokenizer = EsmTokenizer.from_pretrained(config_path)
    config = EsmConfig.from_pretrained(config_path)
    for k, v in (extra_config or {}).items():
        setattr(config, k, v)

    if load_pretrained:
        model = EsmForMaskedLM.from_pretrained(config_path, **(extra_config or {}))
    else:
        model = EsmForMaskedLM(config)

    if hasattr(model, "lm_head"):
        model.lm_head = nn.Identity()

    return tokenizer, model


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False
