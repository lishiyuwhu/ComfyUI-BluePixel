"""INT8 weight-only linear layer and quantized transformer loading.

Mirrors vins_infer.py (fb1366a) Int8Linear semantics exactly:
  y = F.linear(x, weight.to(x.dtype)) * weight_scale.to(x.dtype) + bias

The int8 checkpoint carries `quantization_config` in transformer/config.json
listing the 153 quantized `*.weight` keys and their per-output-channel
`*_scale` tensors.
"""
from __future__ import annotations

import glob
import json
import os
from os import path as osp
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """Weight-only int8 linear with per-output-channel scale."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        y = F.linear(x, w)
        y = y * self.weight_scale.to(x.dtype)
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)
        return y


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def read_quantization_config(transformer_dir: str) -> Optional[Dict]:
    """Return quantization_config from transformer/config.json, or None (fp32)."""
    cfg_path = osp.join(transformer_dir, "config.json")
    if not osp.isfile(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("quantization_config")


def load_int8_transformer(transformer_dir: str, build_from_config, compute_dtype: torch.dtype) -> nn.Module:
    """Build the transformer with Int8Linear layers and load int8 weights.

    build_from_config: callable(config_dict_without_quantization) -> nn.Module
    (injected so this module stays decoupled from diffusers).
    """
    with open(osp.join(transformer_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    qcfg = cfg.pop("quantization_config")
    qkeys = set(qcfg["quantized_keys"])
    suffix = qcfg.get("scale_suffix", "_scale")
    for key in [k for k in cfg if k.startswith("_")]:
        cfg.pop(key)

    with torch.device("meta"):
        model = build_from_config(cfg)

    n_replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and f"{name}.weight" in qkeys:
            _set_submodule(
                model,
                name,
                Int8Linear(module.in_features, module.out_features, module.bias is not None),
            )
            n_replaced += 1
    if n_replaced != len(qkeys):
        raise RuntimeError(
            f"int8 layer mismatch: replaced {n_replaced} Linear modules, "
            f"but quantization_config lists {len(qkeys)} keys"
        )

    state = {}
    for shard in sorted(glob.glob(osp.join(transformer_dir, "*.safetensors"))):
        with _safe_open(shard) as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    missing_scales = [k for k in qkeys if (k + suffix) not in state]
    if missing_scales:
        raise RuntimeError(f"missing scale tensors for {len(missing_scales)} quantized weights")

    model.load_state_dict(state, strict=True, assign=True)

    leftover = [
        n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta
    ]
    if leftover:
        raise RuntimeError(f"tensors still on meta after load: {leftover[:5]}")

    for p in model.parameters():
        p.data = p.data.to(compute_dtype)

    return model


def _safe_open(shard_path: str):
    from safetensors.torch import safe_open
    return safe_open(shard_path, "pt", device="cpu")


def is_int8_model_dir(model_dir: str) -> bool:
    """True when <model_dir>/transformer/config.json carries quantization_config."""
    return read_quantization_config(osp.join(model_dir, "transformer")) is not None
