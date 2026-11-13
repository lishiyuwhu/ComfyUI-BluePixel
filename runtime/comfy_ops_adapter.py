"""Convert diffusers/HF modules to ComfyUI operations for lowvram streaming.

Strategy (plan P2): instead of global monkey patching, each model instance is
walked once on the offload device and its nn.Linear / nn.Conv2d / nn.LayerNorm
/ nn.Embedding modules are swapped for ComfyUI `operations` equivalents that
carry `comfy_cast_weights`. ModelPatcher.partial_load then streams weights
layer-by-layer on low-VRAM GPUs.

Weight key maps: ComfyUI patcher needs state-dict compatible modules; we keep
the diffusers module tree as-is, only replacing leaf op classes, so state dict
keys stay identical to the HF checkpoint (verified by a strict-ish load test).
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch

logger = logging.getLogger("bluepixel.ops")

try:
    import comfy.ops
    import comfy.model_management
    COMFY_AVAILABLE = True
except Exception:
    comfy = None
    COMFY_AVAILABLE = False


def _make_comfy_ops(device=None):
    if COMFY_AVAILABLE:
        return comfy.ops.disable_weight_init
    return None


def convert_to_comfy_ops(model: torch.nn.Module) -> Tuple[torch.nn.Module, Dict[str, int]]:
    """In-place-convert leaf ops of `model` to ComfyUI streaming ops.

    Returns (model, stats) where stats counts each converted op type.
    The model must be on CPU/meta-free weights when this runs.
    """
    if not COMFY_AVAILABLE:
        return model, {}

    ops = _make_comfy_ops()
    stats: Dict[str, int] = {}

    for name, module in list(model.named_children()):
        converted = _convert_module(module, ops, stats)
        if converted is not module:
            setattr(model, name, converted)
        else:
            convert_to_comfy_ops(module)
    return model, stats


def _convert_module(module: torch.nn.Module, ops, stats: Dict[str, int]):
    """Return a replacement module or the same module to recurse into."""
    if isinstance(module, torch.nn.Linear):
        out = ops.Linear(module.in_features, module.out_features,
                         bias=module.bias is not None,
                         dtype=module.weight.dtype, device=module.weight.device)
        _copy_weights(module, out)
        stats["linear"] = stats.get("linear", 0) + 1
        return out
    if isinstance(module, torch.nn.Conv2d):
        out = ops.Conv2d(module.in_channels, module.out_channels,
                         kernel_size=module.kernel_size, stride=module.stride,
                         padding=module.padding, dilation=module.dilation,
                         groups=module.groups, bias=module.bias is not None,
                         dtype=module.weight.dtype, device=module.weight.device)
        _copy_weights(module, out)
        stats["conv2d"] = stats.get("conv2d", 0) + 1
        return out
    if isinstance(module, torch.nn.Embedding):
        out = ops.Embedding(module.num_embeddings, module.embedding_dim,
                            dtype=module.weight.dtype, device=module.weight.device)
        out.weight.data.copy_(module.weight.data)
        stats["embedding"] = stats.get("embedding", 0) + 1
        return out
    if isinstance(module, torch.nn.LayerNorm):
        out = ops.LayerNorm(module.normalized_shape, eps=module.eps,
                            elementwise_affine=module.elementwise_affine,
                            bias=module.bias is not None,
                            dtype=module.weight.dtype if module.elementwise_affine else None,
                            device=module.weight.device if module.elementwise_affine else None)
        if module.elementwise_affine:
            _copy_weights(module, out)
        stats["layernorm"] = stats.get("layernorm", 0) + 1
        return out
    # recurse into containers
    for child_name, child in list(module.named_children()):
        converted = _convert_module(child, ops, stats)
        if converted is not child:
            setattr(module, child_name, converted)
        else:
            _convert_module(child, ops, stats)
    return module


def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    with torch.no_grad():
        dst.weight.data.copy_(src.weight.data)
        if src.bias is not None and dst.bias is not None:
            dst.bias.data.copy_(src.bias.data)


def model_supports_lowvram(model: torch.nn.Module) -> bool:
    """True when every Linear/Conv leaf carries comfy_cast_weights support."""
    if not COMFY_AVAILABLE:
        return False
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            if not hasattr(m, "comfy_cast_weights"):
                return False
    return True
