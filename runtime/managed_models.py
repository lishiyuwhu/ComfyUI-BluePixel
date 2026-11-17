"""ComfyUI-managed model wrappers.

Each BluePixel component (Qwen3 text encoder, Flux2 DiT, DCAE VAE) is wrapped
in a comfy.ModelPatcher so ComfyUI's memory manager handles device placement,
partial (lowvram) loading, unloading and cross-model eviction.

The adapter module (`comfy_ops_adapter`) converts diffusers/transformers
submodules to ComfyUI ops so weight streaming works; conversion happens on a
freshly instantiated model on the offload device before any ModelPatcher is
created (plan P2, no global monkey patching).

When ComfyUI is not importable (standalone tests), a fallback patcher keeps
the API shape so core code can still run models directly on `device`.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger("bluepixel.runtime")

try:  # ComfyUI present
    import comfy.model_management
    import comfy.model_patcher
    COMFY_AVAILABLE = True
except Exception:  # standalone test mode
    comfy = None
    COMFY_AVAILABLE = False


class _FallbackPatcher:
    """Minimal ModelPatcher stand-in for environments without ComfyUI."""

    def __init__(self, model, load_device, offload_device, size=0):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.size = size
        self.model_options = {}
        self._loaded = False

    def model_memory(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def model_loaded_memory(self) -> int:
        return self.model_memory() if self._loaded else 0

    def model_memory_required(self, device) -> int:
        return 0 if self._loaded else self.model_memory()

    def load_model_gpu(self):
        if not self._loaded or next(self.model.parameters()).device != self.load_device:
            self.model.to(self.load_device)
            self._loaded = True

    def model_unload(self, memory_to_free=0, unpatch_weights=True) -> bool:
        if not self._loaded:
            return False
        self.model.to(self.offload_device)
        self._loaded = False
        return True

    def cleanup(self):
        pass


class DeviceAttrShim(torch.nn.Module):
    """nn.Module wrapper exposing a plain writable `.device` attribute.

    transformers/diffusers ModelMixin classes define `device` as a read-only
    property, which breaks comfy.ModelPatcher's `model.device = ...` writes.
    The shim delegates everything else to the wrapped model.
    """

    def __init__(self, inner):
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        self.device = torch.device("cpu")

    @property
    def inner(self):
        return self._inner

    def forward(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        # called only when normal lookup fails; avoid recursion on _inner
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name == "device":
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)

    def parameters(self, recurse=True):
        return self._inner.parameters(recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self._inner.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        return self._inner.state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self._inner.to(*args, **kwargs)
        for p in self._inner.parameters():
            self.device = p.device
            break
        return self


def make_patcher(model, device=None, force_cpu_offload: bool = False):
    """Create a ComfyUI-managed patcher (or fallback) for `model`.

    device: target CUDA device; defaults to ComfyUI's current torch device.
    The model is expected to already live on CPU (offload_device) with ComfyUI
    ops applied; actual GPU load is deferred to load time. `model` is wrapped
    in a DeviceAttrShim so comfy can track/set `.device` (HF models expose a
    read-only property).
    """
    inner = model
    if not isinstance(model, DeviceAttrShim):
        model = DeviceAttrShim(model)
    if COMFY_AVAILABLE:
        load_device = comfy.model_management.get_torch_device() if device is None else torch.device(device)
        offload_device = torch.device("cpu") if force_cpu_offload else comfy.model_management.unet_offload_device()
        size = sum(p.numel() * p.element_size() for p in inner.parameters())
        return comfy.model_patcher.ModelPatcher(model, load_device=load_device,
                                                offload_device=offload_device, size=size)
    load_device = torch.device(device) if device is not None else torch.device("cpu")
    return _FallbackPatcher(model, load_device, torch.device("cpu"))


def load_patcher_gpu(patchers) -> None:
    """Ask ComfyUI to load one or more patchers onto GPU (or fallback)."""
    if COMFY_AVAILABLE:
        comfy.model_management.load_models_gpu(list(patchers))
        return
    for p in patchers:
        p.load_model_gpu()


def unload_patcher(patcher) -> None:
    """Evict a patcher from the GPU.

    ComfyUI tracks loaded models in `current_loaded_models`, and the patcher's
    weights are only released through that registry's LoadedModel.model_unload().
    Calling patcher.model_unload() directly leaves the registry stale, so the
    weights stay resident until something else forces a reload.
    """
    if COMFY_AVAILABLE:
        released = False
        try:
            mm = comfy.model_management
            # Identity match can fail because ComfyUI may hold a clone/parent
            # reference; fall back to is_clone() like load_models_gpu does.
            for loaded in list(mm.current_loaded_models):
                same = loaded.model is patcher
                if not same:
                    try:
                        same = patcher.is_clone(loaded.model)
                    except Exception:
                        same = False
                if same:
                    if loaded.model_unload():
                        mm.current_loaded_models.remove(loaded)
                        released = True
                    break
        except Exception:
            logger.warning("registry-based unload failed", exc_info=True)

        if not released:
            try:
                patcher.model_unload(unpatch_weights=True)
            except Exception:
                logger.warning("model_unload failed", exc_info=True)

        comfy.model_management.soft_empty_cache()
        return
    patcher.model_unload()


def interrupt_if_requested() -> None:
    """Raise ComfyUI's InterruptProcessingException when user pressed cancel."""
    if COMFY_AVAILABLE:
        comfy.model_management.throw_exception_if_processing_interrupted()
