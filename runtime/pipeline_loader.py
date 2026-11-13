"""BluePixel model loading for ComfyUI.

Loads Qwen3 text encoder, Flux2 DiT and DCAE VAE from a BluePixel-Turbo
diffusers-layout directory, converts to ComfyUI ops, and wraps each component
in a ModelPatcher. No network access; all weights come from local files.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional

import torch

from . import loader as loader_mod
from .comfy_ops_adapter import model_supports_lowvram
from .managed_models import make_patcher

logger = logging.getLogger("bluepixel.loader")

PREDEFINED_MODELS = "bluepixel"
_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()


def _register_folder() -> None:
    """Register the 'bluepixel' folder name with ComfyUI (idempotent).

    The folder is registered with {'.json'} as file extension so that the
    framework's filename list returns `<ModelName>/model_index.json` entries
    (one per complete diffusers-layout model directory, including dirs
    contributed by extra_model_paths.yaml). get_filename_list() provides
    framework-managed caching and recursion.
    """
    try:
        import folder_paths
        if PREDEFINED_MODELS not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                PREDEFINED_MODELS,
                os.path.join(folder_paths.models_dir, PREDEFINED_MODELS),
            )
            # narrow the extension filter to model_index.json only
            paths = folder_paths.folder_names_and_paths[PREDEFINED_MODELS][0]
            folder_paths.folder_names_and_paths[PREDEFINED_MODELS] = (paths, {".json"})
    except Exception:
        pass  # standalone/test mode without ComfyUI


MODEL_INDEX_MARKER = "model_index.json"


def get_model_names() -> list[str]:
    """Registered model names for UI dropdowns.

    Each name is a relative directory path whose entry `<name>/model_index.json`
    exists in the framework filename list, e.g. "BluePixel-Turbo" or
    "nested/MyModel".
    """
    _register_folder()
    try:
        import folder_paths
        entries = folder_paths.get_filename_list(PREDEFINED_MODELS)
    except Exception:
        return []
    return sorted({e[: -(len(MODEL_INDEX_MARKER) + 1)] for e in entries
                   if e.endswith("/" + MODEL_INDEX_MARKER)})


def _resolve_registered_name(name: str) -> Optional[str]:
    """Resolve a registered model name to its directory (path-traversal safe).

    Only names whose `<name>/model_index.json` exists in a registered root
    are accepted; anything else (including absolute paths and `..`) returns
    None.
    """
    try:
        import folder_paths
        full = folder_paths.get_full_path(
            PREDEFINED_MODELS, os.path.join(name, MODEL_INDEX_MARKER))
        if full is not None:
            return os.path.dirname(full)
    except Exception:
        pass
    return None


def get_model_paths() -> list[str]:
    """Model directories for all registered names (debug/helper use)."""
    out = []
    for name in get_model_names():
        resolved = _resolve_registered_name(name)
        if resolved is not None:
            out.append(resolved)
    return out


def _resolve_model_dir(model_name: str) -> str:
    """Resolve a model NAME from the registered dropdown list to its dir.

    Security: only names resolvable through the framework's registered roots
    are accepted; arbitrary absolute paths are rejected.
    """
    _register_folder()
    resolved = _resolve_registered_name(model_name)
    if resolved is None:
        available = get_model_names()
        raise FileNotFoundError(
            f"BluePixel model '{model_name}' is not in the registered list. "
            f"Place the model dir under ComfyUI/models/{PREDEFINED_MODELS}/ "
            f"(or register it via extra_model_paths.yaml) and restart. "
            f"Available: {available}"
        )
    return resolved


def validate_and_load(
    model_name: str,
    precision: str = "bf16",
    vae_tiling: bool = False,
    reload_token: str = "",
    device=None,
    base_model_name: str = "",
    int8_compute_dtype: str = "fp32",
) -> dict:
    """Load or return cached BluePixel components.

    INT8 auto-detection: when <model>/transformer/config.json carries a
    quantization_config, the transformer is built with Int8Linear layers
    (vins_infer fb1366a semantics); vae/text_encoder/tokenizer/scheduler
    then come from base_model (the FP32 release).

    Returns an entry dict with components + patchers + metadata.
    """
    from ..core.int8_linear import read_quantization_config

    if int8_compute_dtype not in ("fp32", "bf16"):
        raise ValueError(f"int8_compute_dtype must be fp32|bf16, got {int8_compute_dtype!r}")

    model_path = _resolve_model_dir(model_name)
    base_path = _resolve_model_dir(base_model_name) if base_model_name else model_path
    transformer_dir = os.path.join(model_path, "transformer")
    if not os.path.isdir(transformer_dir):
        raise FileNotFoundError(f"transformer/ not found under {model_path}")

    qcfg = read_quantization_config(transformer_dir)
    is_int8 = qcfg is not None

    # fingerprint over the actual source dirs (int8: transformer from model, rest from base)
    fp_transformer = loader_mod.compute_fingerprint(model_path)
    fp_base = loader_mod.compute_fingerprint(base_path)
    fingerprint = fp_transformer if not is_int8 else f"{fp_transformer[:8]}+{fp_base[:8]}(int8)"
    cache_key = (f"{os.path.abspath(model_path)}|{os.path.abspath(base_path)}|"
                 f"{precision}|{int8_compute_dtype}|{fingerprint}|{reload_token}")

    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    # validate: transformer files from model_path; the rest from base_path
    ok_t, problems_t = loader_mod.validate_model_dir(model_path)
    if not ok_t:
        raise RuntimeError(
            "BluePixel model validation failed:\n  - " + "\n  - ".join(problems_t))
    if base_path != model_path:
        ok_b, problems_b = loader_mod.validate_model_dir(base_path)
        if not ok_b:
            raise RuntimeError(
                "BluePixel base model validation failed:\n  - " + "\n  - ".join(problems_b))

    scaling_factor = loader_mod.read_vae_scaling_factor(base_path)
    if scaling_factor is None:
        raise RuntimeError("vae/config.json missing scaling_factor")
    t_cfg = loader_mod.read_transformer_config(model_path)
    if t_cfg is None:
        raise RuntimeError("transformer/config.json unreadable")
    if int(t_cfg.get("in_channels", -1)) != 128:
        raise RuntimeError(f"unexpected transformer in_channels {t_cfg.get('in_channels')} (want 128)")

    if precision == "bf16":
        dtype_enc = torch.bfloat16
    elif precision == "fp16":
        dtype_enc = torch.float16
    elif precision == "fp32":
        dtype_enc = torch.float32
    else:
        raise ValueError(f"unsupported precision {precision!r} (use bf16|fp16|fp32)")

    from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler
    from diffusers import Flux2Transformer2DModel
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

    if is_int8:
        from ..core.int8_linear import load_int8_transformer

        logger.info("BluePixel loading INT8 transformer from %s (compute=%s)",
                    model_path, int8_compute_dtype)
        transformer = load_int8_transformer(
            transformer_dir,
            build_from_config=lambda cfg: Flux2Transformer2DModel.from_config(cfg),
            compute_dtype=torch.float32 if int8_compute_dtype == "fp32" else torch.bfloat16,
        )
        logger.info("BluePixel int8 transformer: %d Int8Linear layers",
                    len(qcfg["quantized_keys"]))
    else:
        logger.info("BluePixel loading FP32 transformer from %s", model_path)
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.float32, local_files_only=True,
        )
    # fp32 release is loaded in fp32 (reference behavior); component dtype applies
    # to text encoder only. weight_dtype == text encoder dtype.
    transformer.requires_grad_(False)
    transformer.eval()

    logger.info("BluePixel loading text encoder (Qwen3) from %s", base_path)
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        base_path, subfolder="text_encoder", torch_dtype=dtype_enc, local_files_only=True,
    ).to(torch.device("cpu"))
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    logger.info("BluePixel loading VAE (DCAE) from %s", base_path)
    vae = AutoencoderDC.from_pretrained(
        base_path, subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
    ).to(torch.device("cpu"))
    vae.requires_grad_(False)
    vae.eval()

    if vae_tiling:
        try:
            vae.enable_tiling()
            logger.info("BluePixel VAE tiling enabled")
        except Exception:
            logger.warning("DCAE enable_tiling failed; running untiled", exc_info=True)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        base_path, subfolder="tokenizer", local_files_only=True,
    )
    scheduler_cfg = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_path, subfolder="scheduler", local_files_only=True,
    )
    sched_conf = dict(scheduler_cfg.config)
    del scheduler_cfg

    t_patcher = make_patcher(transformer, device=device)
    e_patcher = make_patcher(text_encoder, device=device)
    v_patcher = make_patcher(vae, device=device)

    entry = {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "vae": vae,
        "tokenizer": tokenizer,
        "scheduler_config": sched_conf,
        "scaling_factor": scaling_factor,
        "fingerprint": fingerprint,
        "model_path": os.path.abspath(model_path),
        "base_model_path": os.path.abspath(base_path),
        "is_int8": is_int8,
        "int8_compute_dtype": int8_compute_dtype if is_int8 else None,
        "patchers": {"transformer": t_patcher, "text_encoder": e_patcher, "vae": v_patcher},
        "precision": precision,
        "lowvram_ready": model_supports_lowvram(transformer),
        "vae_tiling_default": bool(vae_tiling),
    }
    with _CACHE_LOCK:
        _CACHE[cache_key] = entry
    return entry


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def check_generate_params(steps: int, mu_mode: str) -> None:
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if mu_mode not in ("training", "empirical"):
        raise ValueError(f"mu_mode must be training|empirical, got {mu_mode!r}")
