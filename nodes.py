"""ComfyUI node definitions for BluePixel.

Nodes:
  - BluePixelPipelineLoader: validate + load model dir (FP32 or INT8),
    emit BLUEPIXEL_PIPE.
  - BluePixelGenerate: image editing at the 4K bucket area, emit IMAGE +
    metadata STRING. Edit-only; guidance is always disabled (distilled ckpt
    semantics, guidance=None single forward).

Import of this module must not load any weights (plan P3).
"""
from __future__ import annotations

import json
import logging

import torch

from .core.pipeline import BluePixelEngine
from .runtime import pipeline_loader
from .runtime.managed_models import (
    COMFY_AVAILABLE,
    interrupt_if_requested,
    load_patcher_gpu,
    unload_patcher,
)

logger = logging.getLogger("bluepixel.nodes")


class BluePixelPipelineLoader:
    """Load a BluePixel model directory (FP32 or INT8) into ComfyUI-managed components.

    INT8 models are auto-detected: when <model>/transformer/config.json carries
    quantization_config, the transformer is built with Int8Linear layers and
    vae/text_encoder/tokenizer/scheduler are taken from the FP32 base model
    (base_model dropdown or fallback to model itself).
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            paths = pipeline_loader.get_model_names()
        except Exception:
            paths = []
        if not paths:
            paths = ["<place model in ComfyUI/models/bluepixel/>"]
        return {
            "required": {
                "model_name": (paths,),
                "base_model_name": (paths,),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "int8_compute_dtype": (["fp32", "bf16"], {"default": "fp32"}),
                "vae_tiling": ("BOOLEAN", {"default": True}),
                "reload_token": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("BLUEPIXEL_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "bluepixel"

    def load(self, model_name, base_model_name, precision, int8_compute_dtype,
             vae_tiling, reload_token):
        pipe = pipeline_loader.validate_and_load(
            model_name=model_name,
            base_model_name=base_model_name,
            precision=precision,
            int8_compute_dtype=int8_compute_dtype,
            vae_tiling=vae_tiling,
            reload_token=reload_token,
        )
        return (pipe,)


class BluePixelGenerate:
    """BluePixel 4-step distilled IMAGE EDIT (edit-only; no CFG; 4K bucket area).

    Semantics follow the reference vins_infer.py (fb1366a):
      - guidance always disabled: single conditional forward per step
      - bucket resolution: 16 ratios, 4096x4096 area budget, 64 grid
      - CPS stepping (eta 0.9) over denoising timesteps [1000, 750, 500, 250]
      - mu_mode training (seq-len capped at 4096) or empirical
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("BLUEPIXEL_PIPE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 4}),
                "cps_eta": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mu_mode": (["training", "empirical"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "bluepixel"
    OUTPUT_NODE = False

    DESCRIPTION = (
        "BluePixel-Turbo image editing (edit-only, no CFG). Resolution comes "
        "from the bucket system with a 4096x4096 area budget; small inputs "
        "are BICUBIC resize+crop to the bucket before inference. Supports "
        "FP32 and INT8 models (INT8 transformer ~9GB weights). Components are "
        "loaded per stage, so only the one being used occupies VRAM: 4K INT8 "
        "peaks around 20GB instead of ~50GB."
    )

    def generate(self, pipe, image, prompt, seed, steps, cps_eta, mu_mode):
        pipeline_loader.check_generate_params(steps, mu_mode)
        if image is None or (hasattr(image, "shape") and image.shape[0] < 1):
            raise ValueError(
                "BluePixel is an image editing model: the required 'image' "
                "input is missing. Text-to-image is not supported."
            )

        if COMFY_AVAILABLE:
            import comfy.model_management
            device = comfy.model_management.get_torch_device()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- staged loading: only the component a stage needs stays resident.
        # This keeps the 4K INT8 peak around ~18GB instead of ~50GB when all
        # three components would be loaded at once.
        patchers = pipe["patchers"]
        stage_owner = {
            "text_encode": "text_encoder",
            "image_encode": "vae",
            "denoise": "transformer",
            "decode": "vae",
        }
        resident = {"name": None}

        def ensure_stage(stage: str) -> None:
            want = stage_owner[stage]
            if resident["name"] == want:
                return
            # Evict the previous component BEFORE loading the next one, so the
            # two never coexist on the GPU. Loading first and unloading after
            # lets both be resident at once, which is what pushes the 4K peak
            # up to ~24GB (Qwen3 + DCAE during the encode hand-off).
            prev = resident["name"]
            if prev is not None and prev != want:
                unload_patcher(patchers[prev])
            load_patcher_gpu([patchers[want]])
            resident["name"] = want

        engine = BluePixelEngine(
            transformer=pipe["transformer"],
            vae=pipe["vae"],
            text_encoder=pipe["text_encoder"],
            tokenizer=pipe["tokenizer"],
            # CPSScheduler reads attributes off the diffusers config object;
            # wrap the dict so attribute access works.
            scheduler=_AttrDictConfig(pipe["scheduler_config"]),
            weight_dtype={"bf16": torch.bfloat16, "fp16": torch.float16,
                          "fp32": torch.float32}[pipe["precision"]],
            device=device,
            is_int8=pipe["is_int8"],
        )

        images, metadata = engine.generate(
            image=image,
            instruction=prompt,
            seed=seed,
            num_steps=steps,
            cps_eta=cps_eta,
            mu_mode=mu_mode,
            interrupt_check=interrupt_if_requested,
            stage_cb=ensure_stage,
        )
        # release the last resident component so the next node/model can use VRAM
        if resident["name"] is not None:
            unload_patcher(patchers[resident["name"]])
            resident["name"] = None

        metadata.update({
            "model_path": pipe["model_path"],
            "base_model_path": pipe.get("base_model_path", pipe["model_path"]),
            "fingerprint": pipe["fingerprint"],
            "precision": pipe["precision"],
            "int8_compute_dtype": pipe.get("int8_compute_dtype"),
        })
        return (images, json.dumps(metadata, ensure_ascii=False, indent=2))


class _AttrDictConfig:
    """Expose scheduler config dict entries as attributes (for CPSScheduler)."""

    def __init__(self, d: dict):
        object.__setattr__(self, "_d", dict(d))

    @property
    def config(self):
        return self

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_d")[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_d")[name] = value


NODE_CLASS_MAPPINGS = {
    "BluePixelPipelineLoader": BluePixelPipelineLoader,
    "BluePixelGenerate": BluePixelGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BluePixelPipelineLoader": "BluePixel Pipeline Loader",
    "BluePixelGenerate": "BluePixel Generate",
}
