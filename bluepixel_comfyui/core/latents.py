"""Latent packing, position ids and reference-image conditioning helpers.

Extracted from diffusers Flux2Pipeline static helpers (pinned to the semantics
BluePixel relies on) plus the vins_infer prepare_latents flow. All functions
are pure tensor math and unit-testable on CPU.
"""
from __future__ import annotations

import torch
from PIL import Image

VAE_SCALE_FACTOR = 64          # BluePixel: DCAE f64, then /64 for latent grid
LATENT_CHANNELS = 128          # transformer.config.in_channels
REFERENCE_ID_TIME_SCALE = 10   # _prepare_image_ids scale


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)."""
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)


def unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    """(B, N, C) + ids (B, N, 4) -> (B, C, H, W) via scatter on (h, w) ids."""
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = h_ids * w + w_ids

        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list, dim=0)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, 4) ids with T=0, L=0, h/w grid order.

    Matches diffusers _prepare_latent_ids: cartesian_prod(t, h, w, l) which
    yields row-major (h, w) ordering, i.e. token index = h * W + w.
    """
    batch_size, _, height, width = latents.shape
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)


def prepare_image_ids(image_latents: list[torch.Tensor], scale: int = REFERENCE_ID_TIME_SCALE) -> torch.Tensor:
    """list of (1, C, H, W) -> (1, sum(H*W), 4) ids, T = scale * (i+1)."""
    if not isinstance(image_latents, list):
        raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")
    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for x, t in zip(image_latents, t_coords):
        x = x.squeeze(0)
        _, height, width = x.shape
        x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
        image_latent_ids.append(x_ids)
    image_latent_ids = torch.cat(image_latent_ids, dim=0)
    return image_latent_ids.unsqueeze(0)


def prepare_reference_latents(vae, images: list[torch.Tensor], batch_size: int,
                              device, dtype: torch.dtype, scaling_factor: float):
    """Encode reference images with the DCAE and pack with ids.

    images: list of preprocessed (1, 3, H, W) tensors in [-1, 1] (already on
    any device; moved internally). Returns (image_latents, image_latent_ids):
    - image_latents: (B, N_total, 128)
    - image_latent_ids: (B, N_total, 4)
    """
    image_latent_list = []
    for image in images:
        image = image.to(device=device, dtype=dtype)
        latent = vae.encode(image)[0]
        latent = latent * scaling_factor
        image_latent_list.append(latent)

    image_latent_ids = prepare_image_ids(image_latent_list)

    packed = []
    for latent in image_latent_list:
        p = pack_latents(latent).squeeze(0)
        packed.append(p)
    image_latents = torch.cat(packed, dim=0).unsqueeze(0)
    image_latents = image_latents.repeat(batch_size, 1, 1)
    image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1).to(device)
    return image_latents, image_latent_ids


def build_noise_latents(batch_size: int, height: int, width: int, dtype: torch.dtype,
                        device, generator, vae_scale_factor: int = VAE_SCALE_FACTOR,
                        latents: torch.Tensor | None = None):
    """Random noise latents + ids, pre-pack. Matches vins prepare_latents."""
    lat_h = int(height) // vae_scale_factor
    lat_w = int(width) // vae_scale_factor
    shape = (batch_size, LATENT_CHANNELS, lat_h, lat_w)
    if latents is None:
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device=device, dtype=dtype)
    latent_ids = prepare_latent_ids(latents).to(device)
    latents_packed = pack_latents(latents)
    return latents_packed, latent_ids, latents


def resize_condition_image(image, target_pixels: int = 0, processor_preprocess=None):
    """BluePixel reference resize: target-pixel scale then 64-grid pick, LANCZOS.

    image: PIL.Image. Returns (tensor (1,3,h,w) in [-1,1], (w, h)).
    processor_preprocess: callable(image, height, width) -> tensor; when None a
    simple ToTensor-style normalize is used (kept injectable for tests).

    NOTE: run_bluepixel does NOT use this anymore — it computes (width, height)
    itself and pre-resizes the input, then converts via cond_pil_to_tensor.
    Kept for standalone/test use of the reference sizing rule.
    """
    img = image.convert("RGB")
    w, h = img.size
    if target_pixels > 0:
        w, h = _scale_to_target_pixels_pil(w, h, target_pixels)
    ow, oh = img.size
    target_ratio = ow / oh
    cw, ch = _pick_size_64(w, h, target_ratio)
    img = img.resize((cw, ch), Image.LANCZOS)
    return cond_pil_to_tensor(img, processor_preprocess), (cw, ch)


def cond_pil_to_tensor(image, processor_preprocess=None):
    """PIL image (already at inference resolution) -> (1, 3, H, W) in [-1, 1]."""
    img = image.convert("RGB")
    w, h = img.size
    if processor_preprocess is not None:
        return processor_preprocess(img, height=h, width=w)
    import torchvision.transforms.functional as TF
    return TF.to_tensor(img).sub_(0.5).div_(0.5).unsqueeze(0)


def _scale_to_target_pixels_pil(width: int, height: int, target_pixels: int) -> tuple[int, int]:
    scale = (target_pixels / (width * height)) ** 0.5
    return max(1, int(width * scale)), max(1, int(height * scale))


def _pick_size_64(w: int, h: int, target_ratio: float) -> tuple[int, int]:
    """Same search as vins_infer (kept local so tests don't need PIL)."""
    from .helpers import pick_reference_size
    return pick_reference_size(w, h)
