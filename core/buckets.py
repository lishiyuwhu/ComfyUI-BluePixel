"""Resolution bucket system (vins_infer fb1366a semantics).

Aligned target resolutions built with Klein's area convention:
  - 16 predefined aspect ratios, target area 4096*4096 (4K budget)
  - sides rounded DOWN to the 64 grid (min side 64), area clamped <= target
  - selection by minimal log-ratio distance to the input image
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterator, Tuple

import torch

DEFAULT_BUCKET_RATIOS = (
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1",
    "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
)
DEFAULT_BUCKET_AREA = 4096 * 4096
DEFAULT_BUCKET_MULTIPLE = 64


def parse_ratio(value: str) -> Tuple[int, int]:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid bucket ratio {value!r}; expected W:H")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bucket ratio {value!r}; values must be positive")
    divisor = math.gcd(width, height)
    return width // divisor, height // divisor


class ResolutionBucketSet:
    """Aligned target resolutions built with Klein's area convention."""

    def __init__(
        self,
        ratios=DEFAULT_BUCKET_RATIOS,
        *,
        target_area: int = DEFAULT_BUCKET_AREA,
        multiple: int = DEFAULT_BUCKET_MULTIPLE,
        min_side: int = 64,
    ) -> None:
        if target_area <= 0:
            raise ValueError("target_area must be positive")
        if multiple <= 0 or min_side <= 0:
            raise ValueError("multiple and min_side must be positive")
        self.target_area = int(target_area)
        self.multiple = int(multiple)
        self.min_side = max(int(min_side), self.multiple)
        self.ratio_pairs = tuple(parse_ratio(r) for r in ratios)
        if not self.ratio_pairs:
            raise ValueError("At least one resolution bucket is required")
        self.buckets = tuple(
            self._make_bucket(i, pair) for i, pair in enumerate(self.ratio_pairs)
        )

    def _make_bucket(self, bucket_id: int, pair: Tuple[int, int]) -> Dict[str, Any]:
        ratio_w, ratio_h = pair
        ratio = ratio_w / ratio_h
        raw_width = math.sqrt(self.target_area * ratio)
        raw_height = math.sqrt(self.target_area / ratio)
        width = max(self.min_side, int(raw_width) // self.multiple * self.multiple)
        height = max(self.min_side, int(raw_height) // self.multiple * self.multiple)
        while width * height > self.target_area and width > self.min_side:
            width -= self.multiple
        while width * height > self.target_area and height > self.min_side:
            height -= self.multiple
        return {
            "bucket_id": bucket_id,
            "ratio": f"{ratio_w}:{ratio_h}",
            "width": width,
            "height": height,
        }

    def __len__(self) -> int:
        return len(self.buckets)

    def __iter__(self) -> Iterator:
        return iter(self.buckets)

    def __getitem__(self, bucket_id: int) -> Dict[str, Any]:
        return self.buckets[bucket_id]

    def select(self, width: int, height: int) -> Dict[str, Any]:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {width}x{height}")
        image_ratio = width / height
        return min(
            self.buckets,
            key=lambda b: abs(math.log(image_ratio) - math.log(b["width"] / b["height"])),
        )


def preprocess_ref_image_pil(image, target_height: int, target_width: int) -> "torch.Tensor":
    """Resize (BICUBIC, aspect-fill) + center-crop to target and scale to [-1, 1].

    Matches vins_infer.preprocess_ref_image_pil exactly.
    """
    import numpy as np
    from PIL import Image

    if image.mode != "RGB":
        image = image.convert("RGB")

    src_w, src_h = image.size
    scale = max(target_width / src_w, target_height / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    image = image.crop((left, top, left + target_width, top + target_height))

    array = np.array(image, dtype="uint8", copy=True)
    array = torch.from_numpy(array).float() / 255.0
    array = array.permute(2, 0, 1).contiguous()
    return array * 2.0 - 1.0


def pack_flux2_latents(latents: "torch.Tensor") -> "torch.Tensor":
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(
        batch_size, num_channels, height * width).permute(0, 2, 1).contiguous()


def unpack_flux2_latents(tokens: "torch.Tensor", height: int, width: int) -> "torch.Tensor":
    batch_size, num_tokens, num_channels = tokens.shape
    if num_tokens != height * width:
        raise ValueError(
            f"Token count mismatch when unpacking latents: got {num_tokens}, "
            f"expected {height * width}"
        )
    return tokens.permute(0, 2, 1).reshape(batch_size, num_channels, height, width).contiguous()


def _prepare_latent_ids(latents: "torch.Tensor") -> "torch.Tensor":
    batch_size, _, height, width = latents.shape
    device = latents.device
    latent_ids = torch.cartesian_prod(
        torch.arange(1, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        torch.arange(1, device=device),
    )
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)


def _prepare_text_ids(prompt_embeds: "torch.Tensor") -> "torch.Tensor":
    batch_size, seq_len, _ = prompt_embeds.shape
    out_ids = []
    for _ in range(batch_size):
        coords = torch.cartesian_prod(
            torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(seq_len)
        )
        out_ids.append(coords)
    return torch.stack(out_ids)


def _build_ref_image_ids_dcae(ref_latents: "torch.Tensor", t_scale: int = 10,
                              t_index: int = 0) -> "torch.Tensor":
    batch_size, _, height, width = ref_latents.shape
    device = ref_latents.device
    t = torch.arange(1, device=device) * 0 + (t_scale + t_scale * t_index)
    coords = torch.cartesian_prod(
        t,
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        torch.arange(1, device=device),
    )
    return coords.unsqueeze(0).expand(batch_size, -1, -1)
