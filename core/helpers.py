"""Pure helper functions shared by the BluePixel runtime.

This module must stay import-light: no torch, no diffusers, no transformers.
It only contains CPU math on python/numpy objects so it can be unit-tested
without a GPU.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def round_size_to_multiple(size: int, multiple: int) -> int:
    """Round size away from zero to a multiple of `multiple` (min 1*multiple)."""
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    return max(multiple, ((size + multiple - 1) // multiple) * multiple)


def pick_reference_size(width: int, height: int) -> Tuple[int, int]:
    """BluePixel reference-image size rule (vins_infer.py semantics).

    Given a resized image (w, h), search the 64-grid around it and pick the
    (w, h) pair minimizing aspect-ratio error, preferring the smaller w on
    ties (range(w_lo, w_hi+1) iterates ascending w then h, first best wins).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid size: {width}x{height}")
    target_ratio = width / height
    w_lo, w_hi = max(64, (width - 64) // 64 * 64), (width + 64) // 64 * 64
    h_lo, h_hi = max(64, (height - 64) // 64 * 64), (height + 64) // 64 * 64
    best: Tuple[int, int] = (width, height)
    best_err = 1e9
    for cw in range(w_lo, w_hi + 1, 64):
        for ch in range(h_lo, h_hi + 1, 64):
            err = abs(cw / ch - target_ratio)
            if err < best_err:
                best_err = err
                best = (cw, ch)
    return best


def scale_to_target_pixels(width: int, height: int, target_pixels: int) -> Tuple[int, int]:
    """Uniform scale so w*h ~= target_pixels (vins_infer.py semantics)."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid size: {width}x{height}")
    if target_pixels <= 0:
        raise ValueError(f"target_pixels must be positive, got {target_pixels}")
    scale = math.sqrt(target_pixels / (width * height))
    return max(1, int(width * scale)), max(1, int(height * scale))


def compute_training_mu(image_seq_len: int) -> float:
    """Exponential mapping mu from seq len (training mu, vins_infer.py)."""
    if image_seq_len <= 0:
        raise ValueError(f"image_seq_len must be positive, got {image_seq_len}")
    x = float(image_seq_len)
    return float(math.log(1.0 + x) / (math.log(2.0) + x))


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Empirical mu table from diffusers Flux2 (kept local to pin semantics)."""
    if image_seq_len <= 0:
        raise ValueError(f"image_seq_len must be positive, got {image_seq_len}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def validate_mu_combo(mu_mode: str, shift: float) -> None:
    """Mirror configure_inference_scheduler validation from vins_infer.py."""
    if mu_mode not in ("empirical", "training", "fixed"):
        raise ValueError(f"Unknown mu_mode: {mu_mode}")
    if shift < 0 or not math.isfinite(shift):
        raise ValueError("shift must be finite and non-negative")
    if mu_mode == "training" and shift != 0:
        raise ValueError("mu_mode=training requires shift=0")


def exponential_time_shift(mu: float, t: float) -> float:
    """diffusers/ComfyUI exponential sigma shift: mu over (1/t - 1)."""
    if t <= 0 or t > 1.0:
        raise ValueError(f"t must be in (0, 1], got {t}")
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))


def build_sigma_schedule(
    num_steps: int,
    image_seq_len: int,
    mu_mode: str = "empirical",
    shift: float = 0.0,
) -> List[float]:
    """Full BluePixel sigma schedule (descending, includes terminal 0.0).

    Mirrors FluxKleinPipeline.__call__: sigmas = linspace(1, 1/steps, steps)
    then shifted by exponential dynamic shifting with `mu` (empirical or
    training), or by the fixed-shift logistic transform when mu_mode=fixed.
    Returns list of floats, length num_steps + 1, last entry 0.0.
    """
    validate_mu_combo(mu_mode, shift)
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    raw = [1.0 - i * (1.0 - 1.0 / num_steps) / max(num_steps - 1, 1) for i in range(num_steps)]
    # exact linspace(1, 1/steps, steps)
    raw = list(np_linspace(1.0, 1.0 / num_steps, num_steps))

    if mu_mode == "fixed":
        out = [shift * s / (1.0 + (shift - 1.0) * s) for s in raw]
    else:
        if mu_mode == "empirical":
            mu = compute_empirical_mu(image_seq_len, num_steps)
        else:
            mu = compute_training_mu(image_seq_len)
        out = [exponential_time_shift(mu, s) for s in raw]
    out.append(0.0)
    return out


def np_linspace(start: float, stop: float, num: int) -> List[float]:
    """num evenly spaced samples from start to stop inclusive (numpy semantics)."""
    if num < 0:
        raise ValueError("num must be non-negative")
    if num == 0:
        return []
    if num == 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def euler_step(sample: Sequence[Sequence[float]], sigma: float, sigma_next: float,
               model_output: Sequence[Sequence[float]]) -> List[List[float]]:
    """Flow-match Euler update on nested lists (for CPU tests, float32 math).

    x_{t+1} = x_t + (sigma_next - sigma) * v_pred
    """
    dt = sigma_next - sigma
    return [[float(x) + dt * float(v) for x, v in zip(row_x, row_v)]
            for row_x, row_v in zip(sample, model_output)]


def derive_seed(base_seed: int, batch_index: int) -> int:
    """Deterministic per-batch-item seed: base_seed + batch_index."""
    if base_seed < 0:
        raise ValueError(f"seed must be >= 0, got {base_seed}")
    return base_seed + batch_index
