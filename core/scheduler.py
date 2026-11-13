"""Sigma schedule + CPS stepping (BluePixel distilled-model semantics).

Mirrors vins_infer.py (fb1366a):
  - explicit denoising_step_list [1000, 750, 500, 250] -> raw sigmas
  - mu: training (linear, seq-len capped at 4096) | empirical (diffusers table)
  - exponential/linear time shift on numpy-style fp32 tensor math
  - optional shift_terminal stretch
  - CPS step: x0/eps decomposition, eta-controlled stochasticity, fp32 update
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch

DEFAULT_DENOISING_STEPS = [1000, 750, 500, 250]


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """diffusers Flux2 empirical mu table."""
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


def compute_training_mu(image_seq_len: int, seq_len_cap: float = 4096.0) -> float:
    """Training mu: linear fit with seq-len cap (NEW vs old plugin: cap 4096)."""
    effective_seq_len = min(float(image_seq_len), seq_len_cap)
    return float(0.00016927 * effective_seq_len + 0.45666666)


def _time_shift_exponential(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _time_shift_linear(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return mu / (mu + (1 / t - 1) ** sigma)


def compute_sigmas(
    denoising_step_list: Sequence[int],
    scheduler_config,
    image_seq_len: int,
    num_steps: Optional[int] = None,
    mu_mode: str = "training",
) -> List[float]:
    """Sigma schedule from explicit denoising timesteps + mu shifting.

    scheduler_config: object with attributes num_train_timesteps,
    use_dynamic_shifting, time_shift_type, shift, shift_terminal
    (i.e. diffusers scheduler .config).
    """
    steps = list(denoising_step_list[:num_steps] if num_steps is not None else denoising_step_list)
    if not steps:
        raise ValueError("denoising_step_list must contain at least one step.")

    num_train_timesteps = int(getattr(scheduler_config, "num_train_timesteps", 1000))
    raw_sigmas = torch.tensor(
        [float(s) / float(num_train_timesteps) for s in steps], dtype=torch.float32
    )

    use_dynamic = bool(getattr(scheduler_config, "use_dynamic_shifting", False))
    if mu_mode == "training":
        mu = compute_training_mu(image_seq_len=image_seq_len)
    elif mu_mode == "empirical":
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=len(steps))
    elif use_dynamic:
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=len(steps))
    else:
        mu = None

    if mu is None:
        shift = float(getattr(scheduler_config, "shift", 1.0))
        shifted = shift * raw_sigmas / (1 + (shift - 1) * raw_sigmas)
    else:
        time_shift_type = str(getattr(scheduler_config, "time_shift_type", "exponential")).lower()
        if time_shift_type == "linear":
            shifted = _time_shift_linear(mu, 1.0, raw_sigmas)
        else:
            shifted = _time_shift_exponential(mu, 1.0, raw_sigmas)

    shift_terminal = getattr(scheduler_config, "shift_terminal", None)
    if shift_terminal:
        one_minus_z = 1 - shifted
        scale = one_minus_z[-1] / (1 - float(shift_terminal))
        shifted = 1 - (one_minus_z / scale)

    return [float(s.item()) for s in shifted]


class CPSScheduler:
    """Flow-matching CPS step with controllable stochasticity via eta.

    State (timesteps/sigmas/step_index) mirrors diffusers'
    FlowMatchEulerDiscreteScheduler so existing code paths keep working.
    """

    def __init__(self, scheduler_config):
        self.config = scheduler_config
        self.num_train_timesteps = int(getattr(scheduler_config, "num_train_timesteps", 1000))
        self.timesteps: torch.Tensor = torch.zeros(0)
        self.sigmas: torch.Tensor = torch.zeros(0)
        self.num_inference_steps = 0
        self._step_index: Optional[int] = None
        self._eta: float = 0.0

    def set_eta(self, eta: float) -> None:
        if eta < 0 or eta > 1:
            raise ValueError(f"eta must be in [0, 1], got {eta}")
        self._eta = eta

    @property
    def eta(self) -> float:
        return self._eta

    def set_timesteps(self, sigmas: Sequence[float], device=None) -> None:
        sig = torch.tensor(list(sigmas), dtype=torch.float32)
        if len(sig) == 0 or sig[-1] != 0.0:
            sig = torch.cat([sig, torch.zeros(1)])
        timesteps = sig[:-1] * self.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas = sig.to(device=device)
        self.num_inference_steps = len(self.timesteps)
        self._step_index = 0

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: Optional[torch.Generator] = None, return_dict: bool = True):
        """One CPS update. model_output/sample: (B, C, H, W) latent-shaped."""
        if self._step_index is None:
            self._step_index = 0
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        sample_f32 = sample.to(torch.float32)
        v_pred_f32 = model_output.to(torch.float32)

        x0 = sample_f32 - sigma * v_pred_f32
        eps = sample_f32 + (1.0 - sigma) * v_pred_f32

        if sigma_next == 0.0 or self._eta == 0.0:
            prev_sample = sample_f32 + (sigma_next - sigma) * v_pred_f32
        else:
            rho = sigma_next * math.sin(self._eta * math.pi / 2)
            eps_coeff = sigma_next * math.cos(self._eta * math.pi / 2)
            prev_mean = (1.0 - sigma_next) * x0 + eps_coeff * eps
            noise = torch.randn(
                sample.shape, generator=generator, device=sample.device, dtype=torch.float32
            )
            prev_sample = prev_mean + rho * noise

        self._step_index += 1
        prev_sample = prev_sample.to(model_output.dtype)
        if return_dict:
            return type("S", (), {"prev_sample": prev_sample})
        return (prev_sample,)


def validate_mu_combo(mu_mode: str) -> None:
    if mu_mode not in ("empirical", "training"):
        raise ValueError(f"Unknown mu_mode: {mu_mode}")
