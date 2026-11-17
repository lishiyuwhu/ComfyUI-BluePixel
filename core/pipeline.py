"""BluePixel pipeline core (vins_infer fb1366a semantics).

Key differences vs the previous plugin generation:
  - guidance is ALWAYS disabled (guidance=None, single conditional forward);
    the released distilled checkpoints are evaluated with guidance_scale=1.0
  - resolution comes from the bucket system (16 ratios, 4K area, 64 grid);
    the reference image is BICUBIC resize+center-crop to the bucket size
  - explicit denoising_step_list [1000, 750, 500, 250] with CPS stepping
    (eta=0.9 default), x0 accumulated in fp32
  - transformer auto fp32/int8 (Int8Linear) by quantization_config
  - mu_mode: training (default) with seq-len cap 4096, or empirical
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Sequence

import torch

from . import buckets as bucket_mod
from .scheduler import CPSScheduler, DEFAULT_DENOISING_STEPS, compute_sigmas, validate_mu_combo
from .image_processing import tensor_to_comfy_image

logger = logging.getLogger("bluepixel")

INTERRUPT_CHECK = Callable[[], None]


class _DictConfig:
    """Attribute access over a plain config dict."""

    def __init__(self, d: dict):
        self._d = dict(d)

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _default_interrupt() -> None:
    return None


@torch.no_grad()
def encode_prompts_bluepixel(text_encoder, tokenizer, prompts: Sequence[str], device,
                             max_sequence_length: int = 512,
                             hidden_states_layers=(9, 18, 27)):
    """Qwen3 chat-template prompt embeds + text ids (unchanged from reference)."""
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )

    tokenized = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True,
        max_length=max_sequence_length,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    outputs = text_encoder(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    stacked = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    stacked = stacked.to(dtype=text_encoder.dtype, device=device)

    batch_size, num_layers, seq_len, hidden_dim = stacked.shape
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_layers * hidden_dim)
    text_ids = bucket_mod._prepare_text_ids(prompt_embeds).to(device)
    return prompt_embeds, text_ids


class BluePixelEngine:
    """One loaded BluePixel model set (transformer/vae/text_encoder/tokenizer).

    Stateless per generation: generate() builds its own scheduler, RNG and
    intermediate tensors; nothing mutates module-level state.
    """

    def __init__(self, transformer, vae, text_encoder, tokenizer, scheduler,
                 weight_dtype: torch.dtype, device, is_int8: bool,
                 bucket_area: int = bucket_mod.DEFAULT_BUCKET_AREA,
                 bucket_multiple: int = bucket_mod.DEFAULT_BUCKET_MULTIPLE):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.weight_dtype = weight_dtype
        self.device = device
        self.is_int8 = is_int8
        self.bucket_set = bucket_mod.ResolutionBucketSet(
            target_area=bucket_area, multiple=bucket_multiple)
        self.num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        self.last_mu_mode: Optional[str] = None
        self.last_image_seq_len: Optional[int] = None

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.transformer.parameters()).dtype
        except StopIteration:
            return torch.float32

    @torch.no_grad()
    def _encode_ref_image(self, pil_image, bucket: Dict[str, Any]):
        ref_tensor = bucket_mod.preprocess_ref_image_pil(
            pil_image, target_height=bucket["height"], target_width=bucket["width"]
        )
        ref_pixels = ref_tensor.unsqueeze(0).to(device=self.vae.device, dtype=self.vae.dtype)

        latents = self.vae.encode(ref_pixels)[0]
        latents = latents * self.vae.config.scaling_factor
        ref_ids = bucket_mod._build_ref_image_ids_dcae(latents, t_scale=10, t_index=0)
        ref_tokens = bucket_mod.pack_flux2_latents(latents)
        return ref_tokens, ref_ids

    @torch.no_grad()
    def _predict_noise(self, noisy_latents, prompt_embeds, timesteps, text_ids,
                       ref_tokens, ref_ids):
        batch_size, _, height, width = noisy_latents.shape
        latent_ids = bucket_mod._prepare_latent_ids(noisy_latents)
        model_dtype = self._model_dtype()
        packed = bucket_mod.pack_flux2_latents(noisy_latents).to(model_dtype)

        model_timestep = timesteps.to(device=noisy_latents.device, dtype=packed.dtype)
        model_timestep = model_timestep / float(self.num_train_timesteps)

        full_hidden = torch.cat(
            [packed, ref_tokens.to(device=noisy_latents.device, dtype=model_dtype)], dim=1
        )
        full_ids = torch.cat([latent_ids, ref_ids.to(device=noisy_latents.device)], dim=1)

        pred = self.transformer(
            hidden_states=full_hidden,
            timestep=model_timestep,
            guidance=None,
            encoder_hidden_states=prompt_embeds.to(
                device=noisy_latents.device, dtype=model_dtype),
            txt_ids=text_ids.to(device=noisy_latents.device),
            img_ids=full_ids,
            return_dict=False,
        )[0]
        pred_tokens = pred[:, :packed.size(1), :]
        return bucket_mod.unpack_flux2_latents(pred_tokens, height=height, width=width).to(
            noisy_latents.dtype
        )

    @torch.no_grad()
    def _decode(self, latents) -> torch.Tensor:
        """Decode latents to ComfyUI IMAGE (B,H,W,C) float32 [0,1], rounded to uint8 grid."""
        latents = (latents / self.vae.config.scaling_factor).to(
            device=self.vae.device, dtype=self.vae.dtype
        )
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # match reference: round to the uint8 grid before returning
        array = (
            (image * 255)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .permute(0, 2, 3, 1)
            .numpy()
        )
        return torch.from_numpy(array).float() / 255.0

    @torch.no_grad()
    def generate(
        self,
        image,
        instruction: str,
        seed: int = 42,
        num_steps: int = 4,
        denoising_step_list: Optional[Sequence[int]] = None,
        cps_eta: float = 0.9,
        mu_mode: str = "training",
        bucket_set: Optional[bucket_mod.ResolutionBucketSet] = None,
        sigma_image_seq_len: Optional[int] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        interrupt_check: INTERRUPT_CHECK = _default_interrupt,
        stage_cb: Optional[Callable[[str], None]] = None,
    ):
        """Generate one edited image for a single (image, instruction) pair.

        image: ComfyUI IMAGE tensor (B,H,W,C) [0,1] — batch item 0 is used.
        stage_cb: optional callback invoked with a stage name right before that
        stage starts ("text_encode" | "image_encode" | "denoise" | "decode"),
        so a host can load/unload only the components that stage needs.
        Returns (comfy_image, metadata_dict).
        """
        def _stage(name):
            if stage_cb is not None:
                stage_cb(name)

        validate_mu_combo(mu_mode)
        denoising_step_list = denoising_step_list or DEFAULT_DENOISING_STEPS
        bucket_set = bucket_set or self.bucket_set
        interrupt_check()

        from .image_processing import comfy_image_to_tensor, tensor_to_pil
        if image is None:
            raise ValueError("BluePixel requires an input image (edit-only model).")
        first = comfy_image_to_tensor(image[:1])
        pil = tensor_to_pil(first)

        _stage("text_encode")
        sched_cfg = self.scheduler.config
        if isinstance(sched_cfg, dict):
            sched_cfg = _DictConfig(sched_cfg)
        self.x0_scheduler = CPSScheduler(sched_cfg)
        self.x0_scheduler.set_eta(cps_eta)

        prompt_embeds, text_ids = encode_prompts_bluepixel(
            prompts=[instruction],
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        bucket = bucket_set.select(*pil.size)
        _stage("image_encode")
        ref_tokens, ref_ids = self._encode_ref_image(pil, bucket)
        interrupt_check()

        effective_seq_len = (
            sigma_image_seq_len if sigma_image_seq_len is not None else int(ref_ids.shape[1])
        )
        sigmas = compute_sigmas(
            denoising_step_list=denoising_step_list,
            scheduler_config=sched_cfg,
            image_seq_len=effective_seq_len,
            num_steps=min(num_steps, len(denoising_step_list)),
            mu_mode=mu_mode,
        )
        actual_steps = min(num_steps, len(sigmas))
        self.last_mu_mode = mu_mode
        self.last_image_seq_len = effective_seq_len

        _stage("denoise")
        self.x0_scheduler.set_timesteps(sigmas=sigmas, device=self.device)
        self.x0_scheduler._step_index = 0

        # Reference behavior (vins_infer): global RNG is seeded per sample so the
        # CPS eta noise (drawn from the global RNG, generator=None) is reproducible.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent_h = int(ref_ids[..., 1].max().item()) + 1
        latent_w = int(ref_ids[..., 2].max().item()) + 1
        latent_channels = self.transformer.config.in_channels
        x_t = torch.randn(
            1, latent_channels, latent_h, latent_w,
            device=self.device, dtype=self.weight_dtype, generator=generator,
        )
        x0 = x_t

        for idx in range(actual_steps):
            interrupt_check()
            t = self.x0_scheduler.timesteps[idx]
            t_batch = t.expand(1).to(device=self.device, dtype=torch.float32)

            v_pred = self._predict_noise(
                x_t, prompt_embeds, t_batch, text_ids, ref_tokens, ref_ids
            )

            step_output = self.x0_scheduler.step(v_pred, t, x_t, return_dict=True)
            sigma = self.x0_scheduler.sigmas[idx]
            x0 = (x_t.float() - sigma * v_pred.float()).to(self.weight_dtype)
            x_t = step_output.prev_sample
            if progress_cb is not None:
                progress_cb(idx + 1, actual_steps)

        _stage("decode")
        output = self._decode(x0)
        metadata = {
            "width": bucket["width"],
            "height": bucket["height"],
            "bucket_ratio": bucket["ratio"],
            "bucket_id": bucket["bucket_id"],
            "steps": actual_steps,
            "denoising_steps": list(denoising_step_list[:actual_steps]),
            "cps_eta": cps_eta,
            "seed": seed,
            "mu_mode": mu_mode,
            "image_seq_len": effective_seq_len,
            "bucket_area": bucket_set.target_area,
            "cfg_enabled": False,
            "is_int8": self.is_int8,
            "reference_image": True,
        }
        return output, metadata


def run_bluepixel(**kwargs):
    """Back-compat wrapper: engine.generate().

    Accepts engine + generate kwargs; returns dict {images, metadata}.
    """
    engine = kwargs.pop("engine")
    progress_cb = kwargs.pop("progress_cb", None)
    interrupt_check = kwargs.pop("interrupt_check", None) or _default_interrupt
    image_out, metadata = engine.generate(
        progress_cb=progress_cb, interrupt_check=interrupt_check, **kwargs)
    return {"images": image_out, "metadata": metadata}
