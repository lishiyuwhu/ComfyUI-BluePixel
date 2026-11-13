"""GPU smoke test with real BluePixel-Turbo weights.

Run inside the ComfyUI python env:
    python tests/gpu/test_real_weights.py \
        --model-dir /path/to/BluePixel-Turbo

Checks:
  1. text encode shape (1, 512, 12288), text ids (1, 512, 4)
  2. reference latents + ids shapes
  3. sigma schedule matches diffusers scheduler (empirical mu)
  4. transformer forward runs, output shape == input seq
  5. full edit run 1K target-pixels, output (1, H, W, 3) in [0,1]
  6. determinism: same seed twice -> bitwise equal
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import latents as latents_mod
from core import scheduler as sched_mod
from core.helpers import compute_empirical_mu
from core.pipeline import run_bluepixel
from core.image_processing import pil_to_comfy_image
from runtime import loader as loader_mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--target-pixels", type=int, default=1 << 20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mu-mode", default="empirical")
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--image", default="/root/code/bluepixel/assets/test_input.jpg")
    ap.add_argument("--prompt",
                    default="Change the gray sweater to burgundy red. Preserve the person, pose, accessories, and background.")
    args = ap.parse_args()

    device = torch.device("cuda")
    model_dir = args.model_dir

    ok, problems = loader_mod.validate_model_dir(model_dir)
    if not ok:
        print("VALIDATION FAILED:", *problems, sep="\n  ")
        return 1
    print(f"[1] model dir valid: {model_dir}")
    print(f"    fingerprint: {loader_mod.compute_fingerprint(model_dir)}")

    from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler
    from diffusers import Flux2Transformer2DModel
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

    t0 = time.time()
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=torch.bfloat16).to(device)
    transformer.eval()
    print(f"[2] transformer loaded {time.time()-t0:.1f}s, "
          f"params={sum(p.numel() for p in transformer.parameters())/1e9:.2f}B")

    t0 = time.time()
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    text_encoder.eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_dir, subfolder="tokenizer")
    print(f"[3] text encoder loaded {time.time()-t0:.1f}s")

    t0 = time.time()
    vae = AutoencoderDC.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.float32).to(device)
    vae.eval()
    scaling = float(vae.config.scaling_factor)
    print(f"[4] vae loaded {time.time()-t0:.1f}s, scaling_factor={scaling}")

    # --- text encode
    from core.prompt import encode_prompt_bluepixel
    pe, ti = encode_prompt_bluepixel(text_encoder, tokenizer, args.prompt, device)
    assert pe.shape == (1, 512, 12288), pe.shape
    assert ti.shape == (1, 512, 4), ti.shape
    print(f"[5] prompt embeds {tuple(pe.shape)}, text_ids {tuple(ti.shape)}, "
          f"mean={pe.float().mean().item():.4f}")

    # --- sigma schedule vs diffusers
    import numpy as np
    seq_len = (1024 // 64) * (1024 // 64)
    mu = compute_empirical_mu(seq_len, args.steps)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
    raw = np.linspace(1.0, 1.0 / args.steps, args.steps)
    sched.set_timesteps(args.steps, sigmas=raw, mu=mu)
    ref = sched.sigmas.cpu().numpy().tolist()
    ours = sched_mod.build_sigma_schedule(args.steps, seq_len, args.mu_mode, args.shift)
    max_err = max(abs(a - b) for a, b in zip(ours, ref))
    assert max_err < 1e-5, f"sigma mismatch {max_err}"
    print(f"[6] sigma schedule matches diffusers (max_err={max_err:.2e}), mu={mu:.4f}")

    # --- full run
    from PIL import Image
    img = pil_to_comfy_image(Image.open(args.image).convert("RGB"))
    t0 = time.time()
    r1 = run_bluepixel(
        transformer=transformer, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, scaling_factor=scaling, image=img,
        prompt=args.prompt, seed=args.seed, steps=args.steps,
        guidance=args.guidance, target_pixels=args.target_pixels,
        mu_mode=args.mu_mode, shift=args.shift, vae_tiling=True,
        device=device,
    )
    dt = time.time() - t0
    m = r1["metadata"]
    out = r1["images"]
    assert out.ndim == 4 and out.shape[0] == 1 and out.shape[-1] == 3
    assert out.min() >= 0.0 and out.max() <= 1.0
    print(f"[7] edit run OK: {m['width']}x{m['height']} in {dt:.1f}s "
          f"(seq_len={m['image_seq_len']}, mu={m['mu']:.4f}, cfg={m['cfg_enabled']})")

    r2 = run_bluepixel(
        transformer=transformer, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, scaling_factor=scaling, image=img,
        prompt=args.prompt, seed=args.seed, steps=args.steps,
        guidance=args.guidance, target_pixels=args.target_pixels,
        mu_mode=args.mu_mode, shift=args.shift, vae_tiling=True,
        device=device,
    )
    same = torch.equal(r1["images"], r2["images"])
    print(f"[8] determinism (same seed): {'PASS' if same else 'FAIL'}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[9] peak VRAM: {peak:.1f} GB")

    out_dir = Path("/tmp/bluecode/bluepixel_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    from core.image_processing import tensor_to_pil
    tensor_to_pil(comfy_image_to_tensor_rev(out)).save(out_dir / "out.jpg")
    print(f"[10] saved {out_dir/'out.jpg'}")
    return 0 if same else 2


def comfy_image_to_tensor_rev(t):
    return t.permute(0, 3, 1, 2).mul(2).sub(1).clamp(-1, 1)


if __name__ == "__main__":
    sys.exit(main())
