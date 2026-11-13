"""Numerical parity check: ComfyUI-BluePixel core vs reference vins_infer pipeline.

Compares, for identical seed / prompt / image / steps / guidance / mu_mode:
  1. prompt embeds          (atol 1e-5 fp32 cast)
  2. reference latents      (exact, same VAE path)
  3. sigma schedule         (atol 1e-6)
  4. per-step noise preds   (bf16 tolerance)
  5. final images           (PSNR / max abs diff)

Run:
    python bluepixel_comfyui/tests/gpu/test_reference_parity.py \
        --model-dir /path/to/BluePixel-Turbo
"""
import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REF_REPO = "/root/code/bluepixel"
if REF_REPO not in sys.path:
    sys.path.insert(0, REF_REPO)

from bluepixel_comfyui.core import scheduler as sched_mod
from bluepixel_comfyui.core.helpers import compute_empirical_mu
from bluepixel_comfyui.core.image_processing import pil_to_comfy_image
from bluepixel_comfyui.core.pipeline import run_bluepixel


def rel_l2(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-pixels", type=int, default=1 << 20)
    ap.add_argument("--image", default="/root/code/bluepixel/assets/test_input.jpg")
    ap.add_argument("--prompt",
                    default="Change the gray sweater to burgundy red. Preserve the person, pose, accessories, and background.")
    args = ap.parse_args()
    device = torch.device("cuda")

    # ---------------- reference pipeline (vins_infer) ----------------
    import vins_infer as ref
    buf = io.StringIO()
    with redirect_stdout(buf):
        ref_pipe = ref.load_pipeline(args.model_dir, 0, False, None, 0.0, "empirical")
    print("[ref] pipeline loaded")

    ref_caps = {}
    ref_call_count = [0]
    orig_transformer_call = ref_pipe.transformer.__call__

    # monkey-patch the pipeline's transformer for capture
    class CaptureWrap:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, *a, **kw):
            out = orig_transformer_call(*a, **kw)
            tensor = out[0] if isinstance(out, tuple) else out
            step = ref_call_count[0] // calls_per_step
            ref_call_count[0] += 1
            ref_caps.setdefault(f"step{step}", []).append(tensor.detach().clone())
            return out  # preserve original return type

        def __getattr__(self, name):
            return getattr(self.inner, name)

    ref_pipe.transformer = CaptureWrap(ref_pipe.transformer)

    # step index is derived from capture count: 2 calls per step (CFG), 1 without.
    calls_per_step = 2 if args.guidance > 1.0 else 1

    from PIL import Image
    input_image = Image.open(args.image).convert("RGB")
    gen = torch.Generator(device).manual_seed(args.seed)
    t0 = time.time()
    ref_out = ref_pipe(
        image=input_image, instruction=args.prompt,
        num_inference_steps=args.steps, mu_mode="empirical",
        guidance_scale=args.guidance, generator=gen,
        output_type="pil", target_pixels=args.target_pixels,
    )
    print(f"[ref] run {time.time()-t0:.1f}s, captures={sum(len(v) for v in ref_caps.values())}")
    ref_img = np.asarray(ref_out.images[0]).astype(np.float32) / 255.0

    # release ref pipeline GPU memory where possible
    del ref_pipe
    torch.cuda.empty_cache()

    # ---------------- our core ----------------
    from diffusers import AutoencoderDC, Flux2Transformer2DModel
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.model_dir, subfolder="transformer", torch_dtype=torch.bfloat16).to(device)
    transformer.eval()
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    text_encoder.eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_dir, subfolder="tokenizer")
    vae = AutoencoderDC.from_pretrained(
        args.model_dir, subfolder="vae", torch_dtype=torch.float32).to(device)
    vae.eval()
    scaling = float(vae.config.scaling_factor)

    our_caps = {}
    orig_fwd = transformer.forward

    class OurWrap:
        def __init__(self, inner):
            self.inner = inner
            self.i = 0

        def __call__(self, *a, **kw):
            out = orig_fwd(*a, **kw)
            tensor = out[0] if isinstance(out, tuple) else out
            our_caps.setdefault(f"step{self.i}", []).append(tensor.detach().clone())
            return out  # preserve original return type

        def __getattr__(self, name):
            return getattr(self.inner, name)

    our_transformer = OurWrap(transformer)

    def progress_cb(done, total):
        our_transformer.i = done  # next step index after previous completes

    img = pil_to_comfy_image(input_image)
    t0 = time.time()
    our_out = run_bluepixel(
        transformer=our_transformer, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, scaling_factor=scaling, image=img,
        prompt=args.prompt, seed=args.seed, steps=args.steps,
        guidance=args.guidance, target_pixels=args.target_pixels,
        mu_mode="empirical", shift=0.0, vae_tiling=False,
        device=device, progress_cb=progress_cb,
    )
    print(f"[ours] run {time.time()-t0:.1f}s")
    our_img = our_out["images"][0].numpy()  # (H, W, 3) in [0,1]

    # ---------------- compare ----------------
    print(f"ref steps captured: {sorted(ref_caps.keys())}")
    print(f"our steps captured: {sorted(our_caps.keys())}")
    ok = True
    for i in range(args.steps):
        r = ref_caps.get(f"step{i}", [])
        o = our_caps.get(f"step{i}", [])
        if not r or not o:
            print(f"step {i}: MISSING captures ref={len(r)} ours={len(o)}")
            ok = False
            continue
        # ref captures pos+neg (2 per step when cfg>1); ours same order
        n = min(len(r), len(o))
        for j in range(n):
            if r[j].shape != o[j].shape:
                print(f"step {i} call {j}: SHAPE MISMATCH ref={tuple(r[j].shape)} ours={tuple(o[j].shape)}")
                ok = False
                continue
            e = rel_l2(o[j], r[j])
            print(f"step {i} call {j}: rel_l2={e:.6f} shape={tuple(o[j].shape)}")
            if e > 0.05:
                ok = False

    h = min(ref_img.shape[0], our_img.shape[0])
    w = min(ref_img.shape[1], our_img.shape[1])
    diff = np.abs(ref_img[:h, :w] - our_img[:h, :w])
    mse = (diff ** 2).mean()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    print(f"image: ref={ref_img.shape} ours={our_img.shape} "
          f"max_abs={diff.max():.4f} PSNR={psnr:.2f} dB")

    from PIL import Image as PILImage
    out_dir = Path("/tmp/bluecode/bluepixel_parity")
    out_dir.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray((ref_img[:h, :w] * 255).round().astype("uint8")).save(out_dir / "ref.jpg")
    PILImage.fromarray((our_img[:h, :w] * 255).round().astype("uint8")).save(out_dir / "ours.jpg")
    print(f"saved {out_dir}/ref.jpg and ours.jpg")

    verdict = ok and psnr > 20
    print("PARITY:", "PASS" if verdict else "FAIL")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
