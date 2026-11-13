"""Capture post-update latents at each step in ref pipe vs our core."""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, "/root/code/bluepixel")

from PIL import Image

MODEL = "/data/vjuicefs_ai_camera_lgroup_ql/public_data/72225263/hf_upload/BluePixel-Turbo"
DEVICE = torch.device("cuda")
SEED = 42
PROMPT = ("Change the gray sweater to burgundy red. "
          "Preserve the person, pose, accessories, and background.")
IMAGE = "/root/code/bluepixel/assets/test_input.jpg"

import vins_infer as ref

with redirect_stdout(io.StringIO()):
    ref_pipe = ref.load_pipeline(MODEL, 0, False, None, 0.0, "empirical")

ref_lat, ref_pred = [], []
orig_step = ref_pipe.scheduler.step


def step_cap(model_output, timestep, sample, **kw):
    ref_pred.append(model_output.detach().clone())
    out = orig_step(model_output, timestep, sample, **kw)
    ref_lat.append(out[0].detach().clone())
    return out


ref_pipe.scheduler.step = step_cap

img = Image.open(IMAGE).convert("RGB")
gen = torch.Generator(DEVICE).manual_seed(SEED)
with redirect_stdout(io.StringIO()):
    ref_pipe(image=img, instruction=PROMPT, num_inference_steps=4, mu_mode="empirical",
             guidance_scale=3.0, generator=gen, output_type="pil", target_pixels=1 << 20)
print(f"ref: {len(ref_pred)} preds, {len(ref_lat)} latents")
del ref_pipe
torch.cuda.empty_cache()

from diffusers import AutoencoderDC, Flux2Transformer2DModel
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

transformer = Flux2Transformer2DModel.from_pretrained(
    MODEL, subfolder="transformer", torch_dtype=torch.bfloat16).to(DEVICE).eval()
text_encoder = Qwen3ForCausalLM.from_pretrained(
    MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(DEVICE).eval()
tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL, subfolder="tokenizer")
vae = AutoencoderDC.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).to(DEVICE).eval()

from bluepixel_comfyui.core.image_processing import pil_to_comfy_image
from bluepixel_comfyui.core import scheduler as sched_mod
from bluepixel_comfyui.core.pipeline import run_bluepixel

our_lat, our_pred = [], []
orig_euler = sched_mod.euler_update


def euler_cap(sample, model_output, sigma, sigma_next):
    our_pred.append(model_output.detach().clone())
    out = orig_euler(sample, model_output, sigma, sigma_next)
    our_lat.append(out.detach().clone())
    return out


sched_mod.euler_update = euler_cap

our_img = pil_to_comfy_image(img)
run_bluepixel(transformer=transformer, vae=vae, text_encoder=text_encoder,
              tokenizer=tokenizer, scaling_factor=0.4538, image=our_img,
              prompt=PROMPT, seed=SEED, steps=4, guidance=3.0,
              target_pixels=1 << 20, mu_mode="empirical", shift=0.0,
              vae_tiling=False, device=DEVICE)
print(f"ours: {len(our_pred)} preds, {len(our_lat)} latents")

for i in range(4):
    pd = (our_pred[i].float() - ref_pred[i].float()).abs().max().item()
    ld = (our_lat[i].float() - ref_lat[i].float()).abs().max().item()
    pe = ((our_pred[i].float() - ref_pred[i].float()).norm() / (ref_pred[i].float().norm() + 1e-9)).item()
    le = ((our_lat[i].float() - ref_lat[i].float()).norm() / (ref_lat[i].float().norm() + 1e-9)).item()
    print(f"step {i}: pred rel={pe:.2e} max={pd:.3e} | latent rel={le:.2e} max={ld:.3e}")
