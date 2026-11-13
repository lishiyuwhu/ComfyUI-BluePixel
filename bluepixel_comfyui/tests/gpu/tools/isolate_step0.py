"""Isolate the source of step-0 divergence: prompt embeds, image latents,
noise latents, and a fixed-input forward."""
import sys
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

from diffusers import AutoencoderDC, Flux2Transformer2DModel
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

transformer = Flux2Transformer2DModel.from_pretrained(
    MODEL, subfolder="transformer", torch_dtype=torch.bfloat16).to(DEVICE).eval()
text_encoder = Qwen3ForCausalLM.from_pretrained(
    MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(DEVICE).eval()
tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL, subfolder="tokenizer")
vae = AutoencoderDC.from_pretrained(
    MODEL, subfolder="vae", torch_dtype=torch.float32).to(DEVICE).eval()


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


# ---- 1. prompt embeds
import vins_infer as ref
ref_pe, ref_ti = ref.embedding_prompt(PROMPT, text_encoder, tokenizer, DEVICE)

from bluepixel_comfyui.core.prompt import encode_prompt_bluepixel
our_pe, our_ti = encode_prompt_bluepixel(text_encoder, tokenizer, PROMPT, DEVICE)
print(f"prompt embeds rel={rel(our_pe, ref_pe):.3e}  shape {tuple(our_pe.shape)}")
print(f"text ids equal: {torch.equal(our_ti, ref_ti)}")

# ---- 2. image pipeline
img = Image.open("/root/code/bluepixel/assets/test_input.jpg").convert("RGB")

# ref path
w, h = img.size
scale = ((1 << 20) / (w * h)) ** 0.5
w2, h2 = int(w * scale), int(h * scale)
target_ratio = w / h
w_lo, w_hi = max(64, (w2 - 64) // 64 * 64), (w2 + 64) // 64 * 64
h_lo, h_hi = max(64, (h2 - 64) // 64 * 64), (h2 + 64) // 64 * 64
best, best_err = (w2, h2), 1e9
for cw in range(w_lo, w_hi + 1, 64):
    for ch in range(h_lo, h_hi + 1, 64):
        err = abs(cw / ch - target_ratio)
        if err < best_err:
            best_err, best = err, (cw, ch)
w2, h2 = best
img_r = img.resize((w2, h2), Image.LANCZOS)
ref_tensor = ref_pipe_img = ref.FluxKleinPipeline.__mro__  # noqa placeholder
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2ImageProcessor
proc = Flux2ImageProcessor(vae_scale_factor=128)
ref_cond = proc.preprocess(img_r, height=h2, width=w2).to(DEVICE, torch.float32)

# our path
from bluepixel_comfyui.core import latents as L
our_cond, _ = L.resize_condition_image(img, target_pixels=1 << 20)
our_cond = our_cond.to(DEVICE, torch.float32)
print(f"cond tensor: ref {tuple(ref_cond.shape)} ours {tuple(our_cond.shape)} "
      f"rel={rel(our_cond, ref_cond):.3e} max={ (our_cond-ref_cond).abs().max().item():.3e}")

# ---- 3. image latents
ref_il = ref_pipe = None
lat_ref = vae.encode(ref_cond)[0] * 0.4538
lat_our = vae.encode(our_cond)[0] * 0.4538
print(f"image latent rel={rel(lat_our, lat_ref):.3e}")

# ---- 4. noise latents (same generator semantics)
from diffusers.utils.torch_utils import randn_tensor
gen = torch.Generator(DEVICE).manual_seed(SEED)
H, W = 768, 512  # 1K target-pixels pick for this aspect
shape = (1, 128, H // 64, W // 64)
n_ref = randn_tensor(shape, generator=gen, device=DEVICE, dtype=torch.bfloat16)
gen = torch.Generator(DEVICE).manual_seed(SEED)
n_our = torch.randn(shape, generator=gen, device=DEVICE, dtype=torch.bfloat16)
print(f"noise rel={rel(n_our, n_ref):.3e} equal={torch.equal(n_our, n_ref)}")

# ---- 5. fixed-input forward both paths
ids_ref = ref._prepare_text_ids(ref_pe).to(DEVICE)
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
lat_ids = Flux2Pipeline._prepare_latent_ids(torch.zeros(shape)).to(DEVICE)
packed_ref = Flux2Pipeline._pack_latents(n_ref)
img_lat_packed = Flux2Pipeline._pack_latents(lat_ref.to(torch.bfloat16))
img_ids = Flux2Pipeline._prepare_image_ids([lat_ref])
x = torch.cat([packed_ref, img_lat_packed.squeeze(0).unsqueeze(0)], dim=1)
x_ids = torch.cat([lat_ids, img_ids.to(DEVICE)], dim=1)

timestep = torch.tensor([1.0], device=DEVICE, dtype=torch.bfloat16)
guidance = torch.full([1], 3.0, device=DEVICE)
with torch.no_grad():
    out_ref = transformer(hidden_states=x, timestep=timestep, guidance=guidance,
                          encoder_hidden_states=ref_pe, txt_ids=ids_ref,
                          img_ids=x_ids, return_dict=False)[0]
    out_ref2 = transformer(hidden_states=x, timestep=timestep, guidance=guidance,
                           encoder_hidden_states=ref_pe, txt_ids=ids_ref,
                           img_ids=x_ids, return_dict=False)[0]
print(f"transformer self-consistency rel={rel(out_ref2, out_ref):.3e}")

# our core with identical inputs
import bluepixel_comfyui.core.pipeline as P
inputs = dict(hidden_states=x, timestep=timestep, guidance=guidance,
              encoder_hidden_states=ref_pe, txt_ids=ids_ref, img_ids=x_ids)
with torch.no_grad():
    out_a = transformer(**inputs, return_dict=False)[0]
    out_b = transformer(**inputs, return_dict=False)[0]
print(f"same-input rerun rel={rel(out_b, out_a):.3e}")
