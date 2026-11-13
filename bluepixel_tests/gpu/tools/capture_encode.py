"""Capture the exact image tensor fed to vae.encode in ref pipe vs our core."""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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

captured = {}
orig_encode = ref_pipe.vae.encode


def encode_capture(x, *a, **kw):
    captured.setdefault("ref", []).append(x.detach().clone())
    return orig_encode(x, *a, **kw)


ref_pipe.vae.encode = encode_capture

img = Image.open(IMAGE).convert("RGB")
gen = torch.Generator(DEVICE).manual_seed(SEED)
with redirect_stdout(io.StringIO()):
    ref_pipe(image=img, instruction=PROMPT, num_inference_steps=4,
             mu_mode="empirical", guidance_scale=3.0, generator=gen,
             output_type="pil", target_pixels=1 << 20)
ref_x = captured["ref"][-1]
print("ref encode input:", tuple(ref_x.shape), ref_x.dtype,
      "min", ref_x.min().item(), "max", ref_x.max().item())
del ref_pipe
torch.cuda.empty_cache()

# ours
from diffusers import AutoencoderDC, Flux2Transformer2DModel
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

transformer = Flux2Transformer2DModel.from_pretrained(
    MODEL, subfolder="transformer", torch_dtype=torch.bfloat16).to(DEVICE).eval()
text_encoder = Qwen3ForCausalLM.from_pretrained(
    MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(DEVICE).eval()
tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL, subfolder="tokenizer")
vae = AutoencoderDC.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).to(DEVICE).eval()

our_captured = []
orig_vae_encode = vae.encode


def encode_capture2(x, *a, **kw):
    our_captured.append(x.detach().clone())
    return orig_vae_encode(x, *a, **kw)


vae.encode = encode_capture2

from core.image_processing import pil_to_comfy_image
from core.pipeline import run_bluepixel

our_img = pil_to_comfy_image(img)
run_bluepixel(transformer=transformer, vae=vae, text_encoder=text_encoder,
              tokenizer=tokenizer, scaling_factor=0.4538, image=our_img,
              prompt=PROMPT, seed=SEED, steps=4, guidance=3.0,
              target_pixels=1 << 20, mu_mode="empirical", shift=0.0,
              vae_tiling=False, device=DEVICE)
our_x = our_captured[-1]
print("our encode input:", tuple(our_x.shape), our_x.dtype,
      "min", our_x.min().item(), "max", our_x.max().item())
d = (ref_x.float() - our_x.float()).abs().max().item()
print("encode-input max abs diff:", d)
