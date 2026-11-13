"""Capture exact transformer kwargs from ref pipe and our core, then diff."""
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
STEPS = 4
GUIDANCE = 3.0
PROMPT = ("Change the gray sweater to burgundy red. "
          "Preserve the person, pose, accessories, and background.")
IMAGE = "/root/code/bluepixel/assets/test_input.jpg"

# ---------- ref ----------
import vins_infer as ref

buf = io.StringIO()
with redirect_stdout(buf):
    ref_pipe = ref.load_pipeline(MODEL, 0, False, None, 0.0, "empirical")

ref_kwargs = []
orig_call = ref_pipe.transformer.__call__


class RefWrap:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, *a, **kw):
        ref_kwargs.append({k: (v.detach().clone() if torch.is_tensor(v) else v)
                           for k, v in kw.items()})
        return orig_call(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


ref_pipe.transformer = RefWrap(ref_pipe.transformer)

img = Image.open(IMAGE).convert("RGB")
gen = torch.Generator(DEVICE).manual_seed(SEED)
with redirect_stdout(io.StringIO()):
    ref_pipe(image=img, instruction=PROMPT, num_inference_steps=STEPS,
             mu_mode="empirical", guidance_scale=GUIDANCE, generator=gen,
             output_type="pil", target_pixels=1 << 20)
print(f"ref captured {len(ref_kwargs)} calls")
r0 = ref_kwargs[0]
del ref_pipe
torch.cuda.empty_cache()

# ---------- ours ----------
from diffusers import AutoencoderDC, Flux2Transformer2DModel
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

transformer = Flux2Transformer2DModel.from_pretrained(
    MODEL, subfolder="transformer", torch_dtype=torch.bfloat16).to(DEVICE).eval()
text_encoder = Qwen3ForCausalLM.from_pretrained(
    MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(DEVICE).eval()
tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL, subfolder="tokenizer")
vae = AutoencoderDC.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).to(DEVICE).eval()

from bluepixel_comfyui.core.image_processing import pil_to_comfy_image
from bluepixel_comfyui.core.pipeline import run_bluepixel

our_kwargs = []
orig_fwd = transformer.forward


class OurWrap:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, *a, **kw):
        our_kwargs.append({k: (v.detach().clone() if torch.is_tensor(v) else v)
                           for k, v in kw.items()})
        return orig_fwd(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def progress_cb(d, t):
    pass


our_img = pil_to_comfy_image(img)
run_bluepixel(transformer=OurWrap(transformer), vae=vae, text_encoder=text_encoder,
              tokenizer=tokenizer, scaling_factor=0.4538, image=our_img,
              prompt=PROMPT, seed=SEED, steps=STEPS, guidance=GUIDANCE,
              target_pixels=1 << 20, mu_mode="empirical", shift=0.0,
              vae_tiling=False, device=DEVICE)
print(f"ours captured {len(our_kwargs)} calls")
o0 = our_kwargs[0]

# ---------- diff ----------
print("ref keys :", sorted(r0.keys()))
print("our keys :", sorted(o0.keys()))
for k in sorted(set(r0) | set(o0)):
    rv, ov = r0.get(k), o0.get(k)
    if isinstance(rv, torch.Tensor) and isinstance(ov, torch.Tensor):
        if rv.shape != ov.shape:
            print(f"{k}: SHAPE ref={tuple(rv.shape)} our={tuple(ov.shape)}")
        else:
            d = (rv.float() - ov.float()).abs().max().item()
            if d > 0:
                print(f"{k}: dtype ref={rv.dtype} our={ov.dtype} max_abs_diff={d:.6e}")
                if k == "hidden_states":
                    n_img = 308
                    print("  first 3 tok ref:", rv[0, :3, 0].tolist())
                    print("  first 3 tok our:", ov[0, :3, 0].tolist())
                    print("  ref img-lat tok0:", rv[0, n_img, 0].item(), "our:", ov[0, n_img, 0].item())
                    print("  ref img-lat tok-1:", rv[0, -1, 0].item(), "our:", ov[0, -1, 0].item())
            else:
                print(f"{k}: identical ({rv.dtype})")
    else:
        print(f"{k}: ref={rv!r} our={ov!r}")
