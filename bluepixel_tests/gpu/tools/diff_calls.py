"""Diff exact transformer inputs at each step: ref vs ours."""
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

ref_calls = []
orig_call = ref_pipe.transformer.__call__


class RefWrap:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, *a, **kw):
        ref_calls.append(({k: (v.detach().clone() if torch.is_tensor(v) else v)
                           for k, v in kw.items()}))
        return orig_call(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


ref_pipe.transformer = RefWrap(ref_pipe.transformer)
img = Image.open(IMAGE).convert("RGB")
gen = torch.Generator(DEVICE).manual_seed(SEED)
with redirect_stdout(io.StringIO()):
    ref_pipe(image=img, instruction=PROMPT, num_inference_steps=4, mu_mode="empirical",
             guidance_scale=3.0, generator=gen, output_type="pil", target_pixels=1 << 20)
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

from core.image_processing import pil_to_comfy_image
from core.pipeline import run_bluepixel

our_calls = []
orig_fwd = transformer.forward


class OurWrap:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, *a, **kw):
        our_calls.append(({k: (v.detach().clone() if torch.is_tensor(v) else v)
                           for k, v in kw.items()}))
        return orig_fwd(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


our_img = pil_to_comfy_image(img)
run_bluepixel(transformer=OurWrap(transformer), vae=vae, text_encoder=text_encoder,
              tokenizer=tokenizer, scaling_factor=0.4538, image=our_img,
              prompt=PROMPT, seed=SEED, steps=4, guidance=3.0,
              target_pixels=1 << 20, mu_mode="empirical", shift=0.0,
              vae_tiling=False, device=DEVICE)

print(f"ref calls={len(ref_calls)} ours={len(our_calls)}")
for ci in range(min(len(ref_calls), len(our_calls))):
    r, o = ref_calls[ci], our_calls[ci]
    diffs = []
    for k in sorted(set(r) | set(o)):
        rv, ov = r.get(k), o.get(k)
        if isinstance(rv, torch.Tensor) and isinstance(ov, torch.Tensor):
            if rv.shape != ov.shape:
                diffs.append(f"{k}:SHAPE{tuple(rv.shape)}/{tuple(ov.shape)}")
            else:
                d = (rv.float() - ov.float()).abs().max().item()
                if d > 0:
                    detail = ""
                    if k == "timestep":
                        detail = f" ref={rv.float().tolist()} our={ov.float().tolist()}"
                    diffs.append(f"{k}:max={d:.4e}({rv.dtype}){detail}")
        elif rv != ov:
            diffs.append(f"{k}:{rv!r}/{ov!r}")
    print(f"call {ci}: {'IDENTICAL' if not diffs else '; '.join(diffs)}")
