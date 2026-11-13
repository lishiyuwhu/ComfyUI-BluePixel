"""Unit tests: image conversion + BluePixelEngine with tiny fake models.

Mirrors vins_infer fb1366a semantics: no CFG, bucket resolution, CPS stepping.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.buckets import (
    ResolutionBucketSet,
    pack_flux2_latents,
    unpack_flux2_latents,
    _prepare_latent_ids,
    _build_ref_image_ids_dcae,
)
from core.image_processing import (
    comfy_image_to_tensor,
    tensor_to_comfy_image,
    tensor_to_pil,
)
from core.pipeline import BluePixelEngine
from core.scheduler import CPSScheduler


class _Cfg:
    num_train_timesteps = 1000
    use_dynamic_shifting = True
    time_shift_type = "exponential"
    shift = 3.0
    shift_terminal = None


class _DictCfg(_Cfg):
    """Engine expects scheduler.config; wrap via simple namespace with config attr."""
    def __init__(self):
        self.config = _Cfg()


class FakeTokenizer:
    def apply_chat_template(self, messages, **kw):
        return messages[0]["content"]

    def __call__(self, texts, **kw):
        from transformers import BatchEncoding
        return BatchEncoding({
            "input_ids": torch.zeros(len(texts), 8, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 8, dtype=torch.long),
        })


class FakeTextEncoder(torch.nn.Module):
    """Hidden layers 9/18/27 of constant value k -> embed mean = 18 (as in real)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.hidden = hidden
        self.dtype = torch.float32

    def forward(self, input_ids=None, attention_mask=None, **kw):
        B, L = input_ids.shape
        from transformers.modeling_outputs import BaseModelOutputWithPast

        class Out:
            hidden_states = [None] * 28
            hidden_states[9] = torch.full((B, L, self.hidden), 9.0)
            hidden_states[18] = torch.full((B, L, self.hidden), 18.0)
            hidden_states[27] = torch.full((B, L, self.hidden), 27.0)
        return Out()


class FakeTransformer(torch.nn.Module):
    """v = 0.1*x + mean(text_embeds); deterministic & shape-checked."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.config = type("C", (), {"in_channels": in_channels})()
        self.calls = []

    def forward(self, hidden_states, timestep, guidance, encoder_hidden_states,
                txt_ids, img_ids, return_dict=True):
        assert guidance is None, "distilled ckpt: guidance must be None"
        self.calls.append({
            "timestep": timestep.detach().clone(),
            "n_tokens": hidden_states.shape[1],
        })
        bias = encoder_hidden_states.mean(dim=(1, 2), keepdim=True)
        out = hidden_states * 0.1 + bias
        if not return_dict:
            return (out,)
        return type("O", (), {"sample": out})


class FakeVAE(torch.nn.Module):
    def __init__(self, factor=64):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32
        self.factor = factor
        self.config = type("C", (), {"scaling_factor": 0.4538})()
        self.device = torch.device("cpu")

    def encode(self, x):
        f = self.factor
        class Out:
            def __getitem__(self, i):
                return torch.ones(1, 4, x.shape[-2] // f, x.shape[-1] // f) * 0.5
        return Out()

    def decode(self, z, return_dict=True):
        f = self.factor
        class Out:
            def __getitem__(self, i):
                img = 0.3 * torch.tanh(z[:, :3] * 0.01)
                img = img.repeat_interleave(f, dim=2).repeat_interleave(f, dim=3)
                return img[:, :, : z.shape[-2] * f, : z.shape[-1] * f]
        return Out()


def _make_engine(device=torch.device("cpu"), dtype=torch.float32):
    """Engine with small buckets + small fake VAE factor for fast CPU tests.

    VAE factor == bucket multiple so latent*factor == bucket dims (as in the
    real pipeline where both are 64).
    """
    eng = BluePixelEngine(
        transformer=FakeTransformer(), vae=FakeVAE(factor=16),
        text_encoder=FakeTextEncoder(), tokenizer=FakeTokenizer(),
        scheduler=_DictCfg(), weight_dtype=dtype, device=device, is_int8=False,
        bucket_area=65536, bucket_multiple=16,
    )
    # shrink min_side so tiny buckets work
    eng.bucket_set = ResolutionBucketSet(
        ratios=["1:1", "1:2", "2:1"], target_area=65536, multiple=16, min_side=16)
    return eng


class TestImageConversion:
    def test_roundtrip(self):
        img = torch.rand(1, 32, 32, 3)
        t = comfy_image_to_tensor(img)
        assert t.shape == (1, 3, 32, 32)
        back = tensor_to_comfy_image(t)
        assert torch.allclose(back, img, atol=1e-6)

    def test_range_validation(self):
        bad = torch.full((1, 4, 4, 3), 2.0)
        with pytest.raises(ValueError):
            comfy_image_to_tensor(bad)

    def test_tensor_to_pil_uint8_round(self):
        u = torch.randint(0, 256, (1, 3, 8, 8)).float()
        x = (u / 255.0) * 2 - 1
        pil = tensor_to_pil(x)
        import numpy as np
        arr = np.asarray(pil)
        assert np.array_equal(arr, u[0].permute(1, 2, 0).numpy().astype("uint8"))


class TestEngineGenerate:
    def test_guidance_none_single_forward(self):
        eng = _make_engine()
        ft = eng.transformer
        img = torch.rand(1, 64, 64, 3)
        res = eng.generate(image=img, instruction="p", seed=1, num_steps=2)
        # 2 steps -> exactly 2 forwards (no CFG doubling)
        assert len(ft.calls) == 2
        assert res[1]["steps"] == 2
        assert res[1]["cfg_enabled"] is False

    def test_bucket_resolution(self):
        eng = _make_engine()
        img = torch.rand(1, 200, 100, 3)  # H=200, W=100 -> portrait 0.5
        res = eng.generate(image=img, instruction="p", seed=1, num_steps=1)
        m = res[1]
        assert m["width"] % 16 == 0 and m["height"] % 16 == 0
        assert m["width"] * m["height"] <= 65536
        assert abs(m["width"] / m["height"] - 0.5) < 0.05  # 1:2 bucket
        # output matches bucket dims (NHWC: shape[1]=H, shape[2]=W)
        assert res[0].shape[1] == m["height"] and res[0].shape[2] == m["width"]

    def test_output_range_and_shape(self):
        eng = _make_engine()
        img = torch.rand(1, 64, 64, 3)
        out, m = eng.generate(image=img, instruction="p", seed=1, num_steps=2)
        assert out.ndim == 4 and out.shape[-1] == 3
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.dtype == torch.float32

    def test_determinism(self):
        eng = _make_engine()
        img = torch.rand(1, 64, 64, 3)
        # eta=0: deterministic Euler; eta>0 draws CPS noise from a fresh
        # generator per run — still deterministic for equal seeds, but the
        # roundtrip through the noise path exercises float non-associativity,
        # so verify bitwise determinism in the eta=0 path.
        kw = dict(image=img, instruction="p", num_steps=2, cps_eta=0.0)
        a = eng.generate(seed=7, **kw)[0]
        b = eng.generate(seed=7, **kw)[0]
        assert torch.equal(a, b)
        c = eng.generate(seed=8, **kw)[0]
        assert not torch.equal(a, c)

    def test_eta09_same_seed_stable(self):
        eng = _make_engine()
        img = torch.rand(1, 64, 64, 3)
        kw = dict(image=img, instruction="p", num_steps=2, cps_eta=0.9)
        a = eng.generate(seed=7, **kw)[0]
        b = eng.generate(seed=7, **kw)[0]
        # same seed -> same noise draw sequence; allow 1-LSB round flips only
        assert (a - b).abs().max().item() < 2.0 / 255

    def test_cps_eta_stochastic(self):
        eng = _make_engine()
        img = torch.rand(1, 64, 64, 3)
        kw = dict(image=img, instruction="p", num_steps=3)
        a = eng.generate(seed=7, cps_eta=0.9, **kw)[0]
        b = eng.generate(seed=7, cps_eta=0.0, **kw)[0]
        assert not torch.equal(a, b), "eta>0 must add stochasticity"

    def test_mu_mode_validation(self):
        eng = _make_engine()
        with pytest.raises(ValueError):
            eng.generate(image=torch.rand(1, 32, 32, 3), instruction="p",
                         mu_mode="bogus")

    def test_missing_image(self):
        eng = _make_engine()
        with pytest.raises(ValueError):
            eng.generate(image=None, instruction="p")

    def test_metadata_fields(self):
        eng = _make_engine()
        out, m = eng.generate(image=torch.rand(1, 64, 64, 3), instruction="p",
                              seed=5, num_steps=4, cps_eta=0.5, mu_mode="training")
        for key in ("width", "height", "bucket_ratio", "bucket_id", "steps",
                    "denoising_steps", "cps_eta", "seed", "mu_mode",
                    "image_seq_len", "cfg_enabled", "is_int8"):
            assert key in m
        assert m["denoising_steps"] == [1000, 750, 500, 250][:4]
        assert m["is_int8"] is False
        assert m["cfg_enabled"] is False


class TestLatentHelpers:
    def test_pack_unpack_roundtrip(self):
        lat = torch.randn(2, 4, 8, 8)
        packed = pack_flux2_latents(lat)
        assert packed.shape == (2, 64, 4)
        out = unpack_flux2_latents(packed, height=8, width=8)
        assert torch.equal(out, lat)

    def test_unpack_shape_mismatch(self):
        packed = torch.zeros(1, 5, 4)
        with pytest.raises(ValueError):
            unpack_flux2_latents(packed, height=2, width=2)

    def test_latent_ids_layout(self):
        lat = torch.zeros(1, 4, 4, 6)
        ids = _prepare_latent_ids(lat)
        assert ids.shape == (1, 24, 4)
        assert ids[0, 5, 1].item() == 0 and ids[0, 5, 2].item() == 5

    def test_ref_ids_time_scale(self):
        lat = torch.zeros(1, 4, 2, 2)
        ids = _build_ref_image_ids_dcae(lat)
        assert ids[0, 0, 0].item() == 10.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
