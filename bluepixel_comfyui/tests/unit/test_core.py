"""Unit tests: scheduler, buckets, int8 linear (vins_infer fb1366a semantics)."""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from bluepixel_comfyui.core import buckets as bucket_mod
from bluepixel_comfyui.core import scheduler as sched_mod
from bluepixel_comfyui.core.int8_linear import (
    Int8Linear,
    read_quantization_config,
    is_int8_model_dir,
)


class _Cfg:
    """Mimics diffusers scheduler.config attribute access."""

    def __init__(self, **kw):
        self.num_train_timesteps = 1000
        self.use_dynamic_shifting = True
        self.time_shift_type = "exponential"
        self.shift = 3.0
        self.shift_terminal = None
        for k, v in kw.items():
            setattr(self, k, v)


class TestTrainingMu:
    def test_capped_at_4096(self):
        m4096 = sched_mod.compute_training_mu(4096)
        m9000 = sched_mod.compute_training_mu(9000)
        assert math.isclose(m4096, m9000), "seq-len cap must clamp mu"

    def test_formula(self):
        expect = 0.00016927 * 4096 + 0.45666666
        assert math.isclose(sched_mod.compute_training_mu(12345), expect)


class TestEmpiricalMu:
    def test_matches_reference(self):
        sys.path.insert(0, "/root/code/bluepixel")
        from vins_infer import compute_empirical_mu as ref
        for seq in (1024, 2400, 4352, 8192):
            for steps in (1, 4, 28):
                assert math.isclose(
                    sched_mod.compute_empirical_mu(seq, steps), ref(seq, steps),
                    rel_tol=1e-12)

    def test_large_seq_uses_table_branch(self):
        # above 4300 the table branch (a2*seq+b2) applies; below it the
        # step-interpolated path — matching the reference table exactly.
        sys.path.insert(0, "/root/code/bluepixel")
        from vins_infer import compute_empirical_mu as ref
        for seq in (2048, 4096, 4300, 4301, 8192):
            assert math.isclose(
                sched_mod.compute_empirical_mu(seq, 4), ref(seq, 4), rel_tol=1e-12)


class TestComputeSigmas:
    def test_structure_descending(self):
        sig = sched_mod.compute_sigmas([1000, 750, 500, 250], _Cfg(), 4096, 4, "training")
        assert len(sig) == 4
        assert all(sig[i] > sig[i + 1] for i in range(len(sig) - 1))
        assert 0.0 < sig[-1] < 1.0  # terminal sigma NOT appended (CPS handles last step)

    def test_num_steps_truncates(self):
        sig = sched_mod.compute_sigmas([1000, 750, 500, 250], _Cfg(), 4096, 2, "training")
        assert len(sig) == 2
        assert sig == sched_mod.compute_sigmas(
            [1000, 750, 500, 250], _Cfg(), 4096, 4, "training")[:2]

    def test_matches_reference(self):
        sys.path.insert(0, "/root/code/bluepixel")
        from vins_infer import compute_sigmas as ref, ResolutionBucketSet
        for seq in (240, 1024, 4096, 4352):
            ours = sched_mod.compute_sigmas(
                [1000, 750, 500, 250], _Cfg(), seq, 4, "training")
            theirs = ref(
                [1000, 750, 500, 250], _Cfg(), seq, 4, "training")
            assert ours == theirs, (seq, ours, theirs)

    def test_empirical_mode(self):
        sig = sched_mod.compute_sigmas([1000, 750, 500, 250], _Cfg(), 1024, 4, "empirical")
        assert len(sig) == 4
        # empirical mu differs from training -> schedules differ
        sig_t = sched_mod.compute_sigmas([1000, 750, 500, 250], _Cfg(), 1024, 4, "training")
        assert any(abs(a - b) > 1e-6 for a, b in zip(sig, sig_t))

    def test_training_mu_beats_dynamic_flag(self):
        # reference: mu_mode="training" always uses training mu, even when
        # use_dynamic_shifting=False (fixed shift only when mu_mode invalid).
        cfg = _Cfg(use_dynamic_shifting=False, shift=3.0)
        sig = sched_mod.compute_sigmas([1000, 500], cfg, 1024, 2, "training")
        mu = sched_mod.compute_training_mu(1024)
        expect = [
            math.exp(mu) / (math.exp(mu) + 0.0),                # t=1.0 -> 1.0
            math.exp(mu) / (math.exp(mu) + 1.0),                # t=0.5
        ]
        assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(sig, expect))

    def test_fixed_shift_path_requires_no_mu_mode(self):
        # mu_mode outside training/empirical + !use_dynamic -> fixed logistic shift
        cfg = _Cfg(use_dynamic_shifting=False, shift=3.0)
        sig = sched_mod.compute_sigmas([1000, 500], cfg, 1024, 2, "bogus")
        expect = [3.0 * s / (1 + 2.0 * s) for s in (1.0, 0.5)]
        assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(sig, expect))


class TestCPSScheduler:
    def test_set_timesteps_appends_zero(self):
        s = sched_mod.CPSScheduler(_Cfg())
        s.set_timesteps([0.9, 0.6, 0.3])
        assert s.sigmas.tolist()[-1] == 0.0
        assert s.num_inference_steps == 3
        assert s.timesteps.tolist()[0] == 900.0

    def test_eta0_is_deterministic_euler(self):
        s = sched_mod.CPSScheduler(_Cfg())
        s.set_eta(0.0)
        s.set_timesteps([0.9, 0.5, 0.0])
        torch.manual_seed(0)
        x = torch.randn(1, 2, 8)
        v = torch.randn(1, 2, 8)
        out = s.step(v, 900.0, x, return_dict=False)[0]
        expect = x.float() + (0.5 - 0.9) * v.float()
        assert torch.allclose(out, expect.to(v.dtype), atol=1e-6)

    def test_eta09_stochastic_shaped(self):
        s = sched_mod.CPSScheduler(_Cfg())
        s.set_eta(0.9)
        s.set_timesteps([0.9, 0.5, 0.0])
        g = torch.Generator().manual_seed(3)
        x = torch.randn(1, 2, 8, generator=g)
        v = torch.randn(1, 2, 8, generator=g)
        out = s.step(v, 900.0, x, generator=g, return_dict=False)[0]
        assert out.dtype == v.dtype
        assert out.shape == x.shape
        # stochastic step differs from deterministic euler
        det = x.float() + (0.5 - 0.9) * v.float()
        assert not torch.allclose(out.float(), det, atol=1e-3)

    def test_step_index_advances(self):
        s = sched_mod.CPSScheduler(_Cfg())
        s.set_timesteps([0.9, 0.0])
        x = torch.zeros(1, 1, 4)
        v = torch.ones(1, 1, 4)
        s.step(v, 900.0, x)
        assert s._step_index == 1


class TestBuckets:
    def test_area_budget(self):
        bs = bucket_mod.ResolutionBucketSet()
        for b in bs:
            assert b["width"] % 64 == 0 and b["height"] % 64 == 0
            assert b["width"] * b["height"] <= bs.target_area
            assert b["width"] >= 64 and b["height"] >= 64

    def test_square_bucket_exact(self):
        bs = bucket_mod.ResolutionBucketSet(ratios=["1:1"])
        assert bs[0]["width"] == 4096 and bs[0]["height"] == 4096

    def test_select_log_ratio(self):
        bs = bucket_mod.ResolutionBucketSet()
        b = bs.select(300, 5000)  # extreme portrait -> 1:8 (1408x11584)
        assert b["width"] / b["height"] < 0.3
        # 5000x300 (16.67:1): the set has no 16:1 bucket; closest by log-ratio
        # is 8:1 (reference behavior, verified in test_matches_reference)
        b2 = bs.select(5000, 300)
        assert b2["ratio"] == "8:1"
        b3 = bs.select(1024, 576)  # ~16:9
        assert b3["ratio"] == "16:9"

    def test_select_square(self):
        bs = bucket_mod.ResolutionBucketSet()
        assert bs.select(1024, 1024)["ratio"] == "1:1"

    def test_matches_reference(self):
        sys.path.insert(0, "/root/code/bluepixel")
        from vins_infer import ResolutionBucketSet as RefBuckets
        ref = RefBuckets()
        ours = bucket_mod.ResolutionBucketSet()
        assert len(ref) == len(ours)
        for i in range(len(ref)):
            assert ref[i]["width"] == ours[i]["width"]
            assert ref[i]["height"] == ours[i]["height"]
            assert ref[i]["ratio"] == ours[i]["ratio"]


class TestPreprocessRefImage:
    def test_resize_crop_semantics(self):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.full((100, 50, 3), 128, np.uint8))  # tall
        t = bucket_mod.preprocess_ref_image_pil(img, 200, 100)
        assert t.shape == (3, 200, 100)
        assert -1.0 <= t.min() <= t.max() <= 1.0
        # BICUBIC on constant image introduces ~1 LSB float noise
        assert torch.allclose(t, torch.full_like(t, 128 / 255 * 2 - 1), atol=1e-6)

    def test_matches_reference(self):
        sys.path.insert(0, "/root/code/bluepixel")
        from vins_infer import preprocess_ref_image_pil as ref
        from PIL import Image
        import numpy as np
        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 255, (333, 777, 3), dtype=np.uint8))
        a = bucket_mod.preprocess_ref_image_pil(img, 200, 100)
        b = ref(img, 200, 100)
        assert torch.equal(a, b)


class TestInt8Linear:
    def test_forward_math(self):
        layer = Int8Linear(4, 3, has_bias=True)
        with torch.no_grad():
            layer.weight.copy_(torch.randint(-32, 32, (3, 4), dtype=torch.int8))
            layer.weight_scale.copy_(torch.tensor([0.1, 0.2, 0.4]))
            layer.bias.copy_(torch.tensor([1.0, -1.0, 0.5]))
        x = torch.randn(2, 4, dtype=torch.float32)
        y = layer(x)
        w = layer.weight.float()
        expect = torch.nn.functional.linear(x, w) * layer.weight_scale.float() + layer.bias
        assert torch.allclose(y, expect, atol=1e-5)

    def test_compute_dtype_follows_input(self):
        layer = Int8Linear(4, 3, has_bias=False)
        layer.weight_scale.fill_(1.0)
        x16 = torch.randn(2, 4, dtype=torch.bfloat16)
        y = layer(x16)
        assert y.dtype == torch.bfloat16

    def test_read_quantization_config_fp32(self, tmp_path):
        d = tmp_path / "transformer"
        d.mkdir()
        (d / "config.json").write_text('{"in_channels": 128}')
        assert read_quantization_config(str(d)) is None
        assert not is_int8_model_dir(str(tmp_path))

    def test_read_quantization_config_int8(self, tmp_path):
        d = tmp_path / "transformer"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(
            {"in_channels": 128,
             "quantization_config": {"quantized_keys": ["a.weight"], "scale_suffix": "_scale"}}))
        q = read_quantization_config(str(d))
        assert q["quantized_keys"] == ["a.weight"]
        assert is_int8_model_dir(str(tmp_path))


class TestIds:
    def test_latent_ids_match_reference(self):
        sys.path.insert(0, "/root/code/bluepixel")
        from vins_infer import _prepare_latent_ids as ref
        lat = torch.zeros(1, 2, 4, 6)
        a = bucket_mod._prepare_latent_ids(lat)
        b = ref(lat)
        assert torch.equal(a, b)

    def test_ref_ids_match_reference(self):
        sys.path.insert(0, "/root/code/bluepixel")
        from vins_infer import _build_ref_image_ids_dcae as ref
        lat = torch.zeros(1, 2, 4, 6)
        a = bucket_mod._build_ref_image_ids_dcae(lat)
        b = ref(lat)
        assert torch.equal(a, b)
        assert a[0, 0, 0] == 10.0

    def test_text_ids_match_reference(self):
        sys.path.insert(0, "/root/code/bluepixel")
        from vins_infer import _prepare_text_ids as ref
        pe = torch.zeros(1, 8, 4)
        assert torch.equal(bucket_mod._prepare_text_ids(pe), ref(pe))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
