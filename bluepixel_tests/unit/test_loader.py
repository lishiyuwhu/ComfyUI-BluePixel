"""Unit tests: model dir validation + fingerprinting (no weights needed)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from runtime import loader as loader_mod


@pytest.fixture()
def fake_model_dir(tmp_path):
    """Minimal diffusers-layout skeleton with index files but no weights."""
    for sub, files in loader_mod.REQUIRED_FILES.items():
        d = tmp_path / sub
        d.mkdir()
        for f in files:
            p = d / f
            if f.endswith(".index.json"):
                shard_map = {}
                if sub == "transformer":
                    shard_map["a.weight"] = "diffusion_pytorch_model-00001-of-00002.safetensors"
                    shard_map["b.weight"] = "diffusion_pytorch_model-00002-of-00002.safetensors"
                    (d / shard_map["a.weight"]).write_bytes(b"")
                    (d / shard_map["b.weight"]).write_bytes(b"")
                if sub == "text_encoder":
                    shard_map["t.weight"] = "model-00001-of-00001.safetensors"
                    (d / shard_map["t.weight"]).write_bytes(b"")
                p.write_text(json.dumps({"weight_map": shard_map}))
            elif f == "config.json":
                cfg = {"scaling_factor": 0.4538} if sub == "vae" else {"in_channels": 128}
                p.write_text(json.dumps(cfg))
            else:
                p.write_text("{}")
    return tmp_path


class TestValidation:
    def test_valid_dir(self, fake_model_dir):
        ok, problems = loader_mod.validate_model_dir(str(fake_model_dir))
        assert ok, problems

    def test_missing_subfolder(self, fake_model_dir):
        import shutil
        shutil.rmtree(fake_model_dir / "vae")
        ok, problems = loader_mod.validate_model_dir(str(fake_model_dir))
        assert not ok
        assert any("vae" in p for p in problems)

    def test_missing_shard(self, fake_model_dir):
        (fake_model_dir / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors").unlink()
        ok, problems = loader_mod.validate_model_dir(str(fake_model_dir))
        assert not ok
        assert any("shard" in p for p in problems)

    def test_nonexistent_dir(self):
        ok, problems = loader_mod.validate_model_dir("/definitely/not/here")
        assert not ok


class TestFingerprint:
    def test_stable(self, fake_model_dir):
        f1 = loader_mod.compute_fingerprint(str(fake_model_dir))
        f2 = loader_mod.compute_fingerprint(str(fake_model_dir))
        assert f1 == f2
        assert len(f1) == 16

    def test_changes_with_size(self, fake_model_dir):
        f1 = loader_mod.compute_fingerprint(str(fake_model_dir))
        # transformer shard sizes are fingerprinted via the index-json pass below;
        # mutate a fingerprinted file: vae safetensors (REQUIRED_FILES entry)
        p = fake_model_dir / "vae" / "diffusion_pytorch_model.safetensors"
        old_size = p.stat().st_size
        p.write_bytes(b"x" * (old_size + 7))
        f2 = loader_mod.compute_fingerprint(str(fake_model_dir))
        assert f1 != f2


class TestConfigReads:
    def test_scaling_factor(self, fake_model_dir):
        assert loader_mod.read_vae_scaling_factor(str(fake_model_dir)) == 0.4538

    def test_transformer_config(self, fake_model_dir):
        cfg = loader_mod.read_transformer_config(str(fake_model_dir))
        assert cfg["in_channels"] == 128

    def test_missing(self, tmp_path):
        assert loader_mod.read_vae_scaling_factor(str(tmp_path)) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
