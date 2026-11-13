"""Model fingerprinting and validation (no torch required at import time)."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

REQUIRED_SUBFOLDERS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")
REQUIRED_FILES = {
    "transformer": ("config.json", "diffusion_pytorch_model.safetensors.index.json"),
    "vae": ("config.json", "diffusion_pytorch_model.safetensors"),
    "text_encoder": ("config.json", "model.safetensors.index.json"),
    "tokenizer": ("tokenizer_config.json", "tokenizer.json"),
    "scheduler": ("scheduler_config.json",),
}


def compute_fingerprint(model_path: str) -> str:
    """Stable fingerprint over config files + shard index sizes."""
    h = hashlib.sha256()
    entries = []
    for sub in REQUIRED_SUBFOLDERS:
        cfg = os.path.join(model_path, sub, "config.json")
        if os.path.isfile(cfg):
            entries.append((cfg, _file_hash(cfg)))
        for req in REQUIRED_FILES[sub]:
            p = os.path.join(model_path, sub, req)
            if os.path.isfile(p):
                try:
                    entries.append((p, str(int(os.path.getsize(p)))))
                except OSError:
                    pass
    for path, sig in sorted(entries):
        h.update(path.encode("utf-8", "replace"))
        h.update(sig.encode())
    return h.hexdigest()[:16]


def _file_hash(path: str, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()


def validate_model_dir(model_path: str) -> Tuple[bool, List[str]]:
    """Static checks: folders, required files, shard index presence."""
    problems: List[str] = []
    if not os.path.isdir(model_path):
        return False, [f"model dir not found: {model_path}"]
    for sub in REQUIRED_SUBFOLDERS:
        sub_path = os.path.join(model_path, sub)
        if not os.path.isdir(sub_path):
            problems.append(f"missing subfolder: {sub}/")
            continue
        for req in REQUIRED_FILES[sub]:
            p = os.path.join(sub_path, req)
            if not os.path.isfile(p):
                problems.append(f"missing file: {sub}/{req}")
    # shard existence via index
    idx_path = os.path.join(model_path, "transformer", "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(idx_path):
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            shards = sorted(set(idx.get("weight_map", {}).values()))
            for shard in shards:
                if not os.path.isfile(os.path.join(model_path, "transformer", shard)):
                    problems.append(f"missing transformer shard: {shard}")
        except (OSError, ValueError) as e:
            problems.append(f"unreadable transformer index: {e}")
    tidx = os.path.join(model_path, "text_encoder", "model.safetensors.index.json")
    if os.path.isfile(tidx):
        try:
            with open(tidx, "r", encoding="utf-8") as f:
                idx = json.load(f)
            shards = sorted(set(idx.get("weight_map", {}).values()))
            for shard in shards:
                if not os.path.isfile(os.path.join(model_path, "text_encoder", shard)):
                    problems.append(f"missing text_encoder shard: {shard}")
        except (OSError, ValueError) as e:
            problems.append(f"unreadable text_encoder index: {e}")
    return len(problems) == 0, problems


def read_transformer_config(model_path: str) -> Optional[Dict]:
    p = os.path.join(model_path, "transformer", "config.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_vae_scaling_factor(model_path: str) -> Optional[float]:
    p = os.path.join(model_path, "vae", "config.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        sf = cfg.get("scaling_factor")
        return float(sf) if sf is not None else None
    except (OSError, ValueError, TypeError):
        return None
