#!/usr/bin/env python3
"""Publish this repository to the Comfy Registry (equivalent to `comfy node publish`).

Steps (mirrors comfy-cli's registry/api.py + command/custom_nodes/command.py):
  1. POST /publishers/{publisherId}/nodes/{nodeId}/versions  -> {node_version, signedUrl}
  2. zip git-tracked files, honoring .comfyignore
  3. PUT the zip to signedUrl with Content-Type: application/zip
"""
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import tomli
import urllib.request
import urllib.error

REPO = Path(__file__).resolve().parent
BASE_URL = "https://api.comfy.org"
ZIP_NAME = "node.zip"


def load_config():
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomli.load(f)


def load_comfyignore():
    p = REPO / ".comfyignore"
    patterns = []
    if p.is_file():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def ignored(rel_path: str, patterns) -> bool:
    parts = rel_path.split("/")
    for pat in patterns:
        pat = pat.rstrip("/")
        if fnmatch.fnmatch(rel_path, pat):
            return True
        if fnmatch.fnmatch(rel_path, pat + "/*"):
            return True
        if any(fnmatch.fnmatch(part, pat) for part in parts):
            return True
    return False


def git_tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def make_zip(patterns) -> Path:
    zip_path = REPO / ZIP_NAME
    files = git_tracked_files()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            if rel == ZIP_NAME or ignored(rel, patterns):
                continue
            src = REPO / rel
            if src.is_file():
                zf.write(src, rel)
    print(f"  archived {len(files)} tracked files -> {zip_path.name} "
          f"({zip_path.stat().st_size / 1024:.1f} KB)")
    return zip_path


def post_json(url: str, payload: dict):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        # never echo the token back
        body = body.replace(payload.get("personal_access_token", ""), "<TOKEN>")
        return e.code, body


def put_zip(url: str, zip_path: Path) -> int:
    data = zip_path.read_bytes()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/zip"},
                                 method="PUT")
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status


def main() -> int:
    token_file = Path("/root/RegistryPublishingAPIKey.txt")
    if not token_file.is_file():
        print("ERROR: /root/RegistryPublishingAPIKey.txt not found")
        return 1
    token = token_file.read_text().strip()

    cfg = load_config()
    project = cfg["project"]
    comfy = cfg["tool"]["comfy"]
    publisher_id = comfy["PublisherId"]
    node_id = project["name"]

    deps_file = cfg.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("dependencies", {})
    deps = []
    for rel in deps_file.get("file", []):
        for line in (REPO / rel).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(line)

    # registry expects classifier values with the "Operating System :: " /
    # "Environment :: " prefixes stripped (see comfy-cli config_parser).
    classifiers = project.get("classifiers", [])
    OS_VALUES = [c[len("Operating System :: "):] for c in classifiers
                 if c.startswith("Operating System :: ")]
    ACCEL_VALUES = [c[len("Environment :: "):] for c in classifiers
                    if c.startswith("Environment :: ")]

    license_field = project.get("license", {})
    license_value = license_field.get("file") if isinstance(license_field, dict) else license_field

    body = {
        "personal_access_token": token,
        "node": {
            "id": node_id,
            "description": project.get("description"),
            "icon": comfy.get("Icon"),
            "name": comfy.get("DisplayName"),
            "license": license_value,
            "repository": project.get("urls", {}).get("Repository"),
            "banner_url": comfy.get("Banner"),
            "supported_os": OS_VALUES,
            "supported_accelerators": ACCEL_VALUES,
            "supported_comfyui_version": comfy.get("requires-comfyui"),
            "supported_comfyui_frontend_version": None,
        },
        "node_version": {
            "version": project["version"],
            "dependencies": deps,
            "supported_os": OS_VALUES,
            "supported_accelerators": ACCEL_VALUES,
            "supported_comfyui_version": comfy.get("requires-comfyui"),
            "supported_comfyui_frontend_version": None,
        },
    }

    url = f"{BASE_URL}/publishers/{publisher_id}/nodes/{node_id}/versions"
    print(f"Publishing {publisher_id}/{node_id} v{project['version']} ...")
    status, resp = post_json(url, body)
    if status != 201:
        print(f"ERROR: registry rejected the publish request ({status})")
        print(resp if isinstance(resp, str) else json.dumps(resp, indent=2)[:2000])
        return 1

    version = resp["node_version"]
    signed_url = resp["signedUrl"]
    print(f"  accepted: version_id={version.get('id')} status={version.get('status')}")

    patterns = load_comfyignore()
    zip_path = make_zip(patterns)

    print("  uploading archive ...")
    code = put_zip(signed_url, zip_path)
    print(f"  upload status: {code}")
    zip_path.unlink(missing_ok=True)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
