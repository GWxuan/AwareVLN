#!/usr/bin/env python3
"""Prepare and upload AwareVLN checkpoints to Hugging Face AwareVLN/ck."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "gwx22/AwareVLN"
STAGING = Path(__file__).resolve().parents[1] / ".hf_ck_staging"
SRC_NAVILA = Path(__file__).resolve().parents[1] / "ck" / "navila-llama3-8b-8f"
SRC_AWAREVLN = Path(
    "/mnt/share/algorithm/kimi/outputs/"
    "modified_navila_mulnodes_newsam_withfollow_human_fewcorr_contnull_5e_3node"
)
COMPONENTS = ("config.json", "llm", "mm_projector", "vision_tower")


def _replace_paths(obj, old_prefix: str, new_prefix: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith(old_prefix):
                obj[k] = v.replace(old_prefix, new_prefix, 1)
            else:
                _replace_paths(v, old_prefix, new_prefix)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and item.startswith(old_prefix):
                obj[i] = item.replace(old_prefix, new_prefix, 1)
            else:
                _replace_paths(item, old_prefix, new_prefix)


def patch_config(config_path: Path, *, old_prefix: str, new_prefix: str, architecture: str | None):
    with config_path.open() as f:
        cfg = json.load(f)
    _replace_paths(cfg, old_prefix, new_prefix)
    if architecture is not None:
        cfg["architectures"] = [architecture]
    with config_path.open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def stage_checkpoints():
    for name in ("navila-llama3-8b-8f", "awarevln"):
        (STAGING / name).mkdir(parents=True, exist_ok=True)
    src_map = {
        "navila-llama3-8b-8f": SRC_NAVILA,
        "awarevln": SRC_AWAREVLN,
    }
    for subdir, src in src_map.items():
        dst = STAGING / subdir
        for comp in COMPONENTS:
            src_path = src / comp
            dst_path = dst / comp
            if dst_path.exists():
                if dst_path.is_dir():
                    shutil.rmtree(dst_path)
                else:
                    dst_path.unlink()
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    navila_cfg = STAGING / "navila-llama3-8b-8f" / "config.json"
    patch_config(
        navila_cfg,
        old_prefix="./checkpoints/vila-long-8b-8f-scanqa-rxr-real-v1-seed10-bs10-1e4",
        new_prefix="./navila-llama3-8b-8f",
        architecture=None,
    )

    aware_cfg = STAGING / "awarevln" / "config.json"
    old = str(SRC_AWAREVLN)
    patch_config(
        aware_cfg,
        old_prefix=old,
        new_prefix="./awarevln",
        architecture="AwareVLNModel",
    )


def upload():
    readme = STAGING / "README.md"
    if not readme.exists():
        raise FileNotFoundError(f"Missing {readme}")
    gitattrs = STAGING / ".gitattributes"
    if not gitattrs.exists() and (SRC_NAVILA / ".gitattributes").exists():
        shutil.copy2(SRC_NAVILA / ".gitattributes", gitattrs)

    cmd = [
        "hf",
        "upload",
        REPO_ID,
        str(STAGING),
        ".",
        "--repo-type",
        "model",
        "--commit-message",
        "Add navila-llama3-8b-8f and awarevln checkpoints",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "upload-only":
        upload()
        return
    stage_checkpoints()
    if "--upload" in sys.argv:
        upload()


if __name__ == "__main__":
    main()
