#!/usr/bin/env python3
"""Package training-only data under .hf_data for Hugging Face upload."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "data"
DST = Path(__file__).resolve().parents[1] / ".hf_data"

REASON_SETS = (
    ("r2r", "videos_new"),
    ("rxr", "videos"),
    ("r2rfollow", "videos"),
    ("rxrfollow", "videos"),
)

ANNO_FILES = ("annotations_shuffle_uni.json", "cot_new.json")
HUMAN_ANNO = "annotations_shuffled.json"
HUMAN_FRAMES = "raw_frames"


def episodes_from_vlnce_ann(ann_path: Path) -> set[str]:
    with ann_path.open() as f:
        data = json.load(f)
    episodes: set[str] = set()
    for sample in data:
        for frame in sample.get("frames", []):
            ep = frame.split("/")[0]
            if ep:
                episodes.add(ep)
    return episodes


def hardlink_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if not src.exists():
        print(f"[WARN] missing {src}", file=sys.stderr)
        return
    subprocess.run(["cp", "-al", str(src), str(dst)], check=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def package_reason(name: str, frames_dir: str) -> None:
    ann_dir = SRC / "reason" / name / "_anno_cot"
    out_ann = DST / "reason" / name / "_anno_cot"
    for fname in ANNO_FILES:
        copy_file(ann_dir / fname, out_ann / fname)

    ann_path = ann_dir / ANNO_FILES[0]
    episodes = episodes_from_vlnce_ann(ann_path)
    src_frames = SRC / "reason" / name / frames_dir
    dst_frames = DST / "reason" / name / frames_dir
    dst_frames.mkdir(parents=True, exist_ok=True)

    total = len(episodes)
    print(f"[{name}] {total} episode dirs -> {dst_frames}", flush=True)
    for i, ep in enumerate(sorted(episodes)):
        hardlink_copy(src_frames / ep, dst_frames / ep)
        if (i + 1) % 1000 == 0 or i + 1 == total:
            print(f"  [{name}] {i + 1}/{total}", flush=True)


def package_human() -> None:
    ann_path = SRC / "Human" / HUMAN_ANNO
    copy_file(ann_path, DST / "Human" / HUMAN_ANNO)

    episodes = episodes_from_vlnce_ann(ann_path)
    src_frames = SRC / "Human" / HUMAN_FRAMES
    dst_frames = DST / "Human" / HUMAN_FRAMES
    dst_frames.mkdir(parents=True, exist_ok=True)

    total = len(episodes)
    print(f"[human] {total} episode dirs -> {dst_frames}", flush=True)
    for i, ep in enumerate(sorted(episodes)):
        hardlink_copy(src_frames / ep, dst_frames / ep)
        if (i + 1) % 500 == 0 or i + 1 == total:
            print(f"  [human] {i + 1}/{total}", flush=True)


def main() -> None:
    if DST.exists():
        print(f"Using existing {DST} (incremental hardlink copy)")
    DST.mkdir(parents=True, exist_ok=True)

    for name, frames_dir in REASON_SETS:
        package_reason(name, frames_dir)

    if os.environ.get("PACKAGE_HUMAN", "").lower() in ("1", "true", "yes"):
        if (SRC / "Human" / HUMAN_ANNO).exists():
            package_human()
        else:
            print("[human] skipped (annotations not found)", flush=True)
    else:
        print("[human] skipped (set PACKAGE_HUMAN=1 to include)", flush=True)

    print(f"Done. Output: {DST}", flush=True)


if __name__ == "__main__":
    main()
