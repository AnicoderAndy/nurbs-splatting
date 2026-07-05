#!/usr/bin/env python3
"""Batch runner for vectorize.py.

Iterates over all images in a target directory, runs vectorize.py on each,
and collects the final output images into a shared folder.

Usage:
    python batch.py --target_dir path/to/images --out path/to/output [VECTORIZE_ARGS...]

Example:
    python batch.py --target_dir data/images --out results --max_paths 16 --num_iter 500
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def main():
    # ── Parse only batch-specific args; pass the rest to vectorize.py ──
    parser = argparse.ArgumentParser(
        description="Batch runner for vectorize.py.",
        # Allow unrecognised args to be forwarded to vectorize.py
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory containing target images.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Root output directory.  Per-image results go to <out>/<stem>/.",
    )
    args, extra = parser.parse_known_args()

    target_dir = Path(args.target_dir).resolve()
    out_root = Path(args.out).resolve()

    if not target_dir.is_dir():
        print(f"Error: target directory does not exist: {target_dir}")
        sys.exit(1)

    # Collect image files (sorted for reproducibility)
    images = sorted(
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"No images found in {target_dir}")
        sys.exit(1)

    finals_dir = out_root / "finals"
    finals_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} image(s) in {target_dir}")
    print(f"Output root: {out_root}")
    print(f"Extra args forwarded to vectorize.py: {extra}\n")

    vectorize_script = Path(__file__).resolve().parent / "vectorize.py"

    for idx, img_path in enumerate(images):
        stem = img_path.stem  # filename without extension
        outdir = out_root / stem

        print(f"\n{'='*60}")
        print(f"[{idx + 1}/{len(images)}] Processing: {img_path.name}")
        print(f"  outdir: {outdir}")
        print(f"{'='*60}")

        cmd = [
            sys.executable,
            str(vectorize_script),
            "--target", str(img_path),
            "--outdir", str(outdir),
            *extra,
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: vectorize.py exited with code {result.returncode} "
                  f"for {img_path.name}")
            continue

        # Copy final.png to out/finals/<stem>.png
        final_img = outdir / "final.png"
        if final_img.exists():
            dest = finals_dir / f"{stem}.png"
            shutil.copy2(str(final_img), str(dest))
            print(f"  Copied final image -> {dest}")
        else:
            print(f"  WARNING: {final_img} not found, skipping copy.")

    print(f"\nBatch complete. Final images collected in {finals_dir}/")


if __name__ == "__main__":
    main()
