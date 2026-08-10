"""Split a directory of ``<class>_*.npy`` DeepLense images into train/ and val/.

Files are hard-linked (no extra disk) into ``<out>/train`` and ``<out>/val`` with a
seeded shuffle, preserving the ``<class>_NNNNN.npy`` naming the DatasetRef loaders
expect.

Usage:
    uv run python scripts/prepare_deeplense_dataset.py \
        --src ~/data/raw --out ~/data/model1 --classes no_sub,cdm,axion --val-frac 0.2
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Split DeepLense .npy images into train/val")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--classes", required=True, help="Comma-separated class names.")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    for split in ("train", "val"):
        Path(args.out, split).mkdir(parents=True, exist_ok=True)

    for cls in args.classes.split(","):
        files = sorted(glob.glob(os.path.join(args.src, f"{cls}_*.npy")))
        if not files:
            raise SystemExit(f"No files for class {cls!r} under {args.src}")
        rng.shuffle(files)
        n_val = int(len(files) * args.val_frac)
        for split, chunk in (("val", files[:n_val]), ("train", files[n_val:])):
            for i, fp in enumerate(chunk):
                dst = Path(args.out, split, f"{cls}_{i:05d}.npy")
                if not dst.exists():
                    os.link(fp, dst)
        print(f"{cls}: {len(files) - n_val} train / {n_val} val")
    print(f"done -> {args.out}/train  {args.out}/val")


if __name__ == "__main__":
    main()
