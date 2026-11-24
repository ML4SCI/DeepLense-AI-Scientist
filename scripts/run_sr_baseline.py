"""Evaluate bicubic x2 super resolution on the held-out validation split.

Example:
    python scripts/run_sr_baseline.py --splits-root ../Dataset/splits

This script deliberately opens only val/. The separate test/ split stays held
out until model selection is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dlens.data.sr_dataset import SRPairDataset
from dlens.schemas._super_resolution import SRDatasetRef
from dlens.sr_metrics import METRIC_POLICY, evaluate_batches, summarize_scores
from dlens.sr_visuals import display_font, display_tile


def _save_previews(previews: list[dict], output_dir: Path) -> list[str]:
    saved: list[str] = []
    for page_start in range(0, len(previews), 5):
        page = previews[page_start : page_start + 5]
        width, row_height, top = 1050, 285, 75
        sheet = Image.new("RGB", (width, top + row_height * len(page)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 12), "Bicubic x2 validation baseline", fill="black", font=display_font(20))
        draw.text(
            (20, 40), "Images: shared sqrt stretch. Error: |bicubic-HR|, white = 0.1 or more.",
            fill=(80, 80, 80), font=display_font(13),
        )
        for row, item in enumerate(page):
            y = top + row * row_height
            panels = (
                ("LR", item["lr"], False),
                ("Bicubic", item["prediction"], False),
                ("HR", item["hr"], False),
                ("Absolute error", np.abs(item["prediction"] - item["hr"]), True),
            )
            for col, (label, pixels, is_error) in enumerate(panels):
                x = 20 + col * 255
                draw.text((x, y), label, fill="black", font=display_font(13))
                sheet.paste(display_tile(pixels / 0.1 if is_error else pixels, 240,
                                         sqrt_stretch=not is_error), (x, y + 22))
            draw.text(
                (20, y + 265),
                f"{item['sample_id']} | PSNR {item['psnr_db']:.2f} dB | SSIM {item['ssim']:.4f}",
                fill=(50, 50, 50), font=display_font(13),
            )
        name = f"bicubic_val_preview_{page_start // 5 + 1}.png"
        sheet.save(output_dir / name)
        saved.append(name)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Bicubic x2 baseline on val/ only")
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--lr-size", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--preview-seed", type=int, default=0)
    args = parser.parse_args()
    if args.lr_size < 1 or args.batch_size < 1 or args.preview_count < 0:
        parser.error("lr-size and batch-size must be positive and preview-count nonnegative")

    val_root = args.splits_root / "val"
    output_dir = args.out_dir or args.splits_root.parent / "baselines" / "bicubic_val"
    lr_shape = (args.lr_size, args.lr_size)
    hr_shape = (2 * args.lr_size, 2 * args.lr_size)
    ref = SRDatasetRef(
        root=str(val_root), lr_shape=lr_shape, hr_shape=hr_shape,
        scale_factor=2, channels=1,
    )
    dataset = SRPairDataset(ref, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    preview_indices = set(random.Random(args.preview_seed).sample(
        range(len(dataset)), min(args.preview_count, len(dataset)),
    ))
    rows: list[dict] = []
    previews: list[dict] = []
    below_zero = above_one = 0

    for lr, hr, prediction, batch in evaluate_batches(
        loader, torch.device("cpu"), include_mse=True,
    ):
        below_zero += int((prediction < 0).sum())
        above_one += int((prediction > 1).sum())
        for batch_index, row in enumerate(batch):
            index = len(rows)
            rows.append(row)
            if index in preview_indices:
                previews.append({
                    **row,
                    "lr": lr[batch_index, 0].numpy().copy(),
                    "prediction": prediction[batch_index, 0].numpy().copy(),
                    "hr": hr[batch_index, 0].numpy().copy(),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bicubic_val_per_image.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("sample_id", "mse", "psnr_db", "ssim"))
        writer.writeheader()
        writer.writerows(rows)
    preview_files = _save_previews(previews, output_dir)
    result = {
        "dataset": str(val_root.resolve()),
        "num_pairs": len(rows),
        "torch_version": torch.__version__,
        "interpolation": {"mode": "bicubic", "size": hr_shape,
                          "align_corners": False, "antialias": False, "clamp": False},
        "metrics": {
            "data_range": METRIC_POLICY["data_range"],
            "psnr_cap_db": METRIC_POLICY["psnr_cap_db"],
            "psnr": "10 * log10(data_range^2 / per-image MSE)",
            "ssim_definition": "11x11 valid Gaussian, sigma=1.5, K1=0.01, K2=0.03, population covariance",
            **summarize_scores(rows, ("mse", "psnr_db", "ssim"), extended=True),
        },
        "prediction_pixels_below_zero": below_zero,
        "prediction_pixels_above_one": above_one,
        "preview_seed": args.preview_seed,
        "preview_ids": [item["sample_id"] for item in previews],
        "preview_files": preview_files,
        "per_image_csv": csv_path.name,
    }
    json_path = output_dir / "bicubic_val_summary.json"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Validation pairs: {len(rows)}")
    print(f"PSNR: {result['metrics']['psnr_db']['mean']:.4f} ± "
          f"{result['metrics']['psnr_db']['std']:.4f} dB")
    print(f"SSIM: {result['metrics']['ssim']['mean']:.6f} ± "
          f"{result['metrics']['ssim']['std']:.6f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
