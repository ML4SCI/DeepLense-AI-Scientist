"""Save fixed validation comparisons for an iterative SR baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from dlens.data.sr_dataset import SRPairDataset
from dlens.sr_metrics import bicubic, score_images
from dlens.sr_visuals import fixed_sample_ids, render_grid

BASELINE_CODE = Path(__file__).resolve().parents[1] / "baselines" / "atal_gupta"
sys.path.insert(0, str(BASELINE_CODE))
from iterative_model import SimpleUnet, predict_pair  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to best_model.pt in the baseline directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint or args.baseline_dir / "best_model.pt",
                            map_location=device, weights_only=True)
    summary_path = args.baseline_dir / "baseline_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if "checkpoint_epoch" in summary and summary["checkpoint_epoch"] != checkpoint.get("epoch"):
        parser.error("checkpoint epoch does not match the baseline evaluation summary")
    steps = summary.get("inference_steps", 8)
    config = checkpoint["config"]
    model = SimpleUnet(config["base_width"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = SRPairDataset(args.splits_root / "val")
    positions = {sample_id: index for index, (_, _, sample_id) in enumerate(dataset._pairs)}
    sample_ids = fixed_sample_ids(list(positions), count=4)
    rows = []
    with torch.inference_mode():
        for sample_id in sample_ids:
            lr, hr, _ = dataset[positions[sample_id]]
            low, high = lr[None].to(device), hr[None].to(device)
            prediction = predict_pair(model, low, high, timesteps=config["timesteps"],
                                      steps=steps, seed=config["seed"] + positions[sample_id])
            reference = bicubic(low, high)
            scores = [score_images(image, high, [sample_id])[0] for image in (reference, prediction)]
            captions = [sample_id, *(f"{row['psnr_db']:.2f} dB / {row['ssim']:.4f} SSIM"
                                     for row in scores), "HST target"]
            rows.append(([lr, reference[0], prediction[0], hr], captions))
    render_grid(args.output, "HSC to HST baseline comparison",
                "Fixed validation pairs. Square-root display stretch. Scores use unchanged pixels.",
                ["HSC LR", "Bicubic", "Iterative baseline", "HST HR"], rows,
                tile=160, column_width=200, row_height=215, sqrt_stretch=True)
    print(f"Saved baseline comparison: {args.output}", flush=True)


if __name__ == "__main__":
    main()
