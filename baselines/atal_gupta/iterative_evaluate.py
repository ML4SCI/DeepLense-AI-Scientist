"""Evaluate the compact iterative autoencoder on every validation pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from iterative_model import SimpleUnet, predict_pair
from torch.utils.data import DataLoader

from dlens.data.sr_dataset import SRPairDataset
from dlens.sr_metrics import METRIC_POLICY, METRICS, bicubic, evaluate_batches, score_images, summarize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.inference_steps < 1:
        parser.error("inference steps must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    last = torch.load(args.out_dir / "last.pt", map_location="cpu", weights_only=True)
    if (last["epoch"] != config["epochs"] or last["config"] != config):
        parser.error("complete training before evaluating this baseline")
    history = [json.loads(line) for line in (args.out_dir / "history.jsonl").read_text().splitlines()]
    if len(history) != config["epochs"]:
        parser.error("history does not contain all training epochs")
    model = SimpleUnet(config["base_width"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    pairs = SRPairDataset(args.splits_root / "val", augment=False)
    loader = DataLoader(pairs, batch_size=1, shuffle=False, num_workers=0)
    def predict(lr: torch.Tensor, hr: torch.Tensor, index: int) -> torch.Tensor:
        return predict_pair(model, lr, hr, timesteps=config["timesteps"],
                            steps=args.inference_steps, seed=config["seed"] + index)

    rows = []
    model_scores = []
    bicubic_scores = []
    for lr, hr, _, batch in evaluate_batches(loader, device, predict):
        scores = score_images(bicubic(lr, hr), hr, [row["sample_id"] for row in batch])
        model_scores.extend(batch)
        bicubic_scores.extend(scores)
        for model_row, bicubic_row in zip(batch, scores, strict=True):
            rows.append({"sample_id": model_row["sample_id"],
                         **{f"model_{key}": model_row[key] for key in METRICS},
                         **{f"bicubic_{key}": bicubic_row[key] for key in METRICS}})
        print(f"Validated {len(rows)}/{len(pairs)} pairs", flush=True)
    model_means = {key: stats["mean"] for key, stats in summarize_scores(model_scores).items()}
    bicubic_means = {key: stats["mean"] for key, stats in summarize_scores(bicubic_scores).items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "validation_per_image.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"model": "Iterative Auto Encoder",
               "epochs_planned": config["epochs"],
               "epochs_completed": len(history),
               "optimizer_steps": sum(row["train_batches"] for row in history),
               "steps_per_epoch": history[-1]["train_batches"],
               "train_pairs_available": config["train_pairs"],
               "checkpoint_val_pairs": config["checkpoint_val_pairs"],
               "checkpoint_epoch": checkpoint["epoch"],
               "validation_pairs": len(rows),
               "inference_steps": args.inference_steps,
               "metric_policy": METRIC_POLICY, "normalization": "none",
               "model_validation": model_means,
               "bicubic_validation": bicubic_means,
               "beats_bicubic_on_both": all(model_means[key] > bicubic_means[key]
                                            for key in ("psnr_db", "ssim"))}
    (args.out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
