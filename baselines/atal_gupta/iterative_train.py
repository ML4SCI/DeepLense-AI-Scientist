"""Train the earlier compact iterative autoencoder baseline on paired SR data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from iterative_model import SimpleUnet, alpha_schedule, diffuse
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dlens.data.sr_dataset import SRPairDataset


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _write_history(path: Path, history: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(record) + "\n" for record in history))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-train-batches", type=int, default=200)
    parser.add_argument("--checkpoint-val-pairs", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.base_width, args.timesteps,
            args.max_train_batches, args.checkpoint_val_pairs) < 1
            or args.learning_rate <= 0 or args.base_width % 8):
        parser.error("training sizes must be positive; base width must be divisible by 8")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    train_set = SRPairDataset(args.splits_root / "train", augment=False)
    val_set = SRPairDataset(args.splits_root / "val", augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    model = SimpleUnet(args.base_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    alpha_bar = alpha_schedule(args.timesteps, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {"model": "Iterative Auto Encoder", "epochs": args.epochs,
              "train_pairs": len(train_set), "validation_pairs": len(val_set),
              "batch_size": args.batch_size, "base_width": args.base_width,
              "timesteps": args.timesteps,
              "max_train_batches": args.max_train_batches,
              "checkpoint_val_pairs": args.checkpoint_val_pairs,
              "learning_rate": args.learning_rate, "seed": args.seed,
              "normalization": "none"}
    history_path = args.out_dir / "history.jsonl"
    best_path = args.out_dir / "best_model.pt"
    last_path = args.out_dir / "last.pt"
    history = []
    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        if saved["config"] != config:
            parser.error("resume checkpoint uses different data or training settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"] + 1
        best_val = saved["best_val_l1"]
        history = saved.get("history")
        if history is None:
            lines = history_path.read_text().splitlines()[:saved["epoch"]]
            history = [json.loads(line) for line in lines]
        if [record["epoch"] for record in history] != list(range(1, start_epoch)):
            parser.error("history does not match the checkpoint's committed epochs")
        best_epoch = min(history, key=lambda record: record["val_l1"])["epoch"]
        if best_epoch == saved["epoch"]:
            _save_checkpoint(best_path, {key: saved[key] for key in ("model", "config", "epoch", "val_l1")})
        else:
            best = torch.load(best_path, map_location="cpu", weights_only=True)
            if best["epoch"] != best_epoch or best["config"] != config:
                parser.error("best checkpoint does not match the committed training history")
        _write_history(history_path, history)
        if "rng_state" in saved:
            torch.set_rng_state(saved["rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in saved:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
        print(f"Resuming at epoch {start_epoch}", flush=True)
    steps_per_epoch = min(len(train_loader), args.max_train_batches)
    print(f"Training plan: {args.epochs} epochs, {steps_per_epoch} steps/epoch", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_total = 0.0
        for batch_index, (lr, hr, _) in enumerate(train_loader, 1):
            lr, hr = lr.to(device), hr.to(device)
            lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bilinear",
                                  align_corners=False)
            residual = hr - lr_up
            time = torch.randint(args.timesteps, (len(lr),), device=device)
            noisy = diffuse(residual, time, alpha_bar, torch.randn_like(residual))
            predicted = model(torch.cat((lr_up, noisy), dim=1), time)
            loss = F.l1_loss(predicted, residual)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach())
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == steps_per_epoch:
                print(f"Epoch {epoch}/{args.epochs} | step {batch_index}/{steps_per_epoch} | "
                      f"loss {float(loss.detach()):.6f}", flush=True)
            if batch_index >= steps_per_epoch:
                break

        model.eval()
        val_total = 0.0
        val_count = 0
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        with torch.inference_mode():
            for lr, hr, _ in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bilinear",
                                      align_corners=False)
                residual = hr - lr_up
                time = torch.randint(args.timesteps, (len(lr),), device=device,
                                     generator=generator)
                noise = torch.randn(residual.shape, device=device, generator=generator)
                noisy = diffuse(residual, time, alpha_bar, noise)
                predicted = model(torch.cat((lr_up, noisy), dim=1), time)
                take = min(len(lr), args.checkpoint_val_pairs - val_count)
                val_total += float(F.l1_loss(predicted[:take], residual[:take],
                                             reduction="sum"))
                val_count += take
                if val_count >= args.checkpoint_val_pairs:
                    break
        val_loss = val_total / (val_count * hr[0].numel())
        record = {"epoch": epoch, "train_l1": train_total / steps_per_epoch,
                  "val_l1": val_loss, "train_batches": steps_per_epoch}
        history.append(record)
        checkpoint = {"model": model.state_dict(), "config": config,
                      "epoch": epoch, "val_l1": val_loss}
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
        state = {**checkpoint, "optimizer": optimizer.state_dict(),
                 "best_val_l1": best_val, "history": history,
                 "rng_state": torch.get_rng_state()}
        if device.type == "cuda":
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        # The committed checkpoint can restore both the history and a newly
        # selected best model if the process stops before either export.
        _save_checkpoint(last_path, state)
        if best_epoch == epoch:
            _save_checkpoint(best_path, checkpoint)
        _write_history(history_path, history)
        print(json.dumps(record), flush=True)
    print(f"Baseline epochs completed: {len(history)}/{args.epochs}", flush=True)


if __name__ == "__main__":
    main()
