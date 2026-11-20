"""Compare uniform training budgets with progressive architecture search.

This is a small synthetic CPU experiment, not a scientific performance claim.
Strategies share candidate architectures and dataset splits. Each trial uses
the same training seed across strategies, while seeds vary between trials.
Total epoch budgets differ, and actual wall-clock timings are reported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


SEED = 0


def make_dataset(root: Path):
    """Create a reproducible, tiny two-class image dataset."""
    from dlens.schemas._downstream import DatasetRef

    rng = np.random.default_rng(SEED)
    refs = {}

    for split in ("train", "val"):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)

        for label, class_name in enumerate(("a", "b")):
            for index in range(6):
                base = (
                    np.zeros((16, 16), dtype=np.float32)
                    if label == 0
                    else np.ones((16, 16), dtype=np.float32) * 3
                )
                image = base + rng.normal(0, 0.1, (16, 16))
                np.save(
                    directory / f"{class_name}_{index:04d}.npy",
                    image.astype(np.float32),
                )

        refs[split] = DatasetRef(
            root=str(directory),
            class_names=["a", "b"],
            image_shape=(16, 16),
            num_samples=12,
        )

    return refs


def make_candidates():
    """Return the same candidate architectures for every strategy."""
    from dlens.schemas._model_design import ArchitectureSpec

    candidates = []
    for name, family in (
        ("tiny_cnn", "cnn"),
        ("tiny_resnet", "resnet"),
    ):
        candidates.append(
            ArchitectureSpec(
                name=name,
                family=family,
                input_shape=(16, 16),
                channels=1,
                num_classes=2,
                depths=[1, 1],
                widths=[8, 16],
                rationale="Small synthetic benchmark candidate",
            )
        )
    return candidates


def make_config(epochs: int):
    from dlens.schemas._model_design import TrainingConfig

    return TrainingConfig(
        loss="cross_entropy",
        epochs=epochs,
        batch_size=4,
        learning_rate=1e-3,
    )


def run_uniform(
    *,
    strategy: str,
    epochs: int,
    candidates,
    refs,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Train every candidate independently for the same number of epochs."""
    from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = TorchTrainBackend(
        output_root=str(output_dir / "models"),
        device="cpu",
        seed=seed,
    )
    inferencer = TorchInferBackend(device="cpu")
    config = make_config(epochs)
    rows = []
    started_all = time.monotonic()

    for candidate in candidates:
        train_started = time.monotonic()
        trained = trainer.train(refs["train"], candidate, config)
        training_seconds = time.monotonic() - train_started

        validation_started = time.monotonic()
        inference = inferencer.infer(trained.weights_path, refs["val"])
        validation_seconds = time.monotonic() - validation_started

        rows.append(
            {
                "name": candidate.name,
                "requested_epochs": epochs,
                "actual_epochs": trained.epochs_run,
                "train_accuracy": trained.metrics.get("train_accuracy"),
                "val_accuracy": inference.accuracy,
                "training_seconds": round(training_seconds, 4),
                "validation_seconds": round(validation_seconds, 4),
            }
        )

    elapsed = time.monotonic() - started_all
    winner = max(
        rows,
        key=lambda row: (
            float("-inf")
            if row["val_accuracy"] is None
            else row["val_accuracy"],
            row["name"],
        ),
    )

    return {
        "strategy": strategy,
        "allocation": "uniform_from_scratch",
        "requested_total_epochs": epochs * len(candidates),
        "actual_total_epochs": sum(row["actual_epochs"] for row in rows),
        "total_training_seconds": round(
            sum(row["training_seconds"] for row in rows), 4
        ),
        "total_validation_seconds": round(
            sum(row["validation_seconds"] for row in rows), 4
        ),
        "elapsed_seconds": round(elapsed, 4),
        "winner": winner["name"],
        "winner_val_accuracy": winner["val_accuracy"],
        "candidates": rows,
    }
def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated trials while retaining the individual measurements."""
    if not trials:
        raise ValueError("At least one trial is required.")

    return {
        "trial_count": len(trials),
        "median_elapsed_seconds": median(
            trial["elapsed_seconds"] for trial in trials
        ),
        "median_training_seconds": median(
            trial["total_training_seconds"] for trial in trials
        ),
        "median_validation_seconds": median(
            trial["total_validation_seconds"] for trial in trials
        ),
        "median_actual_total_epochs": median(
            trial["actual_total_epochs"] for trial in trials
        ),
        "trials": trials,
    }



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/progressive_search")
    parser.add_argument("--initial-epochs", type=int, default=1)
    parser.add_argument("--additional-epochs", type=int, default=1)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of independent trials per strategy.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds used for benchmark trials.",
    )
    args = parser.parse_args()

    if args.initial_epochs < 1 or args.additional_epochs < 1:
        parser.error("epoch budgets must be at least 1")
    if not 0 < args.keep_fraction <= 1:
        parser.error("--keep-fraction must be in (0, 1]")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.trials > len(args.seeds):
        parser.error("--trials cannot exceed the number of supplied seeds")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    from dlens.agents._progressive_search import run_progressive_search
    from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refs = make_dataset(output_dir / "synthetic_data")
    candidates = make_candidates()
    initial_epochs = args.initial_epochs
    additional_epochs = args.additional_epochs

    # Warm up each architecture family before measuring trials.
    # Warm-up results are excluded from the reported benchmark.
    warmup_dir = output_dir / "warmup"
    warmup_trainer = TorchTrainBackend(
        output_root=str(warmup_dir / "models"),
        device="cpu",
        seed=SEED,
    )

    for candidate in candidates:
        warmup_trainer.train(
            refs["train"],
            candidate,
            make_config(1),
        )
    strategy_trials: dict[str, list[dict[str, Any]]] = {
        "uniform_short": [],
        "uniform_long": [],
        "progressive": [],
    }

    for trial_index in range(args.trials):
        seed = args.seeds[trial_index]
        trial_dir = output_dir / f"trial_{trial_index + 1:02d}_seed_{seed}"

        print(
            f"\nTrial {trial_index + 1}/{args.trials}",
            flush=True,
        )

        short_result = run_uniform(
            strategy="uniform_short",
            epochs=initial_epochs,
            candidates=candidates,
            refs=refs,
            output_dir=trial_dir / "uniform_short",
            seed=seed,
        )
        short_result["seed"] = seed
        short_result["trial"] = trial_index + 1
        strategy_trials["uniform_short"].append(short_result)

        long_result = run_uniform(
            strategy="uniform_long",
            epochs=initial_epochs + additional_epochs,
            candidates=candidates,
            refs=refs,
            output_dir=trial_dir / "uniform_long",
            seed=seed,
        )
        long_result["seed"] = seed
        long_result["trial"] = trial_index + 1
        strategy_trials["uniform_long"].append(long_result)

        progressive_dir = trial_dir / "progressive"
        progressive_trainer = TorchTrainBackend(
            output_root=str(progressive_dir / "models"),
            device="cpu",
            seed=seed,
        )
        progressive_inferencer = TorchInferBackend(device="cpu")

        started = time.monotonic()
        progressive_result = run_progressive_search(
            candidates=candidates,
            train_dataset=refs["train"],
            val_dataset=refs["val"],
            trainer=progressive_trainer,
            inferencer=progressive_inferencer,
            config=make_config(initial_epochs),
            initial_epochs=initial_epochs,
            additional_epochs=additional_epochs,
        keep_fraction=args.keep_fraction,
        )
        progressive_result["elapsed_seconds"] = round(
            time.monotonic() - started, 4
        )
        progressive_result["actual_total_epochs"] = sum(
            row["epochs_run"]
            for row in progressive_result["candidates"]
        )
        progressive_result["winner_val_accuracy"] = next(
            row["stages"][-1]["val_accuracy"]
            for row in progressive_result["candidates"]
            if row["name"] == progressive_result["winner"]
        )
        progressive_result["trial"] = trial_index + 1
        progressive_result["seed"] = seed
        strategy_trials["progressive"].append(progressive_result)

    summaries = {
        strategy: summarize_trials(trials)
        for strategy, trials in strategy_trials.items()
    }

    report = {
        "experiment": "uniform_vs_progressive_architecture_search",
        "seeds": args.seeds[:args.trials],
        "device": "cpu",
        "trial_count": args.trials,
        "warmup": {
            "performed": True,
            "included_in_timing_summaries": False,
        },
        "dataset": {
            "type": "synthetic",
            "image_shape": [16, 16],
            "classes": ["a", "b"],
            "train_samples": 12,
            "validation_samples": 12,
        },
        "candidate_names": [candidate.name for candidate in candidates],
        "budget_note": (
            "Uniform-short trains every candidate for initial_epochs. "
            "Uniform-long trains every candidate for initial_epochs + "
            "additional_epochs. Progressive trains all candidates for "
            "initial_epochs and resumes only survivors for additional_epochs. "
            "These strategies have different total epoch budgets. Each trial "
            "uses separate model-output directories. "
            "Each trial uses its assigned seed consistently across all three strategies. "
            "Seeds vary between trials, but this small synthetic experiment is not "
            "sufficient to establish general performance differences. Runtime is measured, "
            "not assumed to be proportional to epochs."
        ),
        "strategies": summaries,
    }


    result_path = output_dir / "comparison_results.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nResults saved to: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
