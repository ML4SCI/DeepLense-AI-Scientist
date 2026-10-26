"""Compare uniform and progressive training on a reproducible synthetic CPU task.

Run with: PYTHONPATH=src python scripts/benchmark_progressive_search.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from dlens.agents._progressive_search import ProgressiveArchitectureSearch
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig
from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend


class CountingTrainBackend(TorchTrainBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_requested = 0
        self.resumes = 0

    def train(self, dataset, architecture, config, *, resume_from=None):
        self.epochs_requested += config.epochs
        self.resumes += int(resume_from is not None)
        return super().train(dataset, architecture, config, resume_from=resume_from)


def make_dataset(root: Path, seed: int) -> DatasetRef:
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        directory = root / split
        directory.mkdir(parents=True)
        for label, name in enumerate(("smooth", "ring")):
            for index in range(8):
                image = rng.normal(0, 0.08, (16, 16)).astype(np.float32)
                if label == 0:
                    image[5:11, 5:11] += 1.0
                else:
                    image[3:13, 3:13] += 1.0
                    image[6:10, 6:10] -= 1.0
                np.save(directory / f"{name}_{index:03d}.npy", image)
    return DatasetRef(root=str(root / "train"), class_names=["smooth", "ring"],
                      image_shape=(16, 16), num_samples=16)


def candidates() -> list[ArchitectureSpec]:
    return [
        ArchitectureSpec(name="cnn_narrow", family=ArchFamily.CNN,
                         input_shape=(16, 16), channels=1, num_classes=2,
                         depths=[1, 1], widths=[8, 16]),
        ArchitectureSpec(name="cnn_wide", family=ArchFamily.CNN,
                         input_shape=(16, 16), channels=1, num_classes=2,
                         depths=[1, 1], widths=[16, 32]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--initial-epochs", type=int, default=1)
    parser.add_argument("--additional-epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.initial_epochs < 1 or args.additional_epochs < 1:
        parser.error("epoch counts must be positive")

    with tempfile.TemporaryDirectory(prefix="dlens-progressive-benchmark-") as temp:
        root = Path(temp)
        train_ref = make_dataset(root / "data", args.seed)
        val_ref = train_ref.model_copy(update={"root": str(root / "data" / "val")})
        cfg = TrainingConfig(
            loss="cross_entropy", epochs=args.initial_epochs + args.additional_epochs,
            batch_size=4, learning_rate=1e-3, augment=False, lr_scheduler="none",
        )
        specs = candidates()
        uniform_backend = CountingTrainBackend(
            output_root=str(root / "uniform"), device=args.device, seed=args.seed
        )
        uniform_started = time.perf_counter()
        uniform_results = []
        for spec in specs:
            trained = uniform_backend.train(train_ref, spec, cfg)
            scored = TorchInferBackend(device=args.device).infer(trained.weights_path, val_ref)
            uniform_results.append({"name": spec.name, "val_accuracy": scored.accuracy})
        uniform_seconds = time.perf_counter() - uniform_started

        progressive_backend = CountingTrainBackend(
            output_root=str(root / "progressive"), device=args.device, seed=args.seed
        )
        progressive_started = time.perf_counter()
        progressive_result = asyncio.run(ProgressiveArchitectureSearch(
            progressive_backend, TorchInferBackend(device=args.device),
            initial_epochs=args.initial_epochs, additional_epochs=args.additional_epochs,
            survivor_count=1,
        ).search(specs, train_ref=train_ref, val_ref=val_ref, config=cfg))
        progressive_seconds = time.perf_counter() - progressive_started

        print(json.dumps({
            "seed": args.seed,
            "dataset": "synthetic 16x16 two-class images (16 train, 16 validation)",
            "uniform": {
                "seconds": round(uniform_seconds, 3),
                "epochs_requested": uniform_backend.epochs_requested,
                "results": uniform_results,
            },
            "progressive": {
                "seconds": round(progressive_seconds, 3),
                "epochs_requested": progressive_backend.epochs_requested,
                "resumed_candidates": progressive_backend.resumes,
                "survivors": progressive_result.survivors,
                "results": [row.model_dump(mode="json") for row in progressive_result.candidates],
            },
        }, indent=2))


if __name__ == "__main__":
    main()
