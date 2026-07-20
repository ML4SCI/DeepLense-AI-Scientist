# tools/_training.py
"""Training tool for the Train Agent: real (numpy) backend + deterministic mock.

The real backend is a dependency-free nearest-centroid classifier (standardized
features) — a genuine, fast, deterministic baseline that needs no deep-learning
stack. The proposed architecture/config are recorded on the result; swapping in a
heavier architecture-specific trainer is left as future work. The mock backend
returns fixed metrics for fully-offline agent tests.
"""

from __future__ import annotations

import glob
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic_ai import Agent, RunContext

from dlens.schemas._downstream import DatasetRef, TrainResult
from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig


def load_dataset(ref: DatasetRef) -> tuple[np.ndarray, np.ndarray]:
    """Load ``<class>_*.npy`` arrays under ``ref.root`` into (X[N, D], y[N])."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label, name in enumerate(ref.class_names):
        for fp in sorted(glob.glob(os.path.join(ref.root, f"{name}_*.npy"))):
            xs.append(np.load(fp).astype(np.float64).ravel())
            ys.append(label)
    if not xs:
        raise RuntimeError(f"No .npy samples found under {ref.root!r} for {ref.class_names}.")
    return np.asarray(xs), np.asarray(ys)


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd, mu, sd


@runtime_checkable
class TrainBackend(Protocol):
    name: str

    def train(
        self, dataset: DatasetRef, architecture: ArchitectureSpec, config: TrainingConfig
    ) -> TrainResult:
        ...


class CentroidTrainBackend:
    """Real, dependency-free nearest-centroid classifier (numpy)."""

    name = "centroid"

    def __init__(self, output_root: str = "models") -> None:
        self._output_root = output_root

    def train(
        self, dataset: DatasetRef, architecture: ArchitectureSpec, config: TrainingConfig
    ) -> TrainResult:
        x, y = load_dataset(dataset)
        xn, mu, sd = _standardize(x)
        k = dataset.num_classes
        centroids = np.stack([xn[y == c].mean(axis=0) for c in range(k)])
        # nearest-centroid train accuracy
        dists = np.linalg.norm(xn[:, None, :] - centroids[None, :, :], axis=2)
        acc = float((dists.argmin(axis=1) == y).mean())

        run_id = str(uuid.uuid4())[:8]
        out_dir = Path(self._output_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = str(out_dir / "centroids.npz")
        np.savez(weights_path, centroids=centroids, mean=mu, std=sd)

        return TrainResult(
            run_id=run_id,
            weights_path=weights_path,
            backend=self.name,
            num_classes=k,
            metrics={"train_accuracy": acc},
            epochs_run=config.epochs,
        )


class MockTrainBackend:
    """Deterministic synthetic training (no data, no disk) for offline agent tests."""

    name = "mock"

    def train(
        self, dataset: DatasetRef, architecture: ArchitectureSpec, config: TrainingConfig
    ) -> TrainResult:
        return TrainResult(
            run_id="mocktrain",
            weights_path="<mock>/weights.npz",
            backend=self.name,
            num_classes=dataset.num_classes,
            metrics={"train_accuracy": 0.9, "val_accuracy": 0.85},
            epochs_run=config.epochs,
        )


def get_train_backend(name: str = "auto", *, output_root: str = "models") -> TrainBackend:
    name = (name or "auto").lower()
    if name == "mock":
        return MockTrainBackend()
    if name in ("auto", "centroid"):
        return CentroidTrainBackend(output_root=output_root)
    if name == "torch":
        from dlens.tools._torch_backends import TorchTrainBackend  # lazy: needs torch

        return TorchTrainBackend(output_root=output_root)
    raise ValueError(f"Unknown train backend: {name!r} (expected auto | centroid | torch | mock)")


@dataclass
class TrainDeps:
    backend: TrainBackend
    last_result: TrainResult | None = None


def register_train_tool(agent: Agent) -> None:
    """Register the ``train_model`` tool onto ``agent``."""

    @agent.tool
    def train_model(
        ctx: RunContext[TrainDeps],
        dataset: DatasetRef,
        architecture: ArchitectureSpec,
        training_config: TrainingConfig,
    ) -> dict:
        """Train a classification model on the dataset and return run id, weights path, metrics."""
        result = ctx.deps.backend.train(dataset, architecture, training_config)
        ctx.deps.last_result = result
        return result.model_dump(mode="json")
