# tools/_inference.py
"""Inference tool for the Infer Agent: real (numpy) backend + deterministic mock.

The real backend loads the nearest-centroid weights saved by the training backend
and scores a dataset; the mock returns a fixed, labeled prediction set so the agent
runs fully offline with no weights or data on disk.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic_ai import Agent, RunContext

from dlens.schemas._downstream import DatasetRef, InferResult
from dlens.tools._training import load_dataset


@runtime_checkable
class InferBackend(Protocol):
    name: str

    def infer(self, weights_path: str, dataset: DatasetRef) -> InferResult:
        ...


class CentroidInferBackend:
    """Real nearest-centroid inference (numpy), matching CentroidTrainBackend."""

    name = "centroid"

    def infer(self, weights_path: str, dataset: DatasetRef) -> InferResult:
        data = np.load(weights_path)
        centroids, mu, sd = data["centroids"], data["mean"], data["std"]
        x, y = load_dataset(dataset)
        xn = (x - mu) / sd
        dists = np.linalg.norm(xn[:, None, :] - centroids[None, :, :], axis=2)
        preds = dists.argmin(axis=1)
        # softmax over negative distances -> pseudo-probabilities
        logits = -dists
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return InferResult(
            run_id=str(uuid.uuid4())[:8],
            num_samples=int(len(y)),
            num_classes=int(centroids.shape[0]),
            predictions=preds.astype(int).tolist(),
            probabilities=probs.tolist(),
            true_labels=y.astype(int).tolist(),
            accuracy=float((preds == y).mean()),
            backend=self.name,
        )


class MockInferBackend:
    """Deterministic synthetic predictions (no weights/data) for offline agent tests."""

    name = "mock"

    def infer(self, weights_path: str, dataset: DatasetRef) -> InferResult:
        k = dataset.num_classes
        n = dataset.num_samples or (k * 4)
        true = [i % k for i in range(n)]
        # mostly-correct deterministic predictions (every 4th sample wrong)
        preds = [(t if i % 4 else (t + 1) % k) for i, t in enumerate(true)]
        probs = [[(0.7 if c == p else 0.3 / max(k - 1, 1)) for c in range(k)] for p in preds]
        return InferResult(
            run_id="mockinfer",
            num_samples=n,
            num_classes=k,
            predictions=preds,
            probabilities=probs,
            true_labels=true,
            accuracy=float(sum(int(p == t) for p, t in zip(preds, true)) / n),
            backend=self.name,
        )


def get_infer_backend(name: str = "auto") -> InferBackend:
    name = (name or "auto").lower()
    if name == "mock":
        return MockInferBackend()
    if name in ("auto", "centroid"):
        return CentroidInferBackend()
    if name == "torch":
        from dlens.tools._torch_backends import TorchInferBackend  # lazy: needs torch

        return TorchInferBackend()
    raise ValueError(f"Unknown infer backend: {name!r} (expected auto | centroid | torch | mock)")


@dataclass
class InferDeps:
    backend: InferBackend
    last_result: InferResult | None = None


def register_infer_tool(agent: Agent) -> None:
    """Register the ``run_inference`` tool onto ``agent``."""

    @agent.tool
    def run_inference(ctx: RunContext[InferDeps], weights_path: str, dataset: DatasetRef) -> dict:
        """Score ``dataset`` with the trained model at ``weights_path``.

        Returns a compact summary (run id, sample/class counts, accuracy). The full
        prediction arrays are kept on deps (``InferDeps.last_result``) — they are too
        large to round-trip through the model context.
        """
        result = ctx.deps.backend.infer(weights_path, dataset)
        ctx.deps.last_result = result
        return {
            "run_id": result.run_id,
            "num_samples": result.num_samples,
            "num_classes": result.num_classes,
            "accuracy": result.accuracy,
            "backend": result.backend,
        }
