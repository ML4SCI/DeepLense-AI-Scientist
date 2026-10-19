"""Two-stage architecture search with checkpoint-based Torch training resumption."""

from __future__ import annotations

import math
import time
from typing import Any

from pydantic import BaseModel, Field

from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig
from dlens.tools._inference import InferBackend
from dlens.tools._training import TrainBackend


class ProgressiveCandidateResult(BaseModel):
    architecture: ArchitectureSpec
    stage1_accuracy: float
    final_accuracy: float
    stage1_seconds: float
    stage2_seconds: float
    checkpoint: str


class ProgressiveSearchResult(BaseModel):
    survivors: list[str]
    candidates: list[ProgressiveCandidateResult]
    timings: dict[str, float] = Field(default_factory=dict)


def select_survivors(scores: dict[str, float], count: int) -> list[str]:
    """Select top candidates deterministically (score descending, name ascending)."""
    if count < 1:
        raise ValueError("survivor count must be at least 1")
    if not scores or any(
        not isinstance(v, (int, float)) or not math.isfinite(float(v))
        for v in scores.values()
    ):
        raise ValueError("candidate scores must be a non-empty mapping of finite numbers")
    return sorted(scores, key=lambda name: (-float(scores[name]), name))[:count]


class ProgressiveArchitectureSearch:
    """Train candidates briefly, prune by validation accuracy, then resume winners."""

    def __init__(
        self,
        train_backend: TrainBackend,
        infer_backend: InferBackend,
        *,
        initial_epochs: int = 2,
        additional_epochs: int = 4,
        survivor_count: int = 2,
    ) -> None:
        if initial_epochs < 1 or additional_epochs < 1 or survivor_count < 1:
            raise ValueError("epoch budgets and survivor_count must be positive")
        if not callable(getattr(train_backend, "train", None)):
            raise TypeError("train_backend must provide train()")
        if not callable(getattr(infer_backend, "infer", None)):
            raise TypeError("infer_backend must provide infer()")
        self.train_backend = train_backend
        self.infer_backend = infer_backend
        self.initial_epochs = initial_epochs
        self.additional_epochs = additional_epochs
        self.survivor_count = survivor_count

    async def search(
        self,
        candidates: list[ArchitectureSpec],
        *,
        train_ref: DatasetRef,
        val_ref: DatasetRef,
        config: TrainingConfig | None = None,
    ) -> ProgressiveSearchResult:
        if not candidates:
            raise ValueError("at least one candidate architecture is required")
        if len({c.name for c in candidates}) != len(candidates):
            raise ValueError("candidate names must be unique")
        if any(not candidate.is_buildable() for candidate in candidates):
            raise ValueError("all candidates must be buildable")
        base = config or TrainingConfig(loss="cross_entropy")
        if self.initial_epochs + self.additional_epochs > base.epochs:
            raise ValueError("total progressive epochs cannot exceed config.epochs")
        first_config = base.model_copy(update={"epochs": self.initial_epochs})
        continuation = base.model_copy(update={"epochs": self.additional_epochs})
        stage1: dict[str, tuple[ArchitectureSpec, Any, float, float]] = {}
        started = time.monotonic()
        for spec in candidates:
            t0 = time.monotonic()
            trained = self.train_backend.train(train_ref, spec, first_config)
            scored = self.infer_backend.infer(trained.weights_path, val_ref)
            if scored.accuracy is None:
                raise ValueError(f"validation accuracy unavailable for {spec.name}")
            stage1[spec.name] = (spec, trained, float(scored.accuracy), time.monotonic() - t0)
        survivor_names = select_survivors(
            {name: row[2] for name, row in stage1.items()}, self.survivor_count
        )
        results = []
        for name in survivor_names:
            spec, first, accuracy, stage1_s = stage1[name]
            t0 = time.monotonic()
            # Resume is an explicit backend capability; fail loudly instead of
            # silently restarting from scratch on unsupported backends.
            try:
                second = self.train_backend.train(
                    train_ref, spec, continuation, resume_from=first.weights_path
                )
            except TypeError as exc:
                if "resume_from" in str(exc):
                    raise TypeError(
                        "progressive search requires a checkpoint-resumable train backend"
                    ) from exc
                raise
            final = self.infer_backend.infer(second.weights_path, val_ref)
            if final.accuracy is None:
                raise ValueError(f"validation accuracy unavailable for {spec.name}")
            results.append(ProgressiveCandidateResult(
                architecture=spec, stage1_accuracy=accuracy,
                final_accuracy=float(final.accuracy), stage1_seconds=round(stage1_s, 3),
                stage2_seconds=round(time.monotonic() - t0, 3), checkpoint=second.weights_path,
            ))
        return ProgressiveSearchResult(
            survivors=survivor_names, candidates=results,
            timings={"total_seconds": round(time.monotonic() - started, 3)},
        )
