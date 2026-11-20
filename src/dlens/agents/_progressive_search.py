"""Utilities for progressive architecture-search allocation."""

from __future__ import annotations

import math
import time
from typing import Any

from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig
from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend


def select_survivors(
    scores: dict[str, float],
    *,
    keep_fraction: float = 0.5,
) -> list[str]:
    """Return the highest-scoring candidates in deterministic order.

    Candidates with equal scores are ordered by name.
    At least one candidate survives when scores is non-empty.
    """
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1].")

    if not scores:
        raise ValueError("At least one candidate score is required.")

    for name, score in scores.items():
        if not name.strip():
            raise ValueError("Candidate names must not be empty.")
        if not math.isfinite(score):
            raise ValueError(f"Score for {name!r} must be finite.")

    ranked = sorted(scores, key=lambda name: (-scores[name], name))
    survivor_count = max(1, math.ceil(len(ranked) * keep_fraction))

    return ranked[:survivor_count]


def run_progressive_search(
    candidates: list[ArchitectureSpec],
    train_dataset: DatasetRef,
    val_dataset: DatasetRef,
    trainer: TorchTrainBackend,
    inferencer: TorchInferBackend,
    config: TrainingConfig,
    *,
    initial_epochs: int = 2,
    additional_epochs: int = 2,
    keep_fraction: float = 0.5,
) -> dict[str, Any]:
    """Run a two-stage successive-halving experiment with checkpoint resumes."""
    if not candidates:
        raise ValueError("At least one candidate is required.")

    if initial_epochs < 1 or additional_epochs < 1:
        raise ValueError("Epoch budgets must be at least 1.")

    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1].")

    names = [candidate.name for candidate in candidates]

    if any(not name.strip() for name in names):
        raise ValueError("Candidate names must not be empty.")

    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique.")


    stage_config = config.model_copy(update={"epochs": initial_epochs})
    results: dict[str, dict[str, Any]] = {}
    # Stage 1: train and evaluate every candidate.
    for candidate in candidates:
        started = time.monotonic()
        trained = trainer.train(train_dataset, candidate, stage_config)
        training_seconds = time.monotonic() - started

        validation_started = time.monotonic()
        inference = inferencer.infer(trained.weights_path, val_dataset)
        validation_seconds = time.monotonic() - validation_started

        if inference.accuracy is None or not math.isfinite(inference.accuracy):
            raise ValueError(
                f"Candidate {candidate.name!r} has no finite validation accuracy."
            )

        results[candidate.name] = {
            "name": candidate.name,
            "status": "active",
            "checkpoint_path": trained.weights_path,
            "epochs_run": trained.epochs_run,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "stages": [
                {
                    "stage": 1,
                    "epochs_run": trained.epochs_run,
                    "train_accuracy": trained.metrics.get("train_accuracy"),
                    "val_accuracy": inference.accuracy,
                    "training_seconds": training_seconds,
                    "validation_seconds": validation_seconds,
                }
            ],
        }

    # Select survivors using actual validation results, not the LLM judge.
    initial_scores = {
        name: result["stages"][-1]["val_accuracy"]
        for name, result in results.items()
    }
    survivors = set(
        select_survivors(initial_scores, keep_fraction=keep_fraction)
    )

    for name, result in results.items():
        if name not in survivors:
            result["status"] = "eliminated"

    # Stage 2: resume only the surviving candidates.
    for candidate in candidates:
        if candidate.name not in survivors:
            continue

        result = results[candidate.name]
        started = time.monotonic()
        trained = trainer.resume(
            train_dataset,
            candidate,
            stage_config,
            checkpoint_path=result["checkpoint_path"],
            additional_epochs=additional_epochs,
        )
        training_seconds = time.monotonic() - started

        validation_started = time.monotonic()
        inference = inferencer.infer(trained.weights_path, val_dataset)
        validation_seconds = time.monotonic() - validation_started

        if inference.accuracy is None or not math.isfinite(inference.accuracy):
            raise ValueError(
                f"Candidate {candidate.name!r} has no finite validation accuracy."
            )

        result["checkpoint_path"] = trained.weights_path
        result["epochs_run"] = trained.epochs_run
        result["training_seconds"] += training_seconds
        result["validation_seconds"] += validation_seconds
        result["stages"].append(
            {
                "stage": 2,
                "epochs_run": trained.epochs_run,
                "train_accuracy": trained.metrics.get("train_accuracy"),
                "val_accuracy": inference.accuracy,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
            }
        )

    final_scores = {
        name: result["stages"][-1]["val_accuracy"]
        for name, result in results.items()
        if result["status"] == "active"
    }
    winner = max(final_scores, key=lambda name: (final_scores[name], name))

    return {
        "strategy": "successive_halving",
        "initial_epochs": initial_epochs,
        "additional_epochs": additional_epochs,
        "keep_fraction": keep_fraction,
        "winner": winner,
        "candidates": list(results.values()),
        "total_training_seconds": sum(
            result["training_seconds"] for result in results.values()
        ),
        "total_validation_seconds": sum(
            result["validation_seconds"] for result in results.values()
        ),
    }
