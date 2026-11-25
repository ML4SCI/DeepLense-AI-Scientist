"""Progressive search and checkpoint resumption tests (small CPU workloads)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from dlens.agents._progressive_search import (
    ProgressiveArchitectureSearch,
    select_survivors,
)
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig
from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

torch = pytest.importorskip("torch")


def _dataset(root: Path) -> DatasetRef:
    rng = np.random.default_rng(7)
    root.mkdir()
    for label, name in enumerate(("a", "b")):
        for index in range(4):
            image = np.full((16, 16), label * 2.0, dtype=np.float32)
            image += rng.normal(0, 0.05, image.shape).astype(np.float32)
            np.save(root / f"{name}_{index}.npy", image)
    return DatasetRef(root=str(root), class_names=["a", "b"], image_shape=(16, 16), num_samples=8)


def _spec(name: str, width: int = 8) -> ArchitectureSpec:
    return ArchitectureSpec(
        name=name, family=ArchFamily.CNN, input_shape=(16, 16), channels=1,
        num_classes=2, depths=[1, 1], widths=[width, width * 2],
    )


def test_select_survivors_is_deterministic_and_validates_scores():
    assert select_survivors({"b": 0.8, "a": 0.8, "c": 0.7}, 2) == ["a", "b"]
    with pytest.raises(ValueError, match="count"):
        select_survivors({"a": 1.0}, 0)
    with pytest.raises(ValueError, match="finite"):
        select_survivors({"a": float("nan")}, 1)


def test_checkpoint_resume_and_compatibility(tmp_path: Path):
    data = _dataset(tmp_path / "data")
    backend = TorchTrainBackend(output_root=str(tmp_path / "models"), device="cpu", seed=11)
    spec = _spec("small")
    config = TrainingConfig(loss="cross_entropy", epochs=1, batch_size=4, augment=False)
    first = backend.train(data, spec, config)
    resumed = backend.train(data, spec, config, resume_from=first.weights_path)
    assert Path(resumed.weights_path).is_file()
    assert resumed.epochs_run == 1

    incompatible = TorchTrainBackend(
        output_root=str(tmp_path / "other"), device="cpu", seed=12
    )
    with pytest.raises(ValueError, match="seed"):
        incompatible.train(data, spec, config, resume_from=first.weights_path)
    with pytest.raises(ValueError, match="architecture"):
        backend.train(data, _spec("different"), config, resume_from=first.weights_path)
    different_data = _dataset(tmp_path / "other-data")
    with pytest.raises(ValueError, match="dataset_root"):
        backend.train(different_data, spec, config, resume_from=first.weights_path)


def test_progressive_search_trains_only_survivors_in_stage_two(tmp_path: Path):
    data = _dataset(tmp_path / "data")
    backend = TorchTrainBackend(output_root=str(tmp_path / "models"), device="cpu", seed=3)
    search = ProgressiveArchitectureSearch(
        backend, TorchInferBackend(device="cpu"), initial_epochs=1,
        additional_epochs=1, survivor_count=1,
    )
    result = asyncio.run(search.search(
        [_spec("candidate_a"), _spec("candidate_b", width=16)],
        train_ref=data, val_ref=data,
        config=TrainingConfig(loss="cross_entropy", epochs=2, batch_size=4),
    ))
    assert len(result.survivors) == len(result.candidates) == 1
    assert result.candidates[0].architecture.name == result.survivors[0]
    assert result.candidates[0].stage1_accuracy >= 0
    assert result.candidates[0].final_accuracy >= 0
    assert result.timings["total_seconds"] >= 0


@pytest.mark.parametrize("kwargs", [
    {"initial_epochs": 0}, {"additional_epochs": 0}, {"survivor_count": 0},
])
def test_invalid_search_configuration_rejected(kwargs):
    class Train:
        def train(self, *args, **kwargs):
            raise AssertionError("should not train")

    class Infer:
        def infer(self, *args, **kwargs):
            raise AssertionError("should not infer")

    with pytest.raises(ValueError):
        ProgressiveArchitectureSearch(Train(), Infer(), **kwargs)


def test_search_rejects_empty_duplicate_and_over_budget_candidates(tmp_path: Path):
    data = _dataset(tmp_path / "data")
    search = ProgressiveArchitectureSearch(
        TorchTrainBackend(output_root=str(tmp_path / "models"), device="cpu"),
        TorchInferBackend(device="cpu"), initial_epochs=2, additional_epochs=2,
    )
    cfg = TrainingConfig(loss="cross_entropy", epochs=3)
    with pytest.raises(ValueError, match="at least one"):
        asyncio.run(search.search([], train_ref=data, val_ref=data, config=cfg))
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(search.search([_spec("same"), _spec("same")], train_ref=data,
                                  val_ref=data, config=cfg))
    with pytest.raises(ValueError, match="cannot exceed"):
        asyncio.run(search.search([_spec("one")], train_ref=data, val_ref=data, config=cfg))
