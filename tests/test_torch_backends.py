"""Real torch backends on a tiny dataset (skipped if torch isn't installed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlens.schemas._downstream import DatasetRef  # noqa: E402
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig  # noqa: E402
from dlens.tools._inference import get_infer_backend  # noqa: E402
from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend  # noqa: E402
from dlens.tools._training import get_train_backend  # noqa: E402


def _tiny_dataset(tmp_path: Path, n_per: int = 8, shape=(16, 16)) -> DatasetRef:
    rng = np.random.default_rng(0)
    for label, c in enumerate(["a", "b"]):
        for i in range(n_per):
            base = np.zeros(shape) if label == 0 else np.ones(shape) * 3.0
            np.save(tmp_path / f"{c}_{i:04d}.npy", (base + rng.normal(0, 0.1, shape)).astype(np.float32))
    return DatasetRef(root=str(tmp_path), class_names=["a", "b"], image_shape=shape, num_samples=n_per * 2)


def test_torch_train_and_infer_roundtrip(tmp_path: Path):
    ds = _tiny_dataset(tmp_path)
    arch = ArchitectureSpec(
        name="resnet18", family=ArchFamily.RESNET, input_shape=(16, 16), channels=1, num_classes=2
    )
    cfg = TrainingConfig(loss="cross_entropy", epochs=2, batch_size=8, learning_rate=1e-3)

    tr = TorchTrainBackend(output_root=str(tmp_path / "m"), device="cpu").train(ds, arch, cfg)
    assert tr.backend == "torch" and tr.epochs_run == 2
    assert Path(tr.weights_path).exists()
    assert "train_accuracy" in tr.metrics

    ir = TorchInferBackend(device="cpu").infer(tr.weights_path, ds)
    assert ir.backend == "torch"
    assert ir.num_samples == 16 and ir.num_classes == 2
    assert ir.accuracy is not None and 0.0 <= ir.accuracy <= 1.0
    assert len(ir.probabilities) == 16 and len(ir.probabilities[0]) == 2


def test_factories_expose_torch():
    assert get_train_backend("torch").name == "torch"
    assert get_infer_backend("torch").name == "torch"

def test_train_seed_is_configurable_and_reproducible(tmp_path: Path):
    # Multi-seed replication needs the training seed to be a real knob: the same
    # seed must reproduce bit-for-bit, different seeds must actually diverge.
    ds = _tiny_dataset(tmp_path, n_per=16)
    arch = ArchitectureSpec(
        name="resnet18", family=ArchFamily.RESNET, input_shape=(16, 16), channels=1, num_classes=2
    )
    cfg = TrainingConfig(loss="cross_entropy", epochs=2, batch_size=8, learning_rate=1e-2)

    def _weights(seed: int, tag: str):
        be = TorchTrainBackend(output_root=str(tmp_path / tag), device="cpu", seed=seed)
        assert be.seed == seed
        tr = be.train(ds, arch, cfg)
        sd = torch.load(tr.weights_path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        return torch.cat([v.flatten() for v in sd.values() if hasattr(v, "flatten")])

    a1, a2, b = _weights(0, "s0a"), _weights(0, "s0b"), _weights(1, "s1")
    assert torch.equal(a1, a2), "same seed must reproduce identical weights"
    assert not torch.equal(a1, b), "different seeds must produce different weights"


def test_train_default_seed_is_zero(tmp_path: Path):
    # Default must stay 0 so previously published single-seed runs reproduce.
    assert TorchTrainBackend(output_root=str(tmp_path)).seed == 0


def test_early_stopping_metric_consistency_matches_restored_checkpoint(tmp_path: Path):
    """When early stopping restores best_state, metrics must describe the restored checkpoint."""
    from unittest.mock import patch

    ds = _tiny_dataset(tmp_path, n_per=16)
    arch = ArchitectureSpec(
        name="resnet18", family=ArchFamily.RESNET, input_shape=(16, 16), channels=1, num_classes=2
    )
    cfg = TrainingConfig(
        loss="cross_entropy", epochs=4, batch_size=8, learning_rate=1e-3, early_stop_patience=1
    )

    # Deterministic sequence:
    # Epoch 1: holdout=0.90 -> best_state saved, best_epoch=1
    # Epoch 2: holdout=0.40 -> degrades, since_best reaches patience (1) -> early stop at epoch 2
    # Checkpoint restored to epoch 1.
    backend = TorchTrainBackend(output_root=str(tmp_path / "es_m"), device="cpu", seed=0)
    with patch("dlens.tools._torch_backends._holdout_acc", side_effect=[0.90, 0.40]), \
         patch("dlens.tools._torch_backends.print"):
        tr = backend.train(ds, arch, cfg)

    assert tr.epochs_run == 2
    assert tr.metrics["best_epoch"] == 1.0
    assert tr.metrics["holdout_accuracy"] == 0.90
    assert "last_epoch_train_accuracy" in tr.metrics
    assert "last_epoch_loss" in tr.metrics
    assert "train_accuracy" in tr.metrics
    assert "final_loss" in tr.metrics

    # Checkpoint weights should match the restored epoch
    ckpt = torch.load(tr.weights_path, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt


def test_no_early_stopping_metrics_match_final_epoch(tmp_path: Path):
    """When early_stop_patience=0, metrics describe the final epoch without holdout keys."""
    ds = _tiny_dataset(tmp_path, n_per=8)
    arch = ArchitectureSpec(
        name="resnet18", family=ArchFamily.RESNET, input_shape=(16, 16), channels=1, num_classes=2
    )
    cfg = TrainingConfig(loss="cross_entropy", epochs=2, batch_size=8, learning_rate=1e-3)

    backend = TorchTrainBackend(output_root=str(tmp_path / "no_es"), device="cpu", seed=0)
    tr = backend.train(ds, arch, cfg)

    assert tr.epochs_run == 2
    assert "holdout_accuracy" not in tr.metrics
    assert "best_epoch" not in tr.metrics
    assert "train_accuracy" in tr.metrics
    assert "final_loss" in tr.metrics

