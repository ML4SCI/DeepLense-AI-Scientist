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
