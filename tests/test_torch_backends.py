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


def test_train_accuracy_matches_saved_checkpoint_eval_mode(tmp_path: Path):
    # High dropout + augmentation diverges between training-mode step logits and
    # eval-mode checkpoint inference; train_accuracy must match the saved checkpoint.
    ds = _tiny_dataset(tmp_path, n_per=16)
    arch = ArchitectureSpec(
        name="resnet18", family=ArchFamily.RESNET, input_shape=(16, 16), channels=1, num_classes=2
    )
    cfg = TrainingConfig(
        loss="cross_entropy", epochs=3, batch_size=8, learning_rate=1e-2,
        dropout=0.9, augment=True, early_stop_patience=0,
    )

    tr = TorchTrainBackend(output_root=str(tmp_path / "m"), device="cpu", seed=0).train(ds, arch, cfg)
    ir = TorchInferBackend(device="cpu").infer(tr.weights_path, ds)
    assert tr.metrics["train_accuracy"] == pytest.approx(round(ir.accuracy, 4))


def test_train_accuracy_reflects_restored_checkpoint_with_early_stopping(tmp_path: Path):
    from dlens.tools._torch_backends import _load_images, build_model

    ds = _tiny_dataset(tmp_path, n_per=20)
    arch = ArchitectureSpec(
        name="resnet18", family=ArchFamily.RESNET, input_shape=(16, 16), channels=1, num_classes=2
    )
    cfg = TrainingConfig(
        loss="cross_entropy", epochs=4, batch_size=8, learning_rate=1e-2,
        dropout=0.8, augment=True, early_stop_patience=1,
    )

    tr = TorchTrainBackend(output_root=str(tmp_path / "m"), device="cpu", seed=0).train(ds, arch, cfg)

    ckpt = torch.load(tr.weights_path, map_location="cpu", weights_only=False)
    model = build_model(arch, dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x, y = _load_images(ds)
    x = (x - ckpt["mean"]) / ckpt["std"]
    idx = np.random.default_rng(0).permutation(len(y))
    keep = idx[max(1, len(y) // 10) :]
    with torch.no_grad():
        preds = model(torch.from_numpy(x[keep])).argmax(1).numpy()
    expected_acc = round(float((preds == y[keep]).mean()), 4)
    assert tr.metrics["train_accuracy"] == pytest.approx(expected_acc)


