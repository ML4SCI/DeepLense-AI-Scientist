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


def test_resume_training_preserves_checkpoint_compatibility(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, n_per=8)
    arch = ArchitectureSpec(
        name="tiny_resnet",
        family=ArchFamily.RESNET,
        input_shape=(16, 16),
        channels=1,
        num_classes=2,
        depths=[1, 1],
        widths=[8, 16],
    )
    cfg = TrainingConfig(
        loss="cross_entropy",
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
    )
    backend = TorchTrainBackend(
        output_root=str(tmp_path / "resume_runs"),
        device="cpu",
        seed=0,
    )

    first = backend.train(ds, arch, cfg)
    assert first.epochs_run == 1
    first_checkpoint = torch.load(
        first.weights_path, map_location="cpu", weights_only=False
    )
    first_weights = first_checkpoint["state_dict"]

    # Continue for one additional epoch and preserve inference compatibility.
    resumed = backend.resume(
        ds,
        arch,
        cfg,
        checkpoint_path=first.weights_path,
        additional_epochs=1,
    )

    assert resumed.epochs_run == 2
    resumed_checkpoint = torch.load(
        resumed.weights_path, map_location="cpu", weights_only=False
    )

    weights_changed = False
    for key in first_weights:
        if not torch.equal(
            first_weights[key],
            resumed_checkpoint["state_dict"][key],
        ):
            weights_changed = True
            break

    assert weights_changed, "Model weights did not change during resumed training."
    assert Path(resumed.weights_path).exists()

    checkpoint = torch.load(
        resumed.weights_path, map_location="cpu", weights_only=False
    )
    assert "state_dict" in checkpoint
    assert "architecture" in checkpoint

    inference = TorchInferBackend(device="cpu").infer(resumed.weights_path, ds)
    assert inference.num_samples == ds.num_samples

def test_resume_rejects_learning_rate_mismatch(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, n_per=8)
    arch = ArchitectureSpec(
        name="tiny_resnet",
        family=ArchFamily.RESNET,
        input_shape=(16, 16),
        channels=1,
        num_classes=2,
        depths=[1, 1],
        widths=[8, 16],
    )
    cfg = TrainingConfig(
        loss="cross_entropy",
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
    )
    backend = TorchTrainBackend(
        output_root=str(tmp_path / "resume_runs"),
        device="cpu",
        seed=0,
    )

    first = backend.train(ds, arch, cfg)

    incompatible_cfg = TrainingConfig(
        loss="cross_entropy",
        epochs=1,
        batch_size=8,
        learning_rate=1e-2,
    )

    with pytest.raises(ValueError, match="Training configuration mismatch"):
        backend.resume(
            ds,
            arch,
            incompatible_cfg,
            checkpoint_path=first.weights_path,
            additional_epochs=1,
        )

def test_resume_rejects_seed_mismatch(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, n_per=8)
    arch = ArchitectureSpec(
        name="tiny_resnet",
        family=ArchFamily.RESNET,
        input_shape=(16, 16),
        channels=1,
        num_classes=2,
        depths=[1, 1],
        widths=[8, 16],
    )
    cfg = TrainingConfig(
        loss="cross_entropy",
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
    )

    original_backend = TorchTrainBackend(
        output_root=str(tmp_path / "original_runs"),
        device="cpu",
        seed=0,
    )
    first = original_backend.train(ds, arch, cfg)

    different_seed_backend = TorchTrainBackend(
        output_root=str(tmp_path / "different_seed_runs"),
        device="cpu",
        seed=42,
    )

    with pytest.raises(ValueError, match="Seed does not match"):
        different_seed_backend.resume(
            ds,
            arch,
            cfg,
            checkpoint_path=first.weights_path,
            additional_epochs=1,
        )

def test_resume_rejects_checkpoint_missing_optimizer_state(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, n_per=8)
    arch = ArchitectureSpec(
        name="tiny_resnet",
        family=ArchFamily.RESNET,
        input_shape=(16, 16),
        channels=1,
        num_classes=2,
        depths=[1, 1],
        widths=[8, 16],
    )
    cfg = TrainingConfig(
        loss="cross_entropy",
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
    )
    backend = TorchTrainBackend(
        output_root=str(tmp_path / "resume_runs"),
        device="cpu",
        seed=0,
    )

    first = backend.train(ds, arch, cfg)
    checkpoint = torch.load(
        first.weights_path, map_location="cpu", weights_only=False
    )
    del checkpoint["optimizer_state_dict"]

    broken_path = tmp_path / "broken_checkpoint.pt"
    torch.save(checkpoint, broken_path)

    with pytest.raises(
        ValueError,
        match="Checkpoint lacks the training state required for resuming",
    ):
        backend.resume(
            ds,
            arch,
            cfg,
            checkpoint_path=str(broken_path),
            additional_epochs=1,
        )

def test_resume_rejects_architecture_mismatch(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, n_per=8)
    arch = ArchitectureSpec(
        name="tiny_resnet",
        family=ArchFamily.RESNET,
        input_shape=(16, 16),
        channels=1,
        num_classes=2,
        depths=[1, 1],
        widths=[8, 16],
    )
    cfg = TrainingConfig(
        loss="cross_entropy",
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
    )
    backend = TorchTrainBackend(
        output_root=str(tmp_path / "resume_runs"),
        device="cpu",
        seed=0,
    )

    first = backend.train(ds, arch, cfg)

    incompatible_arch = ArchitectureSpec(
        name="different_resnet",
        family=ArchFamily.RESNET,
        input_shape=(16, 16),
        channels=1,
        num_classes=2,
        depths=[1, 1],
        widths=[8, 16],
    )

    with pytest.raises(ValueError, match="Architecture does not match"):
        backend.resume(
            ds,
            incompatible_arch,
            cfg,
            checkpoint_path=first.weights_path,
            additional_epochs=1,
        )

def test_resume_rejects_dataset_image_shape_mismatch(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, n_per=8)
    arch = ArchitectureSpec(
        name="tiny_resnet",
        family=ArchFamily.RESNET,
        input_shape=(16, 16),
        channels=1,
        num_classes=2,
        depths=[1, 1],
        widths=[8, 16],
    )
    cfg = TrainingConfig(
        loss="cross_entropy",
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
    )
    backend = TorchTrainBackend(
        output_root=str(tmp_path / "resume_runs"),
        device="cpu",
        seed=0,
    )

    first = backend.train(ds, arch, cfg)

    incompatible_ds = ds.model_copy(update={"image_shape": (32, 32)})

    with pytest.raises(
        ValueError, match="Dataset image shape does not match"
    ):
        backend.resume(
            incompatible_ds,
            arch,
            cfg,
            checkpoint_path=first.weights_path,
            additional_epochs=1,
        )

def test_resume_rejects_dataset_sample_count_mismatch(tmp_path: Path):
    ds = _tiny_dataset(tmp_path, n_per=8)
    arch = ArchitectureSpec(
        name="tiny_resnet",
        family=ArchFamily.RESNET,
        input_shape=(16, 16),
        channels=1,
        num_classes=2,
        depths=[1, 1],
        widths=[8, 16],
    )
    cfg = TrainingConfig(
        loss="cross_entropy",
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
    )
    backend = TorchTrainBackend(
        output_root=str(tmp_path / "resume_runs"),
        device="cpu",
        seed=0,
    )

    first = backend.train(ds, arch, cfg)
    incompatible_ds = ds.model_copy(
        update={"num_samples": ds.num_samples + 1}
    )

    with pytest.raises(
        ValueError, match="Dataset sample count does not match"
    ):
        backend.resume(
            incompatible_ds,
            arch,
            cfg,
            checkpoint_path=first.weights_path,
            additional_epochs=1,
        )
