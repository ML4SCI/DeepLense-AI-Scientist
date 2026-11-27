"""Tests for defensive training provenance capture."""

from __future__ import annotations

import numpy as np

from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig
from dlens.tools import _provenance
from dlens.tools._torch_backends import _device
from dlens.tools._training import CentroidTrainBackend, MockTrainBackend


def _inputs() -> tuple[ArchitectureSpec, TrainingConfig]:
    return (
        ArchitectureSpec(
            name="resnet18",
            family=ArchFamily.RESNET,
            input_shape=(16, 16),
            num_classes=2,
        ),
        TrainingConfig(loss="cross_entropy", epochs=2),
    )


def test_provenance_records_git_and_configuration(monkeypatch):
    arch, config = _inputs()

    class Completed:
        stdout = "a" * 40 + "\n"

    monkeypatch.setattr(_provenance.subprocess, "run", lambda *args, **kwargs: Completed())
    record = _provenance.make_provenance(
        backend="torch",
        device="cpu",
        seed=7,
        architecture=arch,
        training_config=config,
    )

    assert record.git_commit == "a" * 40
    assert record.device == "cpu"
    assert record.seed == 7
    assert record.architecture == arch
    assert record.training_config == config


def test_provenance_survives_missing_git(monkeypatch):
    arch, config = _inputs()

    def missing_git(*args, **kwargs):
        raise _provenance.subprocess.SubprocessError("not a checkout")

    monkeypatch.setattr(_provenance.subprocess, "run", missing_git)
    record = _provenance.make_provenance(
        backend="mock",
        device="cpu",
        seed=None,
        architecture=arch,
        training_config=config,
    )

    assert record.git_commit is None
    assert record.python_version


def test_provenance_rejects_non_commit_output(monkeypatch):
    arch, config = _inputs()

    class Completed:
        stdout = "TOKEN=do-not-record\n"

    monkeypatch.setattr(_provenance.subprocess, "run", lambda *args, **kwargs: Completed())
    record = _provenance.make_provenance(
        backend="centroid",
        device="cpu",
        seed=None,
        architecture=arch,
        training_config=config,
    )

    assert record.git_commit is None
    assert "TOKEN" not in record.model_dump_json()


def test_automatic_device_selection_uses_mps_when_available():
    class MPS:
        @staticmethod
        def is_available():
            return True

    class Backends:
        mps = MPS()

    class Torch:
        backends = Backends()

        @staticmethod
        def device(name):
            return name

    assert _device(Torch) == "mps"


def test_automatic_device_selection_falls_back_to_cpu():
    class MPS:
        @staticmethod
        def is_available():
            return False

    class Backends:
        mps = MPS()

    class Torch:
        backends = Backends()

        @staticmethod
        def device(name):
            return name

    assert _device(Torch) == "cpu"


def test_mock_backend_records_cpu_provenance():
    arch, config = _inputs()
    dataset = DatasetRef(root="/unused", class_names=["a", "b"], image_shape=(16, 16))

    result = MockTrainBackend().train(dataset, arch, config)

    assert result.provenance is not None
    assert result.provenance.backend == "mock"
    assert result.provenance.device == "cpu"
    assert result.provenance.seed is None


def test_centroid_backend_records_cpu_provenance(tmp_path):
    arch, config = _inputs()
    dataset = DatasetRef(root=str(tmp_path), class_names=["a", "b"], image_shape=(2, 2))
    for label, name in enumerate(dataset.class_names):
        np.save(tmp_path / f"{name}_0.npy", np.full((2, 2), label, dtype=np.float32))

    result = CentroidTrainBackend(output_root=str(tmp_path / "models")).train(
        dataset, arch, config
    )

    assert result.provenance is not None
    assert result.provenance.backend == "centroid"
    assert result.provenance.device == "cpu"
    assert result.provenance.seed is None
