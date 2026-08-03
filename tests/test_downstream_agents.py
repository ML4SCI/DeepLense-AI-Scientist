"""Train / Infer / Analysis agents + backends, fully offline.

Real backends are exercised on a tiny separable numpy dataset; the agent flows use
scripted models + mock backends; the typed contracts are checked by assembling an
ExperimentRun from the stage outputs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from dlens.agents._analysis import AnalysisAgent, AnalysisReport
from dlens.agents._inference import InferAgent, InferReport
from dlens.agents._scripted_downstream import (
    make_scripted_analysis_model,
    make_scripted_infer_model,
    make_scripted_train_model,
)
from dlens.agents._train import TrainAgent, TrainReport
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._experiment import ExperimentRun, ExperimentState
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig
from dlens.tools._analysis import compute_analysis
from dlens.tools._inference import CentroidInferBackend, MockInferBackend, get_infer_backend
from dlens.tools._training import CentroidTrainBackend, MockTrainBackend, get_train_backend


def _make_dataset(tmp_path: Path, n_per: int = 6, shape: tuple[int, int] = (4, 4)) -> DatasetRef:
    """Two well-separated classes (~0 vs ~1) so a centroid classifier is ~perfect."""
    rng = np.random.default_rng(0)
    classes = ["a", "b"]
    for label, c in enumerate(classes):
        for i in range(n_per):
            base = np.zeros(shape) if label == 0 else np.ones(shape)
            np.save(tmp_path / f"{c}_{i:04d}.npy", base + rng.normal(0, 0.05, size=shape))
    return DatasetRef(
        root=str(tmp_path), class_names=classes, image_shape=shape, num_samples=n_per * len(classes)
    )


def _arch() -> ArchitectureSpec:
    return ArchitectureSpec(
        name="baseline", family=ArchFamily.RESNET, input_shape=(4, 4), channels=1, num_classes=2
    )


def _cfg() -> TrainingConfig:
    return TrainingConfig(loss="cross_entropy", epochs=5)


def _ref(**kw) -> DatasetRef:
    base = dict(root="/nonexistent", class_names=["a", "b"], image_shape=(4, 4), num_samples=8)
    base.update(kw)
    return DatasetRef(**base)


def test_real_backends_chain(tmp_path: Path):
    ds = _make_dataset(tmp_path)
    tr = CentroidTrainBackend(output_root=str(tmp_path / "models")).train(ds, _arch(), _cfg())
    assert Path(tr.weights_path).exists()
    assert tr.backend == "centroid"
    assert tr.metrics["train_accuracy"] >= 0.9

    ir = CentroidInferBackend().infer(tr.weights_path, ds)
    assert ir.num_samples == ds.num_samples
    assert ir.accuracy is not None and ir.accuracy >= 0.9
    assert ir.probabilities is not None and len(ir.probabilities) == ir.num_samples

    ar = compute_analysis(ir, ds.class_names)
    assert 0.0 <= ar.accuracy <= 1.0
    assert len(ar.confusion_matrix) == 2 and len(ar.confusion_matrix[0]) == 2
    assert set(ar.per_class) == {"a", "b"}
    assert ar.macro_auc is not None and 0.0 <= ar.macro_auc <= 1.0


def test_mock_backends_are_deterministic():
    ds = _ref()
    t1 = MockTrainBackend().train(ds, _arch(), _cfg())
    t2 = MockTrainBackend().train(ds, _arch(), _cfg())
    assert t1 == t2 and t1.backend == "mock" and t1.num_classes == 2
    i1 = MockInferBackend().infer("w", ds)
    i2 = MockInferBackend().infer("w", ds)
    assert i1 == i2 and i1.num_samples == 8 and i1.num_classes == 2
    assert i1.true_labels is not None and len(i1.predictions) == 8


def test_train_agent_offline(tmp_path: Path):
    ds, arch, cfg = _ref(), _arch(), _cfg()
    model = make_scripted_train_model(
        ds.model_dump(mode="json"), arch.model_dump(mode="json"), cfg.model_dump(mode="json")
    )
    agent = TrainAgent(model=model, backend=MockTrainBackend())
    res = asyncio.run(agent.train(ds, arch, cfg))
    assert isinstance(res.output, TrainReport)
    assert res.output.result.backend == "mock"
    assert agent.last_result is not None and agent.last_result.run_id == "mocktrain"


def test_infer_agent_offline():
    ds = _ref()
    model = make_scripted_infer_model("weights.npz", ds.model_dump(mode="json"))
    agent = InferAgent(model=model, backend=MockInferBackend())
    res = asyncio.run(agent.infer("weights.npz", ds))
    assert isinstance(res.output, InferReport)
    assert res.output.result.num_samples == 8
    assert agent.last_result is not None


def test_analysis_agent_offline():
    ir = MockInferBackend().infer("w", _ref())
    model = make_scripted_analysis_model(ir.model_dump(mode="json"))
    agent = AnalysisAgent(model=model)
    res = asyncio.run(agent.analyze(ir))
    assert isinstance(res.output, AnalysisReport)
    assert 0.0 <= res.output.result.accuracy <= 1.0
    assert len(res.output.result.confusion_matrix) == 2


def test_experiment_run_assembly(tmp_path: Path):
    ds, arch, cfg = _make_dataset(tmp_path), _arch(), _cfg()
    tr = CentroidTrainBackend(output_root=str(tmp_path / "m")).train(ds, arch, cfg)
    ir = CentroidInferBackend().infer(tr.weights_path, ds)
    ar = compute_analysis(ir, ds.class_names)
    run = ExperimentRun(
        iteration=0, architecture=arch, training_config=cfg,
        train_result=tr, infer_result=ir, analysis_result=ar,
    )
    state = ExperimentState(hypothesis="can a baseline separate a vs b?", runs=[run])
    assert state.runs[0].analysis_result.accuracy == ar.accuracy
    assert state.runs[0].train_result.weights_path == tr.weights_path


def test_backend_selectors():
    assert get_train_backend("mock").name == "mock"
    assert get_train_backend("auto").name == "centroid"
    assert get_infer_backend("mock").name == "mock"
    assert get_infer_backend("auto").name == "centroid"
    with pytest.raises(ValueError):
        get_train_backend("nope")
    with pytest.raises(ValueError):
        get_infer_backend("nope")
