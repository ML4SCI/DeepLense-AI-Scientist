"""ReAct experiment loop — fully offline (scripted planner LLM + mock/tiny backends)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from dlens.agents._experiment_loop import ExperimentLoop, _apply
from dlens.agents._experiment_planner import ExperimentPlanner, ReActPlannerStrategy
from dlens.agents._scripted_planner import make_scripted_planner_model
from dlens.schemas._downstream import DatasetRef, TrainResult
from dlens.schemas._experiment import PlannerAction
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig
from dlens.tools._inference import MockInferBackend
from dlens.tools._training import MockTrainBackend


def _ref() -> DatasetRef:
    return DatasetRef(root="/nonexistent", class_names=["a", "b"], image_shape=(8, 8), num_samples=8)


def _arch(name: str = "resnet18") -> ArchitectureSpec:
    return ArchitectureSpec(
        name=name, family=ArchFamily.RESNET, input_shape=(8, 8), channels=1, num_classes=2
    )


def _cfg(**kw) -> TrainingConfig:
    base = dict(loss="cross_entropy", epochs=2, batch_size=8)
    base.update(kw)
    return TrainingConfig(**base)


def _scripted_strategy(decisions):
    return ReActPlannerStrategy(ExperimentPlanner(model=make_scripted_planner_model(decisions)))


def test_loop_applies_decisions_then_guard_stops():
    # Mock metrics are constant (train .9 / val .75, gap .15): the planner acts on
    # iterations 0 and 1; the no-improvement guard must stop the loop at iteration 2.
    decisions = [
        {"action": "train", "rationale": "Large train/val gap -> add dropout.",
         "updated_params": {"dropout": 0.5}},
        {"action": "train", "rationale": "Gap persists -> add augmentation.",
         "updated_params": {"augment": True, "weight_decay": 0.001}},
    ]
    loop = ExperimentLoop(
        strategy=_scripted_strategy(decisions),
        train_backend=MockTrainBackend(),
        infer_backend=MockInferBackend(),
    )
    state = asyncio.run(loop.run(
        hypothesis="does regularization close the gap?",
        train_ref=_ref(), val_ref=_ref(),
        architecture=_arch(), training_config=_cfg(),
        max_iterations=4, gap_threshold=0.06, patience=2,
    ))
    assert len(state.runs) == 3
    # decisions were applied to the carried-forward config
    assert state.runs[0].training_config.dropout == 0.0
    assert state.runs[1].training_config.dropout == 0.5
    assert state.runs[2].training_config.augment is True
    assert state.runs[2].training_config.weight_decay == 0.001
    # the last run was stopped by the code-side guard
    assert state.runs[-1].planner_decision.action == PlannerAction.REPORT
    assert "[loop guard]" in state.runs[-1].planner_decision.rationale
    # planner reasoning captured on the LLM-decided runs
    assert "dropout" in state.runs[0].planner_decision.rationale


def test_loop_gap_closed_guard_skips_planner():
    class SmallGapTrain:
        name = "fake"

        def train(self, dataset, architecture, config):
            return TrainResult(run_id="r", weights_path="w", backend="fake",
                               num_classes=2, metrics={"train_accuracy": 0.80}, epochs_run=1)

    class RaisingStrategy:
        name = "should-not-run"

        async def decide(self, state):  # pragma: no cover - must never be called
            raise AssertionError("planner must not be consulted once the gap is closed")

    loop = ExperimentLoop(
        strategy=RaisingStrategy(), train_backend=SmallGapTrain(), infer_backend=MockInferBackend()
    )
    state = asyncio.run(loop.run(
        hypothesis="h", train_ref=_ref(), val_ref=_ref(),
        architecture=_arch(), training_config=_cfg(),
        max_iterations=3, gap_threshold=0.06,
    ))
    # mock infer accuracy is 0.75 -> gap 0.05 <= 0.06 -> stop immediately
    assert len(state.runs) == 1
    assert "gap closed" in state.runs[0].planner_decision.rationale


def test_apply_validates_and_ignores_unknown_keys():
    arch, cfg = _apply(
        {"name": "resnet34", "dropout": 0.3, "bogus_key": 1}, _arch(), _cfg()
    )
    assert arch.name == "resnet34"
    assert cfg.dropout == 0.3
    with pytest.raises(Exception):
        _apply({"dropout": 1.5}, _arch(), _cfg())  # re-validation enforces bounds


def test_apply_accepts_grouped_params():
    # LLMs reasonably emit the grouped form ({"training": {...}}) — regression for
    # the run where nested params were silently ignored.
    arch, cfg = _apply(
        {"training": {"augment": True, "early_stop_patience": 3},
         "architecture": {"name": "resnet18"}},
        _arch("resnet34"), _cfg(),
    )
    assert arch.name == "resnet18"
    assert cfg.augment is True
    assert cfg.early_stop_patience == 3


def test_loop_with_real_torch_backend(tmp_path: Path):
    pytest.importorskip("torch")
    from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        d = tmp_path / split
        d.mkdir()
        for label, c in enumerate(["a", "b"]):
            for i in range(6):
                base = np.zeros((16, 16)) if label == 0 else np.ones((16, 16)) * 3
                np.save(d / f"{c}_{i:04d}.npy", (base + rng.normal(0, 0.1, (16, 16))).astype(np.float32))
    refs = {
        s: DatasetRef(root=str(tmp_path / s), class_names=["a", "b"], image_shape=(16, 16), num_samples=12)
        for s in ("train", "val")
    }
    decisions = [{"action": "train", "rationale": "add dropout + augment + early stop",
                  "updated_params": {"dropout": 0.2, "augment": True, "early_stop_patience": 2}}]
    loop = ExperimentLoop(
        strategy=_scripted_strategy(decisions),
        train_backend=TorchTrainBackend(output_root=str(tmp_path / "m"), device="cpu"),
        infer_backend=TorchInferBackend(device="cpu"),
    )
    state = asyncio.run(loop.run(
        hypothesis="tiny", train_ref=refs["train"], val_ref=refs["val"],
        architecture=_arch(), training_config=_cfg(epochs=2),
        max_iterations=2, gap_threshold=-1.0,  # force full iterations
    ))
    assert 1 <= len(state.runs) <= 2
    for run in state.runs:
        assert run.train_result is not None and run.analysis_result is not None
        assert run.planner_decision is not None
    # second run (if reached) used the scripted regularization knobs
    if len(state.runs) == 2:
        assert state.runs[1].training_config.dropout == 0.2
        assert state.runs[1].training_config.augment is True
