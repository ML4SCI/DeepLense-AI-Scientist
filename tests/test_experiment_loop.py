"""Experiment planner, report agent, and the full feedback loop — fully offline.

The planner/report use scripted models; the train/infer stages use mock backends,
so the entire design -> train -> infer -> analysis -> plan -> report loop runs with
no GPU, no LLM, and no network.
"""

from __future__ import annotations

import asyncio

from dlens.agents._experiment_loop import ExperimentLoop
from dlens.agents._experiment_planner import ExperimentPlanner
from dlens.agents._report import ReportAgent
from dlens.agents._scripted_planner import make_scripted_planner_model, make_scripted_report_model
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._experiment import ExperimentRun, ExperimentState, PlannerAction, PlannerDecision
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec
from dlens.schemas._report import ReportDocument
from dlens.tools._inference import MockInferBackend
from dlens.tools._training import MockTrainBackend


def _dataset() -> DatasetRef:
    return DatasetRef(root="/nonexistent", class_names=["a", "b", "c"], image_shape=(8, 8), num_samples=12)


def _state_with_run(accuracy: float = 0.8, iteration: int = 0) -> ExperimentState:
    arch = ArchitectureSpec(name="baseline", family=ArchFamily.RESNET, input_shape=(8, 8), num_classes=3)
    run = ExperimentRun(
        iteration=iteration,
        architecture=arch,
        analysis_result={  # AnalysisResult-shaped dict is fine for the scripted report
            "accuracy": accuracy, "macro_auc": 0.9, "confusion_matrix": [[1]], "per_class": {}, "summary": "ok",
        },
    )
    return ExperimentState(hypothesis="can a baseline separate the classes?", current_iteration=iteration, runs=[run])


def test_planner_decides_offline():
    planner = ExperimentPlanner(model=make_scripted_planner_model(report_after=1))
    # iteration 0 -> continue
    d0 = asyncio.run(planner.decide(ExperimentState(hypothesis="h", current_iteration=0))).output
    assert isinstance(d0, PlannerDecision) and d0.action == PlannerAction.DESIGN_MODEL
    # iteration 1 -> report
    d1 = asyncio.run(planner.decide(ExperimentState(hypothesis="h", current_iteration=1))).output
    assert d1.action == PlannerAction.REPORT


def test_report_agent_offline():
    report = ReportAgent(model=make_scripted_report_model())
    doc = asyncio.run(report.write(_state_with_run(accuracy=0.83))).output
    assert isinstance(doc, ReportDocument)
    assert doc.best_accuracy == 0.83
    assert doc.final_architecture == "baseline"


def test_full_loop_offline_converges_and_reports():
    loop = ExperimentLoop(
        planner=ExperimentPlanner(model=make_scripted_planner_model(report_after=1)),
        report_agent=ReportAgent(model=make_scripted_report_model()),
        train_backend=MockTrainBackend(),
        infer_backend=MockInferBackend(),
    )
    state, report = asyncio.run(
        loop.run(hypothesis="can a baseline separate the classes?", dataset=_dataset(), max_iterations=3)
    )
    # iter 0 -> design_model (continue), iter 1 -> report
    assert len(state.runs) == 2
    assert state.runs[0].planner_decision.action == PlannerAction.DESIGN_MODEL
    assert state.runs[1].planner_decision.action == PlannerAction.REPORT
    assert all(r.analysis_result is not None for r in state.runs)
    # planner's updated_params were applied to the next iteration
    assert state.runs[1].training_config.learning_rate == 0.0001
    assert isinstance(report, ReportDocument)


def test_loop_stop_skips_report():
    loop = ExperimentLoop(
        planner=ExperimentPlanner(model=make_scripted_planner_model(report_after=0, terminal_action="stop")),
        report_agent=ReportAgent(model=make_scripted_report_model()),
        train_backend=MockTrainBackend(),
        infer_backend=MockInferBackend(),
    )
    state, report = asyncio.run(loop.run(hypothesis="h", dataset=_dataset(), max_iterations=3))
    assert len(state.runs) == 1
    assert state.runs[0].planner_decision.action == PlannerAction.STOP
    assert report is None  # 'stop' ends without a report


def test_loop_respects_iteration_budget():
    # planner never terminates -> loop must stop at max_iterations and still report
    loop = ExperimentLoop(
        planner=ExperimentPlanner(model=make_scripted_planner_model(report_after=999)),
        report_agent=ReportAgent(model=make_scripted_report_model()),
        train_backend=MockTrainBackend(),
        infer_backend=MockInferBackend(),
    )
    state, report = asyncio.run(loop.run(hypothesis="h", dataset=_dataset(), max_iterations=2))
    assert len(state.runs) == 2
    assert isinstance(report, ReportDocument)
