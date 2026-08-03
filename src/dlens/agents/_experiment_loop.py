# agents/_experiment_loop.py
"""ExperimentLoop — the feedback loop that ties the pipeline together.

This is the E2E orchestrator (the "feedback loop" of WORKFLOW_DESIGN). Holding a
fixed dataset, it repeatedly: designs a model, trains, runs inference, analyzes the
result, and asks the ExperimentPlanner what to do next. The planner's
``updated_params`` steer the next iteration (architecture / hyperparameters), so the
loop studies what architecture the system converges to. It stops when the planner
says ``report``/``stop`` or the iteration budget is hit, then the ReportAgent writes
up the campaign.

Deterministic stages run via the tool functions/backends; the LLM reasoning is
concentrated in the planner and report sub-agents (both dependency-injected, so the
whole loop runs offline with mock backends + scripted models).
"""

from __future__ import annotations

from typing import Any, Optional

from dlens.agents._experiment_planner import ExperimentPlanner
from dlens.agents._report import ReportAgent
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._experiment import ExperimentRun, ExperimentState, PlannerAction
from dlens.schemas._model_design import (
    ArchitectureSpec,
    DatasetCharacteristics,
    TaskType,
    TrainingConfig,
)
from dlens.schemas._report import ReportDocument
from dlens.tools._analysis import compute_analysis
from dlens.tools._inference import InferBackend, get_infer_backend
from dlens.tools._model_design import recommend_baseline
from dlens.tools._training import TrainBackend, get_train_backend

_TERMINAL = {PlannerAction.REPORT, PlannerAction.STOP}
_TCFG_FIELDS = ("loss", "optimizer", "learning_rate", "batch_size", "epochs", "weight_decay", "lr_scheduler")


class ExperimentLoop:
    """Runs design -> train -> infer -> analysis -> plan, looping until report/stop."""

    def __init__(
        self,
        *,
        planner: Optional[ExperimentPlanner] = None,
        report_agent: Optional[ReportAgent] = None,
        train_backend: Optional[TrainBackend] = None,
        infer_backend: Optional[InferBackend] = None,
    ) -> None:
        self.planner = planner or ExperimentPlanner()
        self.report_agent = report_agent or ReportAgent()
        self.train_backend = train_backend or get_train_backend("auto")
        self.infer_backend = infer_backend or get_infer_backend("auto")

    async def run(
        self, *, hypothesis: str, dataset: DatasetRef, max_iterations: int = 3
    ) -> tuple[ExperimentState, Optional[ReportDocument]]:
        state = ExperimentState(hypothesis=hypothesis, max_iterations=max_iterations)
        characteristics = self._characteristics(dataset)
        overrides: dict[str, Any] = {}
        last_action: Optional[PlannerAction] = None

        for i in range(max_iterations):
            state.current_iteration = i

            rec = recommend_baseline(characteristics)
            arch = ArchitectureSpec(**rec["architecture"])
            tcfg = TrainingConfig(**rec["training_config"])
            arch, tcfg = self._apply_overrides(arch, tcfg, overrides)

            tr = self.train_backend.train(dataset, arch, tcfg)
            ir = self.infer_backend.infer(tr.weights_path, dataset)
            ar = compute_analysis(ir, dataset.class_names)

            run = ExperimentRun(
                iteration=i, architecture=arch, training_config=tcfg,
                train_result=tr, infer_result=ir, analysis_result=ar,
            )
            state.runs.append(run)

            decision = (await self.planner.decide(state)).output
            run.planner_decision = decision
            last_action = decision.action
            if decision.action in _TERMINAL:
                break
            overrides = decision.updated_params or {}

        report: Optional[ReportDocument] = None
        if last_action != PlannerAction.STOP:
            report = (await self.report_agent.write(state)).output
        return state, report

    @staticmethod
    def _characteristics(dataset: DatasetRef) -> DatasetCharacteristics:
        return DatasetCharacteristics(
            task=TaskType.CLASSIFICATION,
            image_shape=dataset.image_shape,
            channels=1,
            num_classes=dataset.num_classes,
            num_samples=dataset.num_samples,
        )

    @staticmethod
    def _apply_overrides(
        arch: ArchitectureSpec, tcfg: TrainingConfig, overrides: dict[str, Any]
    ) -> tuple[ArchitectureSpec, TrainingConfig]:
        """Apply the planner's parameter changes to the next iteration (typed, safe)."""
        arch_updates = {k: overrides[k] for k in ("name", "family", "physics_informed") if k in overrides}
        if arch_updates:
            arch = arch.model_copy(update=arch_updates)
        tcfg_updates = {k: overrides[k] for k in _TCFG_FIELDS if k in overrides}
        if tcfg_updates:
            tcfg = tcfg.model_copy(update=tcfg_updates)
        return arch, tcfg
