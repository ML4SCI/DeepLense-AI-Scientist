# schemas/_experiment.py
"""Experiment-state contract threaded through the pipeline.

``ExperimentRun`` accumulates the typed output of every stage (simulation, model
design, train, infer, analysis) plus the planner's decision; ``ExperimentState``
holds the hypothesis and the run history. This is the data model the Experiment
Planner will consume — only the *types* live here; no planning logic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from dlens.schemas._downstream import AnalysisResult, InferResult, TrainResult
from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig
from dlens.schemas._simulation import SimConfig, SimOutput


class PlannerAction(str, Enum):
    """Where the planner routes control next (loop targets per WORKFLOW_DESIGN)."""

    SIMULATE = "simulate"          # loop back to agent 1 (Simulation)
    DESIGN_MODEL = "design_model"  # loop back to agent 2 (Model Design)
    TRAIN = "train"                # loop back to agent 3 (Training)
    REPORT = "report"             # proceed to the Report agent
    STOP = "stop"                  # terminate


class PlannerDecision(BaseModel):
    """A planner decision: next stage + rationale + parameter overrides."""

    action: PlannerAction = Field(description="Next stage to route to.")
    rationale: str = Field(description="Why this decision was made.")
    updated_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameter overrides for the next stage."
    )


class ExperimentRun(BaseModel):
    """One iteration of the experiment loop (accumulated in ExperimentState.runs)."""

    iteration: int = Field(description="0-based iteration index.")
    sim_config: Optional[SimConfig] = None
    sim_output: Optional[SimOutput] = None
    architecture: Optional[ArchitectureSpec] = None
    training_config: Optional[TrainingConfig] = None
    train_result: Optional[TrainResult] = None
    infer_result: Optional[InferResult] = None
    analysis_result: Optional[AnalysisResult] = None
    planner_decision: Optional[PlannerDecision] = None


class ExperimentState(BaseModel):
    """Shared state threaded through the pipeline, accumulating run history."""

    hypothesis: str = Field(description="Original problem statement / hypothesis.")
    current_iteration: int = Field(default=0, description="Current iteration index.")
    max_iterations: int = Field(default=3, ge=1, description="Iteration budget (guardrail).")
    runs: list[ExperimentRun] = Field(default_factory=list, description="Full run history.")
