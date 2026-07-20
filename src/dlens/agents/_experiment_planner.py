# agents/_experiment_planner.py
"""ExperimentPlanner — agent #6 of the DLens pipeline (see docs/WORKFLOW_DESIGN.md).

The reasoning core of the loop. Given the experiment history (``ExperimentState``),
it returns a typed ``PlannerDecision``: whether to redesign the model, adjust
hyperparameters, regenerate data, or stop and report — with the parameter changes
to apply next. A thin LLM agent with structured output; no tools.
"""

from __future__ import annotations

from typing import Any

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent
from dlens.config import build_model_from_env
from dlens.prompts._planner import PLANNER_SYSTEM_PROMPT
from dlens.schemas._experiment import ExperimentState, PlannerDecision


class ExperimentPlanner(DLensBaseAgent):
    """Decides the next action in the experiment loop from the run history."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        config: BaseAgentConfig | None = None,
        debug: bool = False,
        retries: int = 2,
    ) -> None:
        if config is None:
            config = BaseAgentConfig(
                name="ExperimentPlanner",
                description=(
                    "Reviews the experiment history and decides whether to redesign the "
                    "model, adjust hyperparameters, regenerate data, or stop and report."
                ),
                custom_system_prompt=PLANNER_SYSTEM_PROMPT,
                model=model or build_model_from_env(),
                debug=debug,
            )
        elif model is not None:
            config = config.model_copy(update={"model": model})

        super().__init__(config=config, output_type=PlannerDecision, retries=retries)

    async def decide(self, state: ExperimentState):
        """Run the planner on a COMPACT view of the experiment state.

        Prediction arrays (predictions / probabilities / true_labels) are excluded —
        they are far too large for the model context; the planner reasons from the
        aggregate metrics (train metrics, val accuracy/AUC, confusion, per-class).
        """
        compact = state.model_dump(
            mode="json",
            exclude={
                "runs": {
                    "__all__": {
                        "infer_result": {"predictions", "probabilities", "true_labels"},
                        "sim_output": True,
                    }
                }
            },
        )
        import json

        return await self.arun(json.dumps(compact))


class ReActPlannerStrategy:
    """The default PlannerStrategy: a ReAct-style LLM planner (one observation ->
    one reasoned decision per iteration). Swappable — e.g. an agentic tree search
    can implement the same ``decide(state)`` protocol."""

    name = "react"

    def __init__(self, planner: ExperimentPlanner | None = None) -> None:
        self.planner = planner or ExperimentPlanner()

    async def decide(self, state: ExperimentState) -> PlannerDecision:
        return (await self.planner.decide(state)).output
