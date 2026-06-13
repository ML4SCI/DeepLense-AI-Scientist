# agents/_analysis.py
"""AnalysisAgent — pipeline agent #5 (see docs/WORKFLOW_DESIGN.md).

A thin wrapper around the ``analyze_predictions`` tool: turns an ``InferResult`` into
a typed ``AnalysisResult`` (accuracy, per-class metrics, confusion matrix, macro AUC)
for the experiment planner to consume.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent, OutputSchema
from dlens.config import build_model_from_env
from dlens.prompts._downstream import ANALYSIS_SYSTEM_PROMPT
from dlens.schemas._downstream import AnalysisResult, InferResult
from dlens.tools._analysis import register_analysis_tool


class AnalysisReport(OutputSchema):
    """The analysis agent's structured output."""

    message: str = Field(description="One-line factual summary of model performance.")
    result: AnalysisResult = Field(description="The evaluation result.")


class AnalysisAgent(DLensBaseAgent):
    """Evaluates predictions into a structured analysis for the planner."""

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
                name="AnalysisAgent",
                description="Evaluates model predictions into a structured analysis report.",
                custom_system_prompt=ANALYSIS_SYSTEM_PROMPT,
                model=model or build_model_from_env(),
                debug=debug,
            )
        elif model is not None:
            config = config.model_copy(update={"model": model})

        super().__init__(
            config=config,
            output_type=AnalysisReport,
            register_tools=register_analysis_tool,
            retries=retries,
        )

    async def analyze(self, infer_result: InferResult):
        """Convenience: run the agent on a typed inference result."""
        return await self.arun(infer_result.model_dump_json())
