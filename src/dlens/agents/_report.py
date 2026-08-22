# agents/_report.py
"""ReportAgent — agent #7 of the DLens pipeline (see docs/WORKFLOW_DESIGN.md).

Turns the full ``ExperimentState`` into a concise, structured ``ReportDocument``:
what was tried, the best result and the architecture that produced it, and a short
conclusion. A thin LLM agent with structured output; no tools.
"""

from __future__ import annotations

from typing import Any

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent
from dlens.config import build_model_from_env
from dlens.prompts._planner import REPORT_SYSTEM_PROMPT
from dlens.schemas._experiment import ExperimentState
from dlens.schemas._report import ReportDocument


class ReportAgent(DLensBaseAgent):
    """Writes the final report for an experiment campaign."""

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
                name="ReportAgent",
                description="Summarizes an experiment campaign into a final structured report.",
                custom_system_prompt=REPORT_SYSTEM_PROMPT,
                model=model or build_model_from_env(),
                debug=debug,
            )
        elif model is not None:
            config = config.model_copy(update={"model": model})

        super().__init__(config=config, output_type=ReportDocument, retries=retries)

    async def write(self, state: ExperimentState):
        """Convenience: run the report agent on the final experiment state."""
        return await self.arun(state.model_dump_json())
