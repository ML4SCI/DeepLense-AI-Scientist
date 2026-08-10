# agents/_inference.py
"""InferAgent — pipeline agent #4 (see docs/WORKFLOW_DESIGN.md).

A thin wrapper around the ``run_inference`` tool: scores a dataset with a trained
model (the weights produced by the Train Agent) and returns a typed ``InferResult``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent, OutputSchema
from dlens.config import build_model_from_env
from dlens.prompts._downstream import INFER_SYSTEM_PROMPT
from dlens.schemas._downstream import DatasetRef, InferResult
from dlens.tools._inference import InferBackend, InferDeps, get_infer_backend, register_infer_tool


class InferReport(OutputSchema):
    """The infer agent's structured output.

    Compact by design: the full prediction arrays never round-trip through the
    model — the authoritative :class:`InferResult` is surfaced code-side via
    ``InferAgent.last_result``.
    """

    message: str = Field(description="One-line summary of the inference run.")
    run_id: str = Field(description="Id of the inference run.")
    num_samples: int = Field(description="Number of samples scored.")
    accuracy: float | None = Field(default=None, description="Accuracy, if labels present.")


class InferAgent(DLensBaseAgent):
    """Runs a trained model over a dataset and reports predictions + metrics."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        config: BaseAgentConfig | None = None,
        backend: InferBackend | None = None,
        debug: bool = False,
        retries: int = 2,
    ) -> None:
        if config is None:
            config = BaseAgentConfig(
                name="InferAgent",
                description="Runs a trained model on a dataset and reports predictions.",
                custom_system_prompt=INFER_SYSTEM_PROMPT,
                model=model or build_model_from_env(),
                debug=debug,
            )
        elif model is not None:
            config = config.model_copy(update={"model": model})

        super().__init__(
            config=config,
            deps_type=InferDeps,
            output_type=InferReport,
            register_tools=register_infer_tool,
            retries=retries,
        )
        self._deps = InferDeps(backend=backend or get_infer_backend("auto"))

    async def infer(self, weights_path: str, dataset: DatasetRef):
        """Convenience: run the agent on typed inputs."""
        query = json.dumps(
            {"weights_path": weights_path, "dataset": dataset.model_dump(mode="json")}
        )
        return await self.arun(query, deps=self._deps)

    @property
    def last_result(self) -> InferResult | None:
        return self._deps.last_result
