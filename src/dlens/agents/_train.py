# agents/_train.py
"""TrainAgent — pipeline agent #3 (see docs/WORKFLOW_DESIGN.md).

A thin wrapper around the ``train_model`` tool: given a dataset and the architecture
+ training configuration from the Model Design Agent, it trains a classification
model and returns a typed ``TrainResult`` (run id, weights path, metrics).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent, OutputSchema
from dlens.config import build_model_from_env
from dlens.prompts._downstream import TRAIN_SYSTEM_PROMPT
from dlens.schemas._downstream import DatasetRef, TrainResult
from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig
from dlens.tools._training import TrainBackend, TrainDeps, get_train_backend, register_train_tool


class TrainReport(OutputSchema):
    """The train agent's structured output."""

    message: str = Field(description="One-line summary of the training run.")
    result: TrainResult = Field(description="The training result.")


class TrainAgent(DLensBaseAgent):
    """Trains a classification model on a dataset per a given architecture/config."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        config: BaseAgentConfig | None = None,
        backend: TrainBackend | None = None,
        output_root: str = "models",
        debug: bool = False,
        retries: int = 2,
    ) -> None:
        if config is None:
            config = BaseAgentConfig(
                name="TrainAgent",
                description="Trains a classification model on a dataset and reports metrics.",
                custom_system_prompt=TRAIN_SYSTEM_PROMPT,
                model=model or build_model_from_env(),
                debug=debug,
            )
        elif model is not None:
            config = config.model_copy(update={"model": model})

        super().__init__(
            config=config,
            deps_type=TrainDeps,
            output_type=TrainReport,
            register_tools=register_train_tool,
            retries=retries,
        )
        self._deps = TrainDeps(backend=backend or get_train_backend("auto", output_root=output_root))

    async def train(
        self, dataset: DatasetRef, architecture: ArchitectureSpec, training_config: TrainingConfig
    ):
        """Convenience: run the agent on typed inputs."""
        query = json.dumps(
            {
                "dataset": dataset.model_dump(mode="json"),
                "architecture": architecture.model_dump(mode="json"),
                "training_config": training_config.model_dump(mode="json"),
            }
        )
        return await self.arun(query, deps=self._deps)

    @property
    def last_result(self) -> TrainResult | None:
        return self._deps.last_result
