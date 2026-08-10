# agents/_model_design.py
"""ModelDesignAgent — agent #2 of the DLens pipeline (see docs/WORKFLOW_DESIGN.md).

A thin wrapper around the deterministic ``recommend_architecture`` tool: given the
dataset characteristics (derivable from the Simulation Agent's ``SimOutput``), it
returns a structured ``ArchitectureSpec`` + ``TrainingConfig`` with a short
rationale. Stateless and single-turn (``DLensBaseAgent``), no HITL gate.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent, OutputSchema
from dlens.config import build_model_from_env
from dlens.prompts._model_design import MODEL_DESIGN_SYSTEM_PROMPT
from dlens.schemas._model_design import ArchitectureSpec, DatasetCharacteristics, TrainingConfig
from dlens.tools._model_design import register_model_design_tool


class ModelDesignReport(OutputSchema):
    """The model-design agent's structured output (architecture + training config)."""

    message: str = Field(description="One-line summary of the recommendation.")
    architecture: ArchitectureSpec = Field(description="Proposed architecture.")
    training_config: TrainingConfig = Field(description="Proposed training hyperparameters.")


class ModelDesignAgent(DLensBaseAgent):
    """Recommends an architecture + training config for the downstream task."""

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
                name="ModelDesignAgent",
                description=(
                    "Designs the neural-network architecture and training configuration "
                    "for the downstream lensing task, given dataset characteristics."
                ),
                custom_system_prompt=MODEL_DESIGN_SYSTEM_PROMPT,
                model=model or build_model_from_env(),
                debug=debug,
            )
        elif model is not None:
            config = config.model_copy(update={"model": model})

        super().__init__(
            config=config,
            output_type=ModelDesignReport,
            register_tools=register_model_design_tool,
            retries=retries,
        )

    async def design(self, characteristics: DatasetCharacteristics):
        """Convenience: run the agent on typed dataset characteristics."""
        return await self.arun(characteristics.model_dump_json())
