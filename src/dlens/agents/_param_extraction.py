# agents/_param_extraction.py
"""ParamExtractionAgent — stage 1 of the two-stage simulation path.

NL prompt -> typed ``LensParameterSet`` (schemas/_lens_params.py). The set is
then checked by the deterministic validator (tools/_param_validator.py); on
failure the extraction is retried with the specific failure messages appended,
mirroring the codegen agent's retry style.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent, OutputSchema
from dlens.agents._models import OpenAIModel
from dlens.prompts._param_extraction import PARAM_EXTRACTION_SYSTEM_PROMPT
from dlens.schemas._lens_params import LensParameterSet

# Same reasoning as codegen: parameter extraction must respect exact 1.9.2
# kwargs; gpt-5.2 matches the paper protocol (chat completions suffice).
DEFAULT_EXTRACTION_MODEL = "gpt-5.2"


class ExtractedParameters(OutputSchema):
    """The agent's typed output: a complete lenstronomy parameter set."""

    params: LensParameterSet = Field(
        description="The complete physical parameter set for the requested simulation."
    )


class ParamExtractionAgent(DLensBaseAgent):
    """Extracts a typed lenstronomy 1.9.2 parameter set from a natural-language spec."""

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
                name="ParamExtractionAgent",
                description=(
                    "Translates a natural-language lensing request into a typed, "
                    "physically validated lenstronomy parameter set."
                ),
                custom_system_prompt=PARAM_EXTRACTION_SYSTEM_PROMPT,
                model=model or OpenAIModel(model_name=DEFAULT_EXTRACTION_MODEL),
                debug=debug,
            )
        elif model is not None:
            config = config.model_copy(update={"model": model})

        super().__init__(config=config, output_type=ExtractedParameters, retries=retries)

    async def extract(self, description: str, *, last_error: str | None = None):
        """One extraction pass (returns the raw run result; .output is ExtractedParameters)."""
        query = description
        if last_error:
            query += (
                "\n\nThe previous parameter set failed deterministic validation:\n"
                f"{last_error}\nFix exactly these issues and keep everything else."
            )
        return await self.arun(query)
