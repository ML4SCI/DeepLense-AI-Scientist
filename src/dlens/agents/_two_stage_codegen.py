# agents/_two_stage_codegen.py
"""Two-stage simulation path: NL -> parameters -> validate -> code generation.

Michael/Lucca's proposed design (meeting 2026-08-01): validating generated
physics code is a trust problem, but validating the *parameters* against known
physical distributions is tractable and deterministic.

This module COMPOSES the existing pieces — ``ParamExtractionAgent`` (stage 1),
the deterministic ``validate_parameters`` tool, and the UNCHANGED
``SimulationCodegenAgent`` (stage 2) — so the direct NL->code path stays intact
and remains the default (the 168-run grounding ablation depends on its exact
behavior). Select this path explicitly (``TwoStageSimulationAgent`` /
``--two-stage`` in the demo script).
"""

from __future__ import annotations

from typing import Optional

from dlens.agents._param_extraction import ParamExtractionAgent
from dlens.agents._simulation_codegen import SimulationCodegenAgent
from dlens.schemas._codegen import SimSpec
from dlens.schemas._lens_params import LensParameterSet, ParamValidationResult, TwoStageResult
from dlens.tools._code_param_check import compare_code_to_params
from dlens.tools._param_validator import ValidationRanges, validate_parameters


def params_to_codegen_notes(params: LensParameterSet) -> str:
    """Render validated parameters as binding constraints for the codegen stage.

    This is the clean seam for Michael's ground-truth data: his NL->lenstronomy
    input dictionaries land as ``LensParameterSet`` fixtures and flow through
    here unchanged.
    """
    return (
        "Use EXACTLY this validated lenstronomy 1.9.2 parameter set — do not "
        "re-derive or alter any value:\n"
        + params.model_dump_json(indent=2, exclude_none=True)
    )


class TwoStageSimulationAgent:
    """Extract -> validate (deterministic, bounded retries) -> generate code."""

    def __init__(
        self,
        *,
        extractor: Optional[ParamExtractionAgent] = None,
        codegen: Optional[SimulationCodegenAgent] = None,
        ranges: Optional[ValidationRanges] = None,
        max_extraction_retries: int = 3,
        max_codegen_passes: int = 2,
    ) -> None:
        self.extractor = extractor or ParamExtractionAgent()
        self.codegen = codegen or SimulationCodegenAgent()
        self.ranges = ranges
        self.max_extraction_retries = max_extraction_retries
        # Outer passes: a pass whose script diverges from the validated
        # parameters (AST diff) is regenerated with the specific divergences.
        self.max_codegen_passes = max_codegen_passes

    async def run(self, spec: SimSpec) -> TwoStageResult:
        query = spec.description
        if spec.notes:
            query += f"\n\nConstraints: {spec.notes}"

        last_error: str | None = None
        reasoning = ""
        params: LensParameterSet | None = None
        validation = ParamValidationResult(passed=False, messages=["no attempt made"])
        for attempt in range(1, self.max_extraction_retries + 1):
            extracted = (await self.extractor.extract(query, last_error=last_error)).output
            params, reasoning = extracted.params, extracted.reasoning
            validation = validate_parameters(params, self.ranges)
            if validation.passed:
                break
            last_error = "\n".join(validation.messages)
        else:
            return TwoStageResult(
                reasoning=reasoning, spec=spec, params=params,
                param_validation=validation, extraction_attempts=self.max_extraction_retries,
                codegen=None, ok=False,
            )

        # Codegen + verification loop: the sandbox proves the script runs; the
        # AST diff proves it used the validated parameters. Divergence triggers
        # a fresh pass carrying the exact per-field feedback (bounded).
        base_notes = params_to_codegen_notes(params)
        notes = base_notes
        codegen_result = None
        comparison = None
        for cpass in range(1, self.max_codegen_passes + 1):
            codegen_spec = SimSpec(description=spec.description, notes=notes)
            codegen_result = await self.codegen.generate_and_validate(codegen_spec)
            if not codegen_result.ok:
                break
            comparison = compare_code_to_params(codegen_result.code, params)
            if comparison.passed:
                break
            notes = (
                base_notes
                + "\n\nThe previous script did not verifiably use the validated "
                "parameters:\n- "
                + "\n- ".join(comparison.messages)
                + "\nRewrite the script using the validated values as literals."
            )
        return TwoStageResult(
            reasoning=reasoning, spec=spec, params=params,
            param_validation=validation, extraction_attempts=attempt,
            codegen=codegen_result, code_param_comparison=comparison,
            codegen_passes=cpass,
            ok=bool(codegen_result.ok and comparison is not None and comparison.passed),
        )
