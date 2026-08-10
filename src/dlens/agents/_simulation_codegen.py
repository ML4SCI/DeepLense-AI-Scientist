# agents/_simulation_codegen.py
"""SimulationCodegenAgent — the V2 dedicated data-simulation agent (code generation).

Turns a natural-language physics spec into a lenstronomy script, runs it in a
sandbox, validates that it produced a sane lensing image, and retries on failure.
This is the "describe the physics -> agent writes lens simulation code -> validate"
workflow agreed on 2026-07-11; the validated code is what gets handed to Michael.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pydantic import Field

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent, OutputSchema
from dlens.agents._models import OpenAIModel
from dlens.prompts._codegen import CODEGEN_SYSTEM_PROMPT, CODEGEN_SYSTEM_PROMPT_UNGROUNDED
from dlens.schemas._codegen import CodegenResult, SimSpec, ValidationResult
from dlens.tools._sandbox import ExecResult, Sandbox, get_sandbox

# Code generation needs a capable model; local Ollama models hallucinate here.
# gpt-5.2 matches the paper protocol and works on chat completions (the Responses
# API is only required for the gpt-5.6 family).
DEFAULT_CODEGEN_MODEL = "gpt-5.2"


class GeneratedProgram(OutputSchema):
    """The agent's typed output: a runnable lenstronomy script."""

    code: str = Field(description="A self-contained, runnable lenstronomy script.")


def validate_output(
    result: ExecResult, expected_shape: tuple[int, int] | None = None
) -> ValidationResult:
    """Validate a sandbox run: did it produce a finite, non-trivial 2-D image?

    When ``expected_shape`` is given (i.e. the request specified an image size),
    the produced image must match it exactly.
    """
    checks = {
        "ran": result.ok,
        "produced_output": result.output_path is not None,
        "is_2d": False,
        "finite": False,
        "non_trivial": False,
    }
    if expected_shape is not None:
        checks["matches_requested_size"] = False
    if not result.output_path:
        if result.ok:
            msg = "script ran (exit 0) but wrote no output to the DLENS_OUTPUT path"
        else:
            msg = result.error or "process failed before producing output"
        return ValidationResult(passed=False, checks=checks, message=msg)
    try:
        arr = np.load(result.output_path)
    except Exception as exc:  # noqa: BLE001 - report any load failure
        return ValidationResult(passed=False, checks=checks, message=f"output not loadable: {exc}")

    checks["is_2d"] = arr.ndim == 2
    checks["finite"] = bool(np.all(np.isfinite(arr)))
    checks["non_trivial"] = bool(arr.size > 0 and float(np.max(arr) - np.min(arr)) > 0.0)
    if expected_shape is not None:
        checks["matches_requested_size"] = tuple(arr.shape) == tuple(expected_shape)
    passed = result.ok and all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    return ValidationResult(
        passed=passed,
        checks=checks,
        image_shape=tuple(int(x) for x in arr.shape),
        message="ok" if passed else "failed checks: " + ", ".join(failed),
    )


class SimulationCodegenAgent(DLensBaseAgent):
    """Generates and validates lenstronomy code from a natural-language spec."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        sandbox: Optional[Sandbox] = None,
        config: BaseAgentConfig | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        grounded: bool = True,
        debug: bool = False,
        retries: int = 2,
    ) -> None:
        if config is None:
            config = BaseAgentConfig(
                name="SimulationCodegenAgent",
                description=(
                    "Generates and validates lenstronomy simulation code from a "
                    "natural-language physics specification."
                ),
                custom_system_prompt=(CODEGEN_SYSTEM_PROMPT if grounded
                                      else CODEGEN_SYSTEM_PROMPT_UNGROUNDED),
                model=model or OpenAIModel(model_name=DEFAULT_CODEGEN_MODEL),
                debug=debug,
            )
        elif model is not None:
            config = config.model_copy(update={"model": model})

        super().__init__(config=config, output_type=GeneratedProgram, retries=retries)
        # Default to the real (Docker) sandbox; tests inject LocalSandbox.
        self.sandbox = sandbox or get_sandbox("docker")
        self.max_retries = max_retries
        self.timeout = timeout

    async def generate(self, spec: SimSpec, *, last_error: str | None = None):
        """One generation pass (returns the raw run result; .output is GeneratedProgram)."""
        query = spec.description
        if spec.notes:
            query += f"\n\nConstraints: {spec.notes}"
        if last_error:
            query += f"\n\nThe previous attempt failed validation:\n{last_error}\nFix the code."
        return await self.arun(query)

    async def generate_and_validate(
        self, spec: SimSpec, *, expected_shape: tuple[int, int] | None = None
    ) -> CodegenResult:
        """Generate -> run in sandbox -> validate, retrying up to ``max_retries``.

        ``expected_shape`` (when the request pins an image size) is enforced by the
        validation harness and included in the retry feedback.
        """
        last_error: str | None = None
        code = ""
        reasoning = ""
        validation = ValidationResult(passed=False, message="no attempt made")
        for attempt in range(1, self.max_retries + 1):
            program = (await self.generate(spec, last_error=last_error)).output
            code, reasoning = program.code, program.reasoning
            exec_result = self.sandbox.run(code, timeout=self.timeout)
            validation = validate_output(exec_result, expected_shape=expected_shape)
            if validation.passed:
                return CodegenResult(
                    reasoning=reasoning, spec=spec, code=code, ok=True,
                    attempts=attempt, validation=validation,
                )
            last_error = f"{validation.message}\n{exec_result.stderr or exec_result.error or ''}"[:2000]
        return CodegenResult(
            reasoning=reasoning, spec=spec, code=code, ok=False,
            attempts=self.max_retries, validation=validation,
        )
