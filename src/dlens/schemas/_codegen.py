# schemas/_codegen.py
"""Schemas for the V2 simulation code-generation agent.

Workflow (agreed 2026-07-11): natural-language spec -> generated lenstronomy code
-> validated in a sandbox -> handed to Michael. These types are the contract for
that flow; ``GeneratedProgram`` (the agent's typed output) lives with the agent.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SimSpec(BaseModel):
    """A natural-language description of the simulation to generate code for."""

    description: str = Field(
        description="Plain-English physics/data requirement, e.g. 'strong lens, SIE + "
        "external shear, Sersic source, HST-like instrument, no substructure'."
    )
    notes: str = Field(default="", description="Extra constraints or preferences.")


class ValidationResult(BaseModel):
    """Outcome of validating generated code by running it in the sandbox."""

    passed: bool = Field(description="Did the code run and produce a sane lensing image?")
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual checks: ran, produced_output, is_2d, finite, non_trivial.",
    )
    image_shape: Optional[tuple[int, ...]] = Field(
        default=None, description="Shape of the produced image array, if any."
    )
    message: str = Field(default="", description="Human-readable summary / failure reason.")


class CodegenResult(BaseModel):
    """Final outcome surfaced to the caller (and, eventually, to Michael)."""

    spec: SimSpec
    code: str = Field(description="The generated lenstronomy script.")
    ok: bool = Field(description="True if validation passed.")
    attempts: int = Field(description="How many generate/validate attempts were made.")
    validation: ValidationResult
