# schemas/_lens_params.py
"""Typed lenstronomy parameter set for the two-stage simulation path.

Two-stage design (Michael/Lucca, meeting 2026-08-01): natural language ->
physical parameter set -> deterministic validation -> code generation.
Rationale: validating generated physics *code* is a trust problem, but
validating the *parameters* against known physical distributions is tractable
and deterministic.

The structure mirrors lenstronomy 1.9.2's actual input dictionaries
(introspected against the sandbox image's install — same method as the codegen
cheat-sheet): model lists are lists of profile-name strings and the kwargs are
lists of one plain dict per profile, exactly as ``LensModel`` / ``LightModel`` /
``ImageModel`` consume them. The container is typed; the per-profile dicts stay
dicts because that IS lenstronomy's structure — per-profile key checks are the
deterministic validator's job (``tools/_param_validator.py``), keyed to the
introspected 1.9.2 ``param_names``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from dlens.schemas._codegen import CodegenResult, SimSpec


class InstrumentBlock(BaseModel):
    """The data/instrument block — mirrors ``sim_util.data_configure_simple``.

    1.9.2 signature (introspected): ``data_configure_simple(numPix, deltaPix,
    exposure_time=None, background_rms=None, ...)`` — camelCase in this version.
    """

    # Field names intentionally match 1.9.2's camelCase arguments.
    numPix: int = Field(description="Pixels per axis of the simulated image.")
    deltaPix: float = Field(description="Pixel scale in arcsec/pixel.")
    exposure_time: Optional[float] = Field(
        default=None, description="Exposure time in seconds (None = noiseless)."
    )
    background_rms: Optional[float] = Field(
        default=None, description="Background noise RMS per pixel (None = noiseless)."
    )


class PSFBlock(BaseModel):
    """PSF kwargs — mirrors ``PSF(psf_type=, fwhm=, pixel_size=)`` in 1.9.2.

    1.9.2 has NO 'sigma' argument (a common version-blend); Gaussian PSFs take
    ``fwhm``. ``pixel_size`` should equal the grid's ``deltaPix``.
    """

    psf_type: str = Field(default="GAUSSIAN", description="'GAUSSIAN' | 'PIXEL' | 'NONE'.")
    fwhm: Optional[float] = Field(
        default=None, description="PSF FWHM in arcsec (required for GAUSSIAN)."
    )
    pixel_size: Optional[float] = Field(
        default=None, description="PSF pixel scale in arcsec (should match deltaPix)."
    )


class LensParameterSet(BaseModel):
    """The full physical parameter set the extraction stage must produce.

    ``lens_model_list``/``kwargs_lens`` and ``source_model_list``/
    ``kwargs_source`` are positionally paired, one kwargs dict per profile —
    lenstronomy's own convention. Redshifts and the DeepLense-recipe masses ride
    alongside as physical metadata (they are validation targets and codegen
    inputs, not ImageModel kwargs).
    """

    model_config = ConfigDict(protected_namespaces=())

    z_lens: float = Field(description="Lens (deflector) redshift.")
    z_source: float = Field(description="Source galaxy redshift (must exceed z_lens).")

    lens_model_list: list[str] = Field(
        description="Lens mass profiles, e.g. ['SIE', 'SHEAR'] (1.9.2 profile names)."
    )
    kwargs_lens: list[dict[str, float]] = Field(
        description="One kwargs dict per lens profile, positionally paired with "
        "lens_model_list (e.g. SIE: theta_E, e1, e2, center_x, center_y)."
    )
    source_model_list: list[str] = Field(
        description="Source light profiles, e.g. ['SERSIC_ELLIPSE']."
    )
    kwargs_source: list[dict[str, float]] = Field(
        description="One kwargs dict per source profile (e.g. SERSIC_ELLIPSE: amp, "
        "R_sersic, n_sersic, e1, e2, center_x, center_y)."
    )

    kwargs_data: InstrumentBlock = Field(description="Grid + exposure/noise block.")
    kwargs_psf: PSFBlock = Field(default_factory=PSFBlock, description="PSF block.")

    halo_mass: Optional[float] = Field(
        default=None,
        description="Main halo mass in M_sun when the request is framed in "
        "DeepLense-recipe terms (canonical value 1e12).",
    )
    axion_mass: Optional[float] = Field(
        default=None,
        description="Axion mass in eV when vortex substructure is requested "
        "(DeepLenseSim range: ~1e-24 to 1e-22).",
    )
    vortex_mass: Optional[float] = Field(
        default=None,
        description="Vortex mass in M_sun when vortex substructure is requested "
        "(canonical value 3e10).",
    )


class ParamValidationResult(BaseModel):
    """Outcome of the deterministic parameter validation (no LLM involved)."""

    passed: bool = Field(description="Did every computed check pass?")
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-check outcomes; checks that could not be computed "
        "(missing inputs) are omitted rather than failed.",
    )
    messages: list[str] = Field(
        default_factory=list,
        description="One specific, actionable message per failed check.",
    )
    snr_estimate: Optional[float] = Field(
        default=None,
        description="Aperture-integrated SNR estimate, when computable from "
        "amp/R_sersic/exposure/noise.",
    )


class FieldComparison(BaseModel):
    """One field where the generated code diverged from the validated set."""

    field: str = Field(description="Dotted field path, e.g. 'kwargs_lens[0].theta_E'.")
    expected: object = Field(description="The validated value.")
    actual: object = Field(description="The value found in the generated script.")


class CodeParamComparison(BaseModel):
    """Structured diff of the generated script against the validated parameters.

    Produced by AST parsing (tools/_code_param_check.py). ``passed`` requires
    every checked field to match: diverged, missing, AND unresolved all block —
    a runtime-computed value is reported as unresolved, never assumed correct.
    """

    passed: bool = Field(description="True only if every checked field matched.")
    matched: list[str] = Field(default_factory=list, description="Fields verified equal.")
    diverged: list[FieldComparison] = Field(
        default_factory=list, description="Fields with a different value in the script."
    )
    missing: list[str] = Field(
        default_factory=list, description="Fields/blocks not found in the script."
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description="Fields whose script value is computed at runtime (not statically "
        "verifiable).",
    )
    messages: list[str] = Field(
        default_factory=list, description="Actionable per-field feedback for the retry."
    )


class CodegenPassRecord(BaseModel):
    """One outer codegen pass, kept whether L3 accepted or rejected it.

    Rejected passes used to be discarded: only the final program and comparison
    survived, so "L3 rejected this program" could not be substantiated after the
    run. Every pass is now recorded.
    """

    pass_index: int = Field(description="1-based outer codegen pass number.")
    code: Optional[str] = Field(
        default=None, description="The generated program for this pass."
    )
    sandbox_passed: Optional[bool] = Field(
        default=None, description="Did L1 (sandbox + structural validation) pass?"
    )
    sandbox_message: Optional[str] = Field(default=None, description="L1 validation message.")
    image_shape: Optional[list[int]] = Field(default=None, description="L1 image shape, if any.")
    codegen_attempts: Optional[int] = Field(
        default=None, description="Inner sandbox retry attempts used within this pass."
    )
    comparison: Optional[CodeParamComparison] = Field(
        default=None, description="L3 AST diff for this pass; None if L1 failed first."
    )
    accepted: bool = Field(
        default=False, description="True if this pass satisfied both L1 and L3."
    )


class TwoStageResult(BaseModel):
    """Outcome of the two-stage path: extract -> validate -> codegen -> verify.

    Carries the extraction ``reasoning`` (framework convention) and the final
    ``CodegenResult`` when the pipeline reached the code-generation stage.
    """

    reasoning: str = Field(description="The reasoning process of the extraction agent.")
    spec: SimSpec
    params: Optional[LensParameterSet] = Field(
        default=None, description="The last extracted parameter set (validated or not)."
    )
    param_validation: ParamValidationResult
    extraction_attempts: int = Field(description="Extraction attempts made (retries on failure).")
    codegen: Optional[CodegenResult] = Field(
        default=None, description="Code-generation outcome; None if extraction never validated."
    )
    code_param_comparison: Optional[CodeParamComparison] = Field(
        default=None,
        description="AST diff of the final script vs the validated parameters; "
        "None if codegen never produced a passing script.",
    )
    codegen_passes: int = Field(
        default=0,
        description="Outer codegen passes (a fresh pass is triggered when the "
        "script diverges from the validated parameters).",
    )
    codegen_pass_log: list[CodegenPassRecord] = Field(
        default_factory=list,
        description="One record per outer codegen pass, INCLUDING passes L3 "
        "rejected. Runs recorded before 2026-08-10 have an empty log.",
    )
    ok: bool = Field(
        description="True if parameters validated, generated code passed the sandbox, "
        "AND the script verifiably used the validated parameters."
    )
