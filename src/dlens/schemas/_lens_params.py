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


class TwoStageResult(BaseModel):
    """Outcome of the two-stage path: extract -> validate -> (codegen).

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
    ok: bool = Field(description="True if parameters validated AND generated code passed.")
