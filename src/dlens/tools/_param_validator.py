# tools/_param_validator.py
"""Deterministic physical-plausibility validator for extracted lens parameters.

This is a plain function, not an LLM: the two-stage design's whole point is
that this step is tractable and reproducible. Every range is configurable via
``ValidationRanges`` and every failure produces a specific, actionable message
that gets fed back to the extraction agent's retry.

Range provenance is cited per field below. Three sources:
  * "DeepLenseSim recipe" — values used by the Model_I–III scripts / the V1
    SimConfig schema (halo 1e12 M_sun, z 0.5/1.0, axion 1e-24..1e-22 eV,
    vortex 3e10 M_sun, grids 150 px @ 0.05" and 64 px).
  * "lenstronomy 1.9.2" — introspected library facts (profile param_names,
    ObservationConfig band values: HST 0.08"/5400 s, Euclid 0.101"/565 s).
  * "heuristic — needs Lucca's distributions" — placeholder bounds to be
    replaced when Lucca's published parameter distributions arrive (Einstein
    radius by redshift, ellipticity, Sersic index, target SNR >= ~25).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dlens.schemas._lens_params import LensParameterSet, ParamValidationResult

# Per-profile parameter names, introspected from lenstronomy 1.9.2
# (LensModel(...).lens_model.func_list[n].param_names). Extend as profiles are
# added to the extraction prompt.
PROFILE_PARAMS: dict[str, list[str]] = {
    # lens mass profiles
    "SIE": ["theta_E", "e1", "e2", "center_x", "center_y"],
    "SIS": ["theta_E", "center_x", "center_y"],
    "EPL": ["theta_E", "gamma", "e1", "e2", "center_x", "center_y"],
    "SHEAR": ["gamma1", "gamma2", "ra_0", "dec_0"],
    "NFW": ["Rs", "alpha_Rs", "center_x", "center_y"],
    # source light profiles
    "SERSIC": ["amp", "R_sersic", "n_sersic", "center_x", "center_y"],
    "SERSIC_ELLIPSE": ["amp", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"],
    "GAUSSIAN": ["amp", "sigma", "center_x", "center_y"],
}


@dataclass
class ValidationRanges:
    """Configurable bounds. Defaults cite their source; tighten when Lucca's
    distributions arrive."""

    # DeepLenseSim recipe: z_halo=0.5, z_gal=1.0; V1 SimConfig allows lens z<5, source z<10.
    z_lens_max: float = 5.0
    z_source_max: float = 10.0
    # Galaxy-scale strong lenses. Heuristic — needs Lucca's Einstein-radius-by-
    # redshift distribution (canonical 1e12 M_sun halo at z=0.5/1.0 gives ~1").
    theta_e_min: float = 0.1
    theta_e_max: float = 10.0
    # lenstronomy e1/e2 convention: |e|<0.5 keeps axis ratio q>~1/3.
    # Heuristic — needs Lucca's ellipticity distribution.
    ellipticity_max: float = 0.5
    # Sersic index: lenstronomy's own profile bounds use ~0.36 as the numerical
    # floor; 8 covers de Vaucouleurs-plus. Heuristic upper — needs Lucca.
    n_sersic_min: float = 0.36
    n_sersic_max: float = 8.0
    # Grid sanity. DeepLenseSim recipes: Model_I 150 px @ 0.05"; Model_II/III
    # 64 px; lenstronomy 1.9.2 ObservationConfigs: Euclid 0.101", HST 0.08".
    numpix_min: int = 16
    numpix_max: int = 1024
    deltapix_min: float = 0.01
    deltapix_max: float = 0.5
    # DeepLenseSim recipe: canonical halo 1e12 M_sun (eval-suite variations
    # span 5e11–3e12); order-of-magnitude guard rails around that.
    halo_mass_min: float = 1e10
    halo_mass_max: float = 1e14
    # DeepLenseSim recipe / V1 SimConfig: axion mass typically 1e-24..1e-22 eV.
    axion_mass_min: float = 1e-24
    axion_mass_max: float = 1e-22
    # DeepLenseSim recipe: canonical vortex 3e10 M_sun (eval variation 1e10).
    vortex_mass_min: float = 1e9
    vortex_mass_max: float = 1e12
    # Lucca's pipeline targets SNR >= ~25 (meeting 2026-08-01).
    # The estimator below is an aperture-integrated proxy — heuristic until
    # Lucca's exact SNR definition arrives.
    snr_min: float = 25.0
    profile_params: dict[str, list[str]] = field(default_factory=lambda: PROFILE_PARAMS)


def _estimate_snr(params: LensParameterSet) -> float | None:
    """Aperture-integrated SNR estimate for the source detection.

    Total Sersic flux ~ amp * 2*pi * R_sersic^2 (counts/s; the n-dependent
    factor is order unity for disks and ignored), integrated over an aperture
    of radius 2*R_sersic against Poisson + background noise. Lensing
    magnification is ignored, which makes the estimate conservative. A
    peak-pixel version of this check was an order of magnitude too strict
    against real gpt-5.2 extractions (textbook amp values scored SNR ~1-4).
    Heuristic — replace with Lucca's SNR definition when his distributions
    arrive.
    """
    data = params.kwargs_data
    if data.exposure_time is None or data.background_rms is None:
        return None
    if data.exposure_time <= 0 or data.background_rms <= 0:
        return None  # covered by their own checks
    pairs = [
        (kw["amp"], kw.get("R_sersic", 0.3))
        for kw in params.kwargs_source
        if "amp" in kw
    ]
    if not pairs:
        return None
    amp, r_sersic = max(pairs)
    total_counts = amp * 2 * 3.141592653589793 * r_sersic**2 * data.exposure_time
    n_pix_aperture = 3.141592653589793 * (2 * r_sersic) ** 2 / data.deltaPix**2
    background_var = n_pix_aperture * (data.background_rms * data.exposure_time) ** 2
    noise = (total_counts + background_var) ** 0.5
    return total_counts / noise if noise > 0 else None


def validate_parameters(
    params: LensParameterSet, ranges: ValidationRanges | None = None
) -> ParamValidationResult:
    """Run every computable check; omit (rather than fail) uncomputable ones."""
    r = ranges or ValidationRanges()
    checks: dict[str, bool] = {}
    messages: list[str] = []

    def fail(name: str, msg: str) -> None:
        checks[name] = False
        messages.append(msg)

    def ok(name: str) -> None:
        checks[name] = True

    # --- redshifts ---------------------------------------------------------
    if 0 < params.z_lens < r.z_lens_max:
        ok("z_lens_range")
    else:
        fail("z_lens_range", f"z_lens={params.z_lens} outside (0, {r.z_lens_max}); "
             "typical DeepLense lenses sit near z=0.5.")
    if 0 < params.z_source < r.z_source_max:
        ok("z_source_range")
    else:
        fail("z_source_range", f"z_source={params.z_source} outside (0, {r.z_source_max}); "
             "typical DeepLense sources sit near z=1.0.")
    if params.z_lens < params.z_source:
        ok("redshift_order")
    else:
        fail("redshift_order", f"z_lens={params.z_lens} must be strictly less than "
             f"z_source={params.z_source} — the source must sit behind the lens.")

    # --- model list / kwargs pairing and profile keys ----------------------
    for label, models, kwargs in (
        ("lens", params.lens_model_list, params.kwargs_lens),
        ("source", params.source_model_list, params.kwargs_source),
    ):
        if len(models) == len(kwargs) and len(models) > 0:
            ok(f"{label}_pairing")
        else:
            fail(f"{label}_pairing",
                 f"{label}_model_list has {len(models)} profiles but kwargs_{label} has "
                 f"{len(kwargs)} dicts — lenstronomy pairs them positionally, one dict "
                 "per profile.")
            continue
        for i, (name, kw) in enumerate(zip(models, kwargs)):
            known = r.profile_params.get(name)
            if known is None:
                fail(f"{label}_profile_{i}_known",
                     f"Unknown {label} profile '{name}' — supported (1.9.2-verified): "
                     f"{sorted(r.profile_params)}.")
                continue
            unknown_keys = sorted(set(kw) - set(known))
            if unknown_keys:
                fail(f"{label}_profile_{i}_keys",
                     f"{name} kwargs contain unknown key(s) {unknown_keys}; 1.9.2 "
                     f"accepts exactly {known}. Check for version-blended names.")
            else:
                ok(f"{label}_profile_{i}_keys")

    # --- Einstein radius ----------------------------------------------------
    thetas = [kw["theta_E"] for kw in params.kwargs_lens if "theta_E" in kw]
    if thetas:
        t = thetas[0]
        if r.theta_e_min <= t <= r.theta_e_max:
            ok("einstein_radius")
        else:
            fail("einstein_radius",
                 f"theta_E={t} arcsec outside [{r.theta_e_min}, {r.theta_e_max}]; "
                 "galaxy-scale lenses are typically ~0.5–3 arcsec (canonical DeepLense "
                 "configuration is ~1 arcsec).")

    # --- ellipticity (lens + source e1/e2) ----------------------------------
    e_bad = [
        (name, k, kw[k])
        for name, kws in (("lens", params.kwargs_lens), ("source", params.kwargs_source))
        for kw in kws
        for k in ("e1", "e2")
        if k in kw and abs(kw[k]) > r.ellipticity_max
    ]
    if any("e1" in kw or "e2" in kw for kw in params.kwargs_lens + params.kwargs_source):
        if e_bad:
            name, k, v = e_bad[0]
            fail("ellipticity",
                 f"{name} {k}={v} exceeds |e|<={r.ellipticity_max} (axis ratio would "
                 "be extreme); use param_util.phi_q2_ellipticity with q>~1/3.")
        else:
            ok("ellipticity")

    # --- Sersic index / radius ----------------------------------------------
    for kw in params.kwargs_source:
        if "n_sersic" in kw:
            n = kw["n_sersic"]
            if r.n_sersic_min <= n <= r.n_sersic_max:
                ok("sersic_index")
            else:
                fail("sersic_index",
                     f"n_sersic={n} outside [{r.n_sersic_min}, {r.n_sersic_max}] "
                     "(1 = exponential disk, 4 = de Vaucouleurs; below ~0.36 the "
                     "profile is numerically invalid in lenstronomy).")
        if "R_sersic" in kw:
            if kw["R_sersic"] > 0:
                ok("sersic_radius")
            else:
                fail("sersic_radius", f"R_sersic={kw['R_sersic']} must be positive (arcsec).")

    # --- data / instrument block ---------------------------------------------
    data = params.kwargs_data
    if r.numpix_min <= data.numPix <= r.numpix_max:
        ok("numpix")
    else:
        fail("numpix", f"numPix={data.numPix} outside [{r.numpix_min}, {r.numpix_max}]; "
             "DeepLense recipes use 150 (Model_I) or 64 (Model_II/III).")
    if r.deltapix_min <= data.deltaPix <= r.deltapix_max:
        ok("deltapix")
    else:
        fail("deltapix", f"deltaPix={data.deltaPix} arcsec outside [{r.deltapix_min}, "
             f"{r.deltapix_max}]; e.g. Model_I uses 0.05, HST 0.08, Euclid 0.101.")
    if data.exposure_time is not None:
        if data.exposure_time > 0:
            ok("exposure_time")
        else:
            fail("exposure_time", f"exposure_time={data.exposure_time} must be positive "
                 "seconds (HST F160W: 5400; Euclid VIS: 565), or omit for noiseless.")
    if data.background_rms is not None:
        if data.background_rms > 0:
            ok("background_rms")
        else:
            fail("background_rms", f"background_rms={data.background_rms} must be positive, "
                 "or omit for noiseless.")

    # --- PSF ------------------------------------------------------------------
    psf = params.kwargs_psf
    if psf.psf_type == "GAUSSIAN":
        if psf.fwhm is not None and psf.fwhm > 0:
            ok("psf_fwhm")
        else:
            fail("psf_fwhm", f"GAUSSIAN PSF requires positive fwhm (got {psf.fwhm}); "
                 "1.9.2 has no 'sigma' argument.")
    if psf.pixel_size is not None:
        if abs(psf.pixel_size - data.deltaPix) < 1e-9:
            ok("psf_pixel_size")
        else:
            fail("psf_pixel_size", f"PSF pixel_size={psf.pixel_size} != grid "
                 f"deltaPix={data.deltaPix}; they must match.")

    # --- DeepLense-recipe masses ----------------------------------------------
    if params.halo_mass is not None:
        if r.halo_mass_min <= params.halo_mass <= r.halo_mass_max:
            ok("halo_mass")
        else:
            fail("halo_mass", f"halo_mass={params.halo_mass:g} M_sun outside "
                 f"[{r.halo_mass_min:g}, {r.halo_mass_max:g}]; canonical DeepLense "
                 "value is 1e12.")
    if params.axion_mass is not None:
        if r.axion_mass_min <= params.axion_mass <= r.axion_mass_max:
            ok("axion_mass")
        else:
            fail("axion_mass", f"axion_mass={params.axion_mass:g} eV outside "
                 f"[{r.axion_mass_min:g}, {r.axion_mass_max:g}] (DeepLenseSim range).")
    if params.vortex_mass is not None:
        if r.vortex_mass_min <= params.vortex_mass <= r.vortex_mass_max:
            ok("vortex_mass")
        else:
            fail("vortex_mass", f"vortex_mass={params.vortex_mass:g} M_sun outside "
                 f"[{r.vortex_mass_min:g}, {r.vortex_mass_max:g}]; canonical value 3e10.")

    # --- target SNR (where computable) ------------------------------------------
    snr = _estimate_snr(params)
    if snr is not None:
        if snr >= r.snr_min:
            ok("snr")
        else:
            fail("snr", f"estimated integrated SNR {snr:.1f} < target {r.snr_min:g} "
                 "(Lucca's threshold); raise source amp (roughly proportionally), "
                 "increase exposure_time, or lower background_rms.")

    return ParamValidationResult(
        passed=all(checks.values()),
        checks=checks,
        messages=messages,
        snr_estimate=snr,
    )
