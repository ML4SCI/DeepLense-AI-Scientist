# tools/_param_validator.py
"""Deterministic physical-plausibility validator for extracted lens parameters.

This is a plain function, not an LLM: the two-stage design's whole point is
that this step is tractable and reproducible. Every range is configurable via
``ValidationRanges`` and every failure produces a specific, actionable message
that gets fed back to the extraction agent's retry.

Range provenance is cited per field below. Sources, strongest first:
  * "recipe <file>:<line>" — the ACTUAL values/draws in DeepLenseSim's code
    (the local mwt5345/DeepLenseSim checkout; lens.py + Model_*/sim_*.py).
  * "measured" — statistics measured on 900-2,400 real Model_I images from
    ~/Personal/GSoC/deeplense_data (generated with the unmodified recipe).
  * "lenstronomy 1.9.2" — introspected library facts (profile param_names,
    ObservationConfig band values; no Roman config exists in 1.9.2 — the
    available classes are Euclid, HST, LSST, DES, ZTF).
  * "Collett 2015 (LensPop)" — verified population statistics used as
    corroboration only (Euclid-discoverable lenses: theta_E ~ 0.66 +/- 0.40").
  * "heuristic" — still a guess; needs Lucca's distributions. OM10 redshift
    medians could not be verified precisely, so redshift guards stay heuristic.
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
    """Configurable bounds. Defaults cite their source; see module docstring
    for the provenance categories."""

    # heuristic guards: the recipe FIXES z_halo=0.5, z_gal=1.0
    # (recipe deeplense/lens.py:37 constructor defaults).
    z_lens_max: float = 5.0
    z_source_max: float = 10.0
    # DERIVED: the recipe's own mass_to_radius (recipe deeplense/lens.py:69-101)
    # maps the halo-mass rails below to theta_E at the recipe redshifts:
    # 1e10 -> 0.128", 1e12 -> 1.281" (canonical), 1e14 -> 12.8".
    # Corroboration: Collett 2015 (LensPop) Euclid-discoverable population has
    # theta_E ~ 0.66 +/- 0.40" — comfortably inside this band.
    theta_e_min: float = 0.12
    theta_e_max: float = 13.0
    # heuristic (|e|<0.5 keeps axis ratio q>~1/3); the recipe fixes lens
    # (e1,e2)=(0.1,0) (recipe deeplense/lens.py:113) and source (-0.1,0.1)
    # (recipe deeplense/lens.py:196) — points, not a range.
    ellipticity_max: float = 0.5
    # floor = lenstronomy Sersic numerical validity (~0.36); upper heuristic.
    # The recipe fixes n_sersic=1 (recipe deeplense/lens.py:196).
    n_sersic_min: float = 0.36
    n_sersic_max: float = 8.0
    # guards around the recipe grids: 150 px (recipe deeplense/lens.py:231)
    # and 64 px (recipe deeplense/lens.py:295).
    numpix_min: int = 16
    numpix_max: int = 1024
    # guards spanning the recipe's 0.05"/px (recipe deeplense/lens.py:232) and
    # lenstronomy 1.9.2 ObservationConfig pixel scales: HST 0.08, Euclid 0.101,
    # LSST 0.2, DES 0.263 (introspected; margins heuristic).
    deltapix_min: float = 0.01
    deltapix_max: float = 0.5
    # heuristic rails around the FIXED recipe halo 1e12 M_sun
    # (recipe Model_I/sim_no_sub.py:13, same in cdm/axion and Model_II/III).
    halo_mass_min: float = 1e10
    halo_mass_max: float = 1e14
    # DERIVED exactly: the recipe draws axion mass 10**U(-24, -22) eV
    # (recipe Model_I/sim_axion.py:12).
    axion_mass_min: float = 1e-24
    axion_mass_max: float = 1e-22
    # heuristic rails around the FIXED recipe vortex 3e10 M_sun
    # (recipe Model_I/sim_axion.py:17).
    vortex_mass_min: float = 1e9
    vortex_mass_max: float = 1e12
    # MEASURED: peak SNR of 900 real Model_I images: median 18.1,
    # IQR [16.3, 20.1], p5 14.8, p95 22.0. snr_min = p5 rounded down (the
    # recipe's own lowest-exposure draws sit exactly there); snr_max is a
    # heuristic runaway-amp guard (~3x measured p95). Note: real recipe images
    # score ~18, slightly BELOW the nominal "SNR ~25" from the meeting.
    snr_min: float = 14.0
    snr_max: float = 60.0
    # MEASURED calibration: peak-pixel estimate x this factor makes canonical
    # recipe parameters reproduce the measured median (18.1), and the recipe's
    # exposure draw 10**U(3,3.5) then predicts [14.8, 21.2] vs measured
    # [p5 14.8, p95 22.0]. Physically: SIE magnification + PSF of the lensed
    # peak. Heuristic for very non-recipe geometries.
    lensing_boost: float = 5.9
    profile_params: dict[str, list[str]] = field(default_factory=lambda: PROFILE_PARAMS)


def _estimate_snr(params: LensParameterSet, boost: float) -> float | None:
    """Peak-pixel SNR of the LENSED image, calibrated against real data.

    peak counts ~ amp * deltaPix^2 * exposure_time * ``boost``, where boost is
    the empirical amplification of the observed peak by SIE lensing (+PSF) —
    calibrated so canonical Model_I recipe parameters reproduce the peak SNR
    measured on 900 real Model_I images (median 18.1; the recipe's exposure
    draw then predicts [14.8, 21.2] vs measured p5/p95 [14.8, 22.0]).

    Definition history, for honesty: v1 was peak-pixel WITHOUT the boost
    (under-predicted ~6x — every live extraction failed); v2 switched to
    aperture-integrated (self-consistent but placed real images at ~305,
    nowhere near the nominal 25 — wrong definition family). Real images
    measure ~18 on peak SNR, which is the same scale as the meeting's "~25"
    target, so peak-with-boost is what the recipe evidently means by SNR.
    """
    data = params.kwargs_data
    if data.exposure_time is None or data.background_rms is None:
        return None
    if data.exposure_time <= 0 or data.background_rms <= 0:
        return None  # covered by their own checks
    amps = [kw["amp"] for kw in params.kwargs_source if "amp" in kw]
    if not amps:
        return None
    peak_counts = max(amps) * data.deltaPix**2 * data.exposure_time * boost
    noise = (peak_counts + (data.background_rms * data.exposure_time) ** 2) ** 0.5
    return peak_counts / noise if noise > 0 else None


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

    # --- SNR in the real-data band (where computable) -----------------------------
    snr = _estimate_snr(params, r.lensing_boost)
    if snr is not None:
        if r.snr_min <= snr <= r.snr_max:
            ok("snr")
        elif snr < r.snr_min:
            fail("snr", f"estimated peak SNR {snr:.1f} < {r.snr_min:g} (real "
                 "DeepLenseSim Model_I images measure median 18, p5 15); raise "
                 "source amp roughly proportionally, increase exposure_time, or "
                 "lower background_rms.")
        else:
            fail("snr", f"estimated peak SNR {snr:.1f} > {r.snr_max:g} — "
                 "implausibly bright vs real DeepLenseSim images (p95 = 22); "
                 "the source amp is likely runaway. Lower it.")

    return ParamValidationResult(
        passed=all(checks.values()),
        checks=checks,
        messages=messages,
        snr_estimate=snr,
    )
