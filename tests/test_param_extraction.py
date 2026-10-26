"""Two-stage path: parameter schema, deterministic validator, extraction agent,
and the extract -> validate -> codegen wiring — fully offline."""

from __future__ import annotations

import asyncio
import copy
import math

import pytest

from dlens.agents._param_extraction import ExtractedParameters, ParamExtractionAgent
from dlens.agents._scripted_codegen import make_scripted_codegen_model
from dlens.agents._scripted_param_extraction import (
    GOOD_PARAMS,
    make_scripted_extraction_model,
    render_offline_program,
)
from dlens.agents._simulation_codegen import SimulationCodegenAgent
from dlens.agents._two_stage_codegen import TwoStageSimulationAgent, params_to_codegen_notes
from dlens.schemas._codegen import SimSpec
from dlens.schemas._lens_params import LensParameterSet
from dlens.tools._param_validator import ValidationRanges, validate_parameters
from dlens.tools._sandbox import LocalSandbox


def _params(**overrides) -> LensParameterSet:
    payload = copy.deepcopy(GOOD_PARAMS)
    payload.update(overrides)
    return LensParameterSet.model_validate(payload)


# --------------------------------------------------------------------------- #
# Validator: the valid case, then one test per failure class
# --------------------------------------------------------------------------- #


def test_validator_passes_canonical_params():
    res = validate_parameters(_params())
    assert res.passed, res.messages
    assert res.messages == []
    # Inside the real-data acceptance band (measured on real Model_I images).
    assert res.snr_estimate is not None and 14.0 <= res.snr_estimate <= 60.0


def test_validator_redshift_order():
    res = validate_parameters(_params(z_lens=1.5, z_source=1.0))
    assert not res.passed
    assert res.checks["redshift_order"] is False
    assert any("behind the lens" in m for m in res.messages)


def test_validator_einstein_radius():
    bad = copy.deepcopy(GOOD_PARAMS)
    bad["kwargs_lens"][0]["theta_E"] = 40.0
    res = validate_parameters(LensParameterSet.model_validate(bad))
    assert res.checks["einstein_radius"] is False
    assert any("theta_E=40.0" in m for m in res.messages)


def test_validator_ellipticity_bound():
    bad = copy.deepcopy(GOOD_PARAMS)
    bad["kwargs_lens"][0]["e1"] = 0.9
    res = validate_parameters(LensParameterSet.model_validate(bad))
    assert res.checks["ellipticity"] is False
    assert any("e1=0.9" in m for m in res.messages)


def test_validator_sersic_index():
    bad = copy.deepcopy(GOOD_PARAMS)
    bad["kwargs_source"][0]["n_sersic"] = 12.0
    res = validate_parameters(LensParameterSet.model_validate(bad))
    assert res.checks["sersic_index"] is False


def test_validator_exposure_and_background():
    res = validate_parameters(
        _params(kwargs_data={"numPix": 64, "deltaPix": 0.08,
                             "exposure_time": -5.0, "background_rms": 0.0})
    )
    assert res.checks["exposure_time"] is False
    assert res.checks["background_rms"] is False


def test_validator_grid_sanity():
    res = validate_parameters(
        _params(kwargs_data={"numPix": 4, "deltaPix": 3.0,
                             "exposure_time": 5400.0, "background_rms": 0.01})
    )
    assert res.checks["numpix"] is False
    assert res.checks["deltapix"] is False


def test_validator_psf_checks():
    res = validate_parameters(
        _params(kwargs_psf={"psf_type": "GAUSSIAN", "fwhm": None, "pixel_size": 0.05})
    )
    assert res.checks["psf_fwhm"] is False
    assert res.checks["psf_pixel_size"] is False  # 0.05 != deltaPix 0.08
    assert any("no 'sigma'" in m for m in res.messages)


def test_validator_mass_ranges():
    res = validate_parameters(
        _params(halo_mass=1e16, axion_mass=1e-10, vortex_mass=1e6)
    )
    assert res.checks["halo_mass"] is False
    assert res.checks["axion_mass"] is False
    assert res.checks["vortex_mass"] is False


def test_validator_snr_band():
    dim = copy.deepcopy(GOOD_PARAMS)
    dim["kwargs_source"][0]["amp"] = 0.001
    res = validate_parameters(LensParameterSet.model_validate(dim))
    assert res.checks["snr"] is False
    assert res.snr_estimate is not None and res.snr_estimate < 14.0
    assert any("SNR" in m for m in res.messages)

    runaway = copy.deepcopy(GOOD_PARAMS)
    runaway["kwargs_source"][0]["amp"] = 1e5
    res = validate_parameters(LensParameterSet.model_validate(runaway))
    assert res.checks["snr"] is False
    assert res.snr_estimate is not None and res.snr_estimate > 60.0
    assert any("runaway" in m for m in res.messages)


def test_validator_snr_skipped_when_noiseless():
    # No exposure/noise -> SNR not computable -> check omitted, not failed.
    res = validate_parameters(
        _params(kwargs_data={"numPix": 64, "deltaPix": 0.08},
                kwargs_psf={"psf_type": "GAUSSIAN", "fwhm": 0.15, "pixel_size": 0.08})
    )
    assert "snr" not in res.checks
    assert res.snr_estimate is None
    assert res.passed


# lenstronomy 1.9.2 renders of one lens at Model_I settings (150 px at 0.05", Gaussian
# PSF FWHM 0.087", no supersampling, 1778 s, background_rms 0.01, the recipe's noise),
# with amp set so each image measures SNR ~24: the median over 50 noise draws of
# (brightest pixel - mean of the 10 px corners) / sqrt(brightest pixel + their variance),
# the measurement the [14, 60] band comes from. Their amps span 25x, so amp alone
# cannot predict the SNR: it is the brightness at R_sersic for a Sersic profile (the
# centre is exp(b_n) brighter, then blurred) and the total flux for a Gaussian.
_SNR_REFERENCES = [
    ("SERSIC_ELLIPSE", {"amp": 76.0, "R_sersic": 0.3, "n_sersic": 0.5}, 25.3),
    ("SERSIC_ELLIPSE", {"amp": 32.0, "R_sersic": 0.3, "n_sersic": 1.0}, 24.2),
    ("SERSIC_ELLIPSE", {"amp": 10.0, "R_sersic": 0.3, "n_sersic": 2.0}, 23.3),
    ("SERSIC_ELLIPSE", {"amp": 3.1, "R_sersic": 0.3, "n_sersic": 4.0}, 23.6),
    ("GAUSSIAN", {"amp": 9.9, "sigma": 0.1}, 24.5),
]


@pytest.mark.parametrize("source,kwargs,measured", _SNR_REFERENCES)
def test_validator_snr_tracks_the_source_profile(source, kwargs, measured):
    shape = {"e1": -0.05, "e2": 0.05} if source == "SERSIC_ELLIPSE" else {}
    params = LensParameterSet(
        z_lens=0.5, z_source=1.0, lens_model_list=["SIE"],
        kwargs_lens=[{"theta_E": 1.2, "e1": 0.1, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}],
        source_model_list=[source],
        kwargs_source=[{**kwargs, **shape, "center_x": 0.12, "center_y": 0.05}],
        kwargs_data={"numPix": 150, "deltaPix": 0.05, "exposure_time": 1778.0, "background_rms": 0.01},
        kwargs_psf={"psf_type": "GAUSSIAN", "fwhm": 0.087, "pixel_size": 0.05},
    )
    res = validate_parameters(params)
    assert res.checks["snr"] is True, res.messages
    assert res.snr_estimate == pytest.approx(measured, rel=0.15)


def test_validator_snr_negative_amp_fails_instead_of_crashing():
    bad = copy.deepcopy(GOOD_PARAMS)
    bad["kwargs_source"][0]["amp"] = -100.0
    res = validate_parameters(LensParameterSet.model_validate(bad))
    assert res.checks["snr"] is False
    assert any("raise source amp" in m for m in res.messages)


def test_blurred_centre_matches_the_gaussian_closed_form():
    from dlens.tools._param_validator import _blurred_centre

    # A Gaussian of width s blurred by widths (sr, st) peaks at s^2 / sqrt((s^2+sr^2)(s^2+st^2)).
    for s, sr, st in ((0.1, 0.04, 0.04), (0.05, 0.08, 0.002), (0.3, 0.02, 0.01)):
        exact = s * s / math.sqrt((s * s + sr * sr) * (s * s + st * st))
        assert _blurred_centre(lambda r: math.exp(-r * r / (2 * s * s)), sr, st) == pytest.approx(exact, rel=1e-4)


def test_validator_unknown_profile_and_keys():
    bad = copy.deepcopy(GOOD_PARAMS)
    bad["lens_model_list"] = ["SIE_FANCY"]
    res = validate_parameters(LensParameterSet.model_validate(bad))
    assert res.checks["lens_profile_0_known"] is False

    blended = copy.deepcopy(GOOD_PARAMS)
    blended["kwargs_source"][0]["sigma"] = 0.2  # version-blended key on SERSIC_ELLIPSE
    res = validate_parameters(LensParameterSet.model_validate(blended))
    assert res.checks["source_profile_0_keys"] is False
    assert any("version-blended" in m for m in res.messages)


def test_validator_pairing_mismatch():
    bad = copy.deepcopy(GOOD_PARAMS)
    bad["lens_model_list"] = ["SIE", "SHEAR"]  # kwargs_lens still has one dict
    res = validate_parameters(LensParameterSet.model_validate(bad))
    assert res.checks["lens_pairing"] is False
    assert any("positionally" in m for m in res.messages)


def test_validator_ranges_configurable():
    tight = ValidationRanges(theta_e_max=0.5)
    res = validate_parameters(_params(), tight)  # canonical theta_E=1.0 now fails
    assert res.checks["einstein_radius"] is False


# --------------------------------------------------------------------------- #
# Extraction agent (scripted model) + two-stage wiring
# --------------------------------------------------------------------------- #


def test_extraction_agent_returns_typed_params():
    agent = ParamExtractionAgent(model=make_scripted_extraction_model())
    out = asyncio.run(agent.extract("canonical DeepLense configuration")).output
    assert isinstance(out, ExtractedParameters)
    assert isinstance(out.params, LensParameterSet)
    assert out.params.lens_model_list == ["SIE"]


def test_two_stage_happy_path():
    agent = TwoStageSimulationAgent(
        extractor=ParamExtractionAgent(model=make_scripted_extraction_model()),
        codegen=SimulationCodegenAgent(
            model=make_scripted_codegen_model(code=render_offline_program(GOOD_PARAMS)),
            sandbox=LocalSandbox(),
        ),
    )
    res = asyncio.run(agent.run(SimSpec(description="canonical lens")))
    assert res.ok
    assert res.param_validation.passed and res.extraction_attempts == 1
    assert res.codegen is not None and res.codegen.ok
    # The generated script verifiably used the validated parameters.
    assert res.code_param_comparison is not None and res.code_param_comparison.passed
    # The validated parameters are injected verbatim into the codegen spec seam.
    assert "theta_E" in params_to_codegen_notes(res.params)


def test_two_stage_retries_extraction_on_validation_failure():
    agent = TwoStageSimulationAgent(
        extractor=ParamExtractionAgent(model=make_scripted_extraction_model(fail_first=True)),
        codegen=SimulationCodegenAgent(
            model=make_scripted_codegen_model(code=render_offline_program(GOOD_PARAMS)),
            sandbox=LocalSandbox(),
        ),
    )
    res = asyncio.run(agent.run(SimSpec(description="x")))
    assert res.ok and res.extraction_attempts == 2


def test_two_stage_gives_up_after_bounded_retries():
    always_bad = {**copy.deepcopy(GOOD_PARAMS), "z_lens": 2.0, "z_source": 1.0}
    agent = TwoStageSimulationAgent(
        extractor=ParamExtractionAgent(model=make_scripted_extraction_model(always_bad)),
        codegen=SimulationCodegenAgent(
            model=make_scripted_codegen_model(), sandbox=LocalSandbox()
        ),
        max_extraction_retries=2,
    )
    res = asyncio.run(agent.run(SimSpec(description="x")))
    assert not res.ok
    assert res.extraction_attempts == 2
    assert res.codegen is None  # codegen never ran on unvalidated parameters


def test_validator_checks_every_lens_profile_einstein_radius():
    # Only the first profile with a theta_E used to be checked, so a bad
    # second profile passed.
    good = GOOD_PARAMS["kwargs_lens"][0]
    params = copy.deepcopy(GOOD_PARAMS)
    params["lens_model_list"] = ["SIE", "SIE"]
    params["kwargs_lens"] = [dict(good), {**good, "theta_E": 40.0}]
    res = validate_parameters(LensParameterSet.model_validate(params))
    assert res.passed is False
    assert res.checks["einstein_radius"] is False
    assert any("kwargs_lens[1]" in m and "theta_E=40.0" in m for m in res.messages)


def test_validator_requires_each_profiles_required_keys():
    # Only unknown keys used to be rejected, so an SIE with no theta_E passed;
    # lenstronomy then raises a TypeError when it renders.
    no_theta = copy.deepcopy(GOOD_PARAMS)
    del no_theta["kwargs_lens"][0]["theta_E"]
    res = validate_parameters(LensParameterSet.model_validate(no_theta))
    assert res.passed is False
    assert res.checks["lens_profile_0_keys"] is False
    assert any("missing required key(s) ['theta_E']" in m for m in res.messages)

    no_gamma2 = copy.deepcopy(GOOD_PARAMS)
    no_gamma2["lens_model_list"] = ["SIE", "SHEAR"]
    no_gamma2["kwargs_lens"].append({"gamma1": 0.02})
    res = validate_parameters(LensParameterSet.model_validate(no_gamma2))
    assert res.checks["lens_profile_1_keys"] is False

    # Centres and the shear origin default to 0 in lenstronomy, so they stay optional.
    centred = copy.deepcopy(GOOD_PARAMS)
    centred["lens_model_list"] = ["SIE", "SHEAR"]
    centred["kwargs_lens"].append({"gamma1": 0.02, "gamma2": 0.0})
    for kw in centred["kwargs_lens"] + centred["kwargs_source"]:
        kw.pop("center_x", None)
        kw.pop("center_y", None)
    res = validate_parameters(LensParameterSet.model_validate(centred))
    assert res.passed, res.messages


def test_required_keys_are_all_named_in_the_extraction_prompt():
    # The stricter key check must never reject a profile written the way the
    # extraction prompt tells the agent to write it.
    import re

    from dlens.prompts._param_extraction import PARAM_EXTRACTION_SYSTEM_PROMPT
    from dlens.tools._param_validator import PROFILE_PARAMS, REQUIRED_PARAMS

    # Prompt lines read "SIE: theta_E, e1, e2, center_x, center_y"; a trailing
    # "(optionally ...)" lists keys the agent may leave out.
    prompted = {
        m.group(1): {k.strip() for k in m.group(2).split(",") if k.strip()}
        for m in re.finditer(r"\b([A-Z_]+): ([A-Za-z0-9_, ]+)", PARAM_EXTRACTION_SYSTEM_PROMPT)
        if m.group(1) in PROFILE_PARAMS
    }
    assert set(prompted) == set(PROFILE_PARAMS)
    for name, required in REQUIRED_PARAMS.items():
        assert set(required) <= prompted[name], name
        assert set(required) <= set(PROFILE_PARAMS[name]), name
