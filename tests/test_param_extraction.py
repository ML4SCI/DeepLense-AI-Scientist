"""Two-stage path: parameter schema, deterministic validator, extraction agent,
and the extract -> validate -> codegen wiring — fully offline."""

from __future__ import annotations

import asyncio
import copy

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
