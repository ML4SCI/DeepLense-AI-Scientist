"""AST-based verification that generated code used the validated parameters."""

from __future__ import annotations

import asyncio
import copy

from dlens.agents._param_extraction import ParamExtractionAgent
from dlens.agents._scripted_codegen import make_scripted_codegen_model
from dlens.agents._scripted_param_extraction import (
    GOOD_PARAMS,
    make_scripted_extraction_model,
    render_offline_program,
)
from dlens.agents._simulation_codegen import SimulationCodegenAgent
from dlens.agents._two_stage_codegen import TwoStageSimulationAgent
from dlens.schemas._codegen import SimSpec
from dlens.schemas._lens_params import LensParameterSet
from dlens.tools._code_param_check import compare_code_to_params
from dlens.tools._sandbox import LocalSandbox

_PARAMS = LensParameterSet.model_validate(GOOD_PARAMS)
_MATCHING_PROGRAM = render_offline_program(GOOD_PARAMS)


def test_exact_match_passes():
    comp = compare_code_to_params(_MATCHING_PROGRAM, _PARAMS)
    assert comp.passed, comp.messages
    assert not comp.diverged and not comp.missing and not comp.unresolved
    assert "kwargs_lens[0].theta_E" in comp.matched
    assert "numPix" in comp.matched and "psf_fwhm" in comp.matched


def test_diverged_field_reports_both_values():
    tampered = copy.deepcopy(GOOD_PARAMS)
    tampered["kwargs_lens"] = [dict(tampered["kwargs_lens"][0], theta_E=2.5)]
    comp = compare_code_to_params(render_offline_program(tampered), _PARAMS)
    assert not comp.passed
    d = {c.field: c for c in comp.diverged}
    assert "kwargs_lens[0].theta_E" in d
    assert d["kwargs_lens[0].theta_E"].expected == 1.0
    assert d["kwargs_lens[0].theta_E"].actual == 2.5
    assert any("theta_E" in m and "validated value" in m for m in comp.messages)


def test_missing_block_reported():
    # No kwargs_lens assignment and no PSF call anywhere in the script.
    program = "\n".join(
        line for line in _MATCHING_PROGRAM.splitlines()
        if not line.startswith(("kwargs_lens", "def PSF", "    return kwargs", "psf_class"))
    ) + "\n"
    comp = compare_code_to_params(program, _PARAMS)
    assert not comp.passed
    assert "kwargs_lens" in comp.missing
    assert "psf_fwhm" in comp.missing
    assert any("not found in the script" in m for m in comp.messages)


def test_unresolved_value_does_not_silently_pass():
    program = _MATCHING_PROGRAM.replace(
        "kwargs_lens = [{'theta_E': 1.0,",
        "theta_E = compute_einstein_radius()\nkwargs_lens = [{'theta_E': theta_E,",
    )
    comp = compare_code_to_params(program, _PARAMS)
    assert not comp.passed
    assert "kwargs_lens[0].theta_E" in comp.unresolved
    assert "kwargs_lens[0].theta_E" not in [c.field for c in comp.diverged]
    assert any("cannot be verified" in m for m in comp.messages)


def test_resolution_through_names_and_expressions():
    # Values via symbol references, unary minus, and constant folding must all
    # resolve — these are idioms gpt-5.2 actually produces.
    program = (
        "numPix = 64\n"
        "deltaPix = 0.08\n"
        "t_exp = 5400.0\n"
        "kwargs_data = data_configure_simple(numPix, deltaPix, "
        "exposure_time=t_exp, background_rms=1e-2)\n"
        "lens_model_list = ['SIE']\n"
        "kwargs_lens = [{'theta_E': 1.0, 'e1': 0.1, 'e2': 0.0, "
        "'center_x': 0.0, 'center_y': 0.0}]\n"
        "source_model_list = ['SERSIC_ELLIPSE']\n"
        "kwargs_source = [{'amp': 12.0, 'R_sersic': 0.3, 'n_sersic': 1.5, "
        "'e1': 0.1, 'e2': -0.0, 'center_x': 0.05, 'center_y': 0.05}]\n"
        "psf_class = PSF(psf_type='GAUSSIAN', fwhm=0.15, pixel_size=deltaPix)\n"
    )
    comp = compare_code_to_params(program, _PARAMS)
    assert comp.passed, comp.messages
    assert "exposure_time" in comp.matched and "background_rms" in comp.matched


def test_resolution_through_dict_subscripts():
    # Live gpt-5.2 idiom: literals routed through a kwargs dict, then
    # subscripted at the call site. Must resolve, not report unresolved.
    program = (
        "numPix = 64\n"
        "deltaPix = 0.08\n"
        "cfg = {'exposure_time': 5400.0, 'background_rms': 0.01, 'fwhm': 0.15}\n"
        "kwargs_data = data_configure_simple(numPix, deltaPix, "
        "exposure_time=cfg['exposure_time'], background_rms=cfg['background_rms'])\n"
        "lens_model_list = ['SIE']\n"
        "kwargs_lens = [{'theta_E': 1.0, 'e1': 0.1, 'e2': 0.0, "
        "'center_x': 0.0, 'center_y': 0.0}]\n"
        "source_model_list = ['SERSIC_ELLIPSE']\n"
        "kwargs_source = [{'amp': 12.0, 'R_sersic': 0.3, 'n_sersic': 1.5, "
        "'e1': 0.1, 'e2': -0.0, 'center_x': 0.05, 'center_y': 0.05}]\n"
        "psf_class = PSF(psf_type='GAUSSIAN', fwhm=cfg['fwhm'], pixel_size=deltaPix)\n"
    )
    comp = compare_code_to_params(program, _PARAMS)
    assert comp.passed, comp.messages
    assert not comp.unresolved
    for f in ("exposure_time", "background_rms", "psf_fwhm"):
        assert f in comp.matched


def test_two_stage_regenerates_on_divergence():
    # First codegen pass embeds a wrong theta_E; the AST diff must trigger a
    # second pass, which then matches.
    tampered = copy.deepcopy(GOOD_PARAMS)
    tampered["kwargs_lens"] = [dict(tampered["kwargs_lens"][0], theta_E=9.9)]
    responses = iter([render_offline_program(tampered), _MATCHING_PROGRAM])

    from pydantic_ai import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            info.output_tools[0].name,
            {"reasoning": "scripted", "code": next(responses)},
        )])

    agent = TwoStageSimulationAgent(
        extractor=ParamExtractionAgent(model=make_scripted_extraction_model()),
        codegen=SimulationCodegenAgent(model=FunctionModel(respond), sandbox=LocalSandbox()),
    )
    res = asyncio.run(agent.run(SimSpec(description="x")))
    assert res.ok
    assert res.codegen_passes == 2
    assert res.code_param_comparison is not None and res.code_param_comparison.passed


def test_two_stage_fails_when_divergence_persists():
    tampered = copy.deepcopy(GOOD_PARAMS)
    tampered["kwargs_lens"] = [dict(tampered["kwargs_lens"][0], theta_E=9.9)]
    agent = TwoStageSimulationAgent(
        extractor=ParamExtractionAgent(model=make_scripted_extraction_model()),
        codegen=SimulationCodegenAgent(
            model=make_scripted_codegen_model(code=render_offline_program(tampered)),
            sandbox=LocalSandbox(),
        ),
        max_codegen_passes=2,
    )
    res = asyncio.run(agent.run(SimSpec(description="x")))
    assert not res.ok
    assert res.codegen_passes == 2
    assert res.codegen is not None and res.codegen.ok  # sandbox fine; params diverged
    assert not res.code_param_comparison.passed


def test_assignments_are_read_in_source_order():
    # ast.walk is breadth-first, so a correct kwargs_lens nested in a block used
    # to win over a later, wrong top-level reassignment.
    wrong = [dict(GOOD_PARAMS["kwargs_lens"][0], theta_E=2.5)]
    good_line = f"kwargs_lens = {GOOD_PARAMS['kwargs_lens']!r}\n"
    program = _MATCHING_PROGRAM.replace(
        good_line, f"if True:\n    {good_line}kwargs_lens = {wrong!r}\n"
    )
    assert program != _MATCHING_PROGRAM
    comp = compare_code_to_params(program, _PARAMS)
    assert not comp.passed
    assert {d.field: d.actual for d in comp.diverged}.get("kwargs_lens[0].theta_E") == 2.5


def test_grid_size_is_read_from_keywords_in_either_spelling():
    # Only the first two positional arguments of data_configure_simple were
    # read, so keyword calls were reported missing. 1.9.2 spells the keywords
    # numPix/deltaPix; lenstronomy 1.14.1+ spells them num_pix/delta_pix.
    base = "".join(line + "\n" for line in _MATCHING_PROGRAM.splitlines()
                   if not line.startswith(("numPix", "deltaPix")))
    call = ("kwargs_data = data_configure_simple({n}=64, {d}=0.08, "
            "exposure_time=exposure_time, background_rms=background_rms)\n")
    for n, d in (("numPix", "deltaPix"), ("num_pix", "delta_pix")):
        comp = compare_code_to_params(base + call.format(n=n, d=d), _PARAMS)
        assert comp.passed, (n, comp.messages)
        assert "numPix" in comp.matched and "deltaPix" in comp.matched
        wrong = compare_code_to_params(base + call.format(n=n, d=d).replace("=64", "=128"), _PARAMS)
        assert "numPix" in [c.field for c in wrong.diverged]


def test_the_kwargs_passed_to_image_are_the_ones_checked():
    # kwargs_lens/kwargs_source were read from assignments to those names, so a
    # script could define the validated list, render a different one, and pass.
    wrong = [dict(GOOD_PARAMS["kwargs_lens"][0], theta_E=2.5)]
    base = _MATCHING_PROGRAM + f"kwargs_lens_render = {wrong!r}\n"
    for call in ("image_model.image(kwargs_lens=kwargs_lens_render, kwargs_source=kwargs_source)",
                 "image_model.image(kwargs_lens_render, kwargs_source)"):
        comp = compare_code_to_params(base + f"image = {call}\n", _PARAMS)
        assert not comp.passed, call
        assert {d.field: d.actual for d in comp.diverged}.get("kwargs_lens[0].theta_E") == 2.5

    # The cheat-sheet's own call shape still passes, and a computed argument
    # is reported as unresolved rather than guessed.
    ok = compare_code_to_params(
        _MATCHING_PROGRAM + "image = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)\n",
        _PARAMS)
    assert ok.passed, ok.messages
    computed = compare_code_to_params(
        _MATCHING_PROGRAM + "image = image_model.image(kwargs_lens=make_lens(), kwargs_source=kwargs_source)\n",
        _PARAMS)
    assert "kwargs_lens" in computed.unresolved


# The shape the codegen cheat-sheet asks for: every value reaches lenstronomy
# through the calls that build and render the image.
_IMAGE_MODEL_PROGRAM = """\
import sys
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

numPix = 64
deltaPix = 0.08
kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exposure_time=5400.0, background_rms=0.01)
data_class = ImageData(**kwargs_data)
psf_class = PSF(psf_type='GAUSSIAN', fwhm=0.15, pixel_size=deltaPix)
lens_model_list = ['SIE']
kwargs_lens = [{'theta_E': 1.0, 'e1': 0.1, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}]
source_model_list = ['SERSIC_ELLIPSE']
kwargs_source = [{'amp': 12.0, 'R_sersic': 0.3, 'n_sersic': 1.5, 'e1': 0.1, 'e2': 0.0,
                  'center_x': 0.05, 'center_y': 0.05}]
image_model = ImageModel(data_class, psf_class,
                         lens_model_class=LensModel(lens_model_list=lens_model_list),
                         source_model_class=LightModel(light_model_list=source_model_list),
                         kwargs_numerics={'supersampling_factor': 1})
image = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)
"""
_LENS = "kwargs_lens = [{'theta_E': 1.0"
_WRONG_LENS = "kwargs_lens = [{'theta_E': 2.5, 'e1': 0.1, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}]\n"


def _diverged(comp) -> dict:
    return {c.field: c.actual for c in comp.diverged}


def test_dict_unpacking_is_applied_in_order():
    # {'theta_E': 1.0, **{'theta_E': 2.5}} renders 2.5; the override must not be ignored.
    comp = compare_code_to_params(_IMAGE_MODEL_PROGRAM.replace(
        _LENS, "kwargs_lens = [{'theta_E': 1.0, **{'theta_E': 2.5}"), _PARAMS)
    assert _diverged(comp) == {"kwargs_lens[0].theta_E": 2.5}

    comp = compare_code_to_params(_IMAGE_MODEL_PROGRAM.replace(
        _LENS, "key = pick()\nkwargs_lens = [{key: 2.5, 'theta_E': 1.0"), _PARAMS)
    assert "kwargs_lens[0]" in comp.unresolved

    comp = compare_code_to_params(_IMAGE_MODEL_PROGRAM.replace(
        "kwargs_lens = [{'theta_E': 1.0, 'e1': 0.1, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}]",
        "base = {'e1': 0.1, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}\n"
        "kwargs_lens = [{**base, 'theta_E': 1.0}]"), _PARAMS)
    assert comp.passed, comp.messages
