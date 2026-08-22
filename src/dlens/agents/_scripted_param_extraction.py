# agents/_scripted_param_extraction.py
"""Scripted FunctionModel for offline tests of the parameter-extraction agent.

Returns a fixed, physically valid parameter set so extract -> validate ->
codegen can be exercised with no LLM. ``fail_first`` makes the first attempt
physically invalid (z_lens > z_source) to test the validation-retry loop —
same pattern as _scripted_codegen.
"""

from __future__ import annotations

from typing import Optional

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

# Canonical DeepLense-style configuration; passes every default validator range.
GOOD_PARAMS: dict = {
    "z_lens": 0.5,
    "z_source": 1.0,
    "lens_model_list": ["SIE"],
    "kwargs_lens": [
        {"theta_E": 1.0, "e1": 0.1, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}
    ],
    "source_model_list": ["SERSIC_ELLIPSE"],
    "kwargs_source": [
        # amp sized so the calibrated peak-SNR estimate (~33) sits inside the
        # real-data acceptance band [14, 60] (see _param_validator._estimate_snr).
        {"amp": 12.0, "R_sersic": 0.3, "n_sersic": 1.5, "e1": 0.1, "e2": 0.0,
         "center_x": 0.05, "center_y": 0.05}
    ],
    "kwargs_data": {"numPix": 64, "deltaPix": 0.08, "exposure_time": 5400.0,
                    "background_rms": 0.01},
    "kwargs_psf": {"psf_type": "GAUSSIAN", "fwhm": 0.15, "pixel_size": 0.08},
    "halo_mass": 1e12,
}

# z_lens > z_source: fails the redshift_order check deterministically.
_BAD_PARAMS: dict = {**GOOD_PARAMS, "z_lens": 1.5, "z_source": 1.0}


def render_offline_program(params: dict) -> str:
    """A runnable no-lenstronomy program that EMBEDS the parameter set.

    Lets offline tests exercise the AST verification loop: the script defines
    the lenstronomy input dicts as literals (so _code_param_check can recover
    them) and writes a small image to DLENS_OUTPUT (so the LocalSandbox run and
    output validation pass) — without importing lenstronomy.
    """
    data = params["kwargs_data"]
    psf = params["kwargs_psf"]
    lines = [
        "import os",
        "import numpy as np",
        f"numPix = {data['numPix']}",
        f"deltaPix = {data['deltaPix']}",
        f"exposure_time = {data['exposure_time']}",
        f"background_rms = {data['background_rms']}",
        f"lens_model_list = {params['lens_model_list']!r}",
        f"kwargs_lens = {params['kwargs_lens']!r}",
        f"source_model_list = {params['source_model_list']!r}",
        f"kwargs_source = {params['kwargs_source']!r}",
        "def PSF(**kwargs):",
        "    return kwargs",
        f"psf_class = PSF(psf_type={psf['psf_type']!r}, fwhm={psf['fwhm']}, "
        "pixel_size=deltaPix)",
        "rng = np.random.default_rng(0)",
        "image = np.abs(rng.normal(size=(numPix, numPix))) + 1.0",
        "np.save(os.environ['DLENS_OUTPUT'], image)",
    ]
    return "\n".join(lines) + "\n"


def make_scripted_extraction_model(
    params: Optional[dict] = None, *, fail_first: bool = False
) -> FunctionModel:
    good = params if params is not None else GOOD_PARAMS
    state = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        payload = _BAD_PARAMS if (fail_first and state["n"] == 1) else good
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"reasoning": "scripted offline parameters", "params": payload},
                )
            ]
        )

    return FunctionModel(respond)
