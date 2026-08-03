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
        # amp sized so the peak-pixel SNR proxy clears Lucca's >=25 target at
        # HST-like exposure/noise (see _param_validator._estimate_peak_snr).
        {"amp": 40.0, "R_sersic": 0.3, "n_sersic": 1.5, "e1": 0.1, "e2": 0.0,
         "center_x": 0.05, "center_y": 0.05}
    ],
    "kwargs_data": {"numPix": 64, "deltaPix": 0.08, "exposure_time": 5400.0,
                    "background_rms": 0.005},
    "kwargs_psf": {"psf_type": "GAUSSIAN", "fwhm": 0.15, "pixel_size": 0.08},
    "halo_mass": 1e12,
}

# z_lens > z_source: fails the redshift_order check deterministically.
_BAD_PARAMS: dict = {**GOOD_PARAMS, "z_lens": 1.5, "z_source": 1.0}


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
