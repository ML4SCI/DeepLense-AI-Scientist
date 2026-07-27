# agents/_scripted_codegen.py
"""Scripted FunctionModel for offline tests/demos of the code-generation agent.

Returns a fixed, valid program (writes a small 2-D array to DLENS_OUTPUT) so the
generate -> run -> validate loop can be exercised with no LLM and no lenstronomy,
using the LocalSandbox. ``fail_first`` makes the first attempt produce no output so
the retry path can be tested.
"""

from __future__ import annotations

from typing import Optional

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

# A trivial but valid "simulation": writes a finite, non-constant 2-D array.
_GOOD_PROGRAM = (
    "import os\n"
    "import numpy as np\n"
    "rng = np.random.default_rng(0)\n"
    "image = np.abs(rng.normal(size=(64, 64))) + 1.0\n"
    "np.save(os.environ['DLENS_OUTPUT'], image)\n"
)
# Runs cleanly but writes nothing -> validation must fail.
_NO_OUTPUT_PROGRAM = "print('no output written')\n"


def make_scripted_codegen_model(
    code: Optional[str] = None, *, fail_first: bool = False
) -> FunctionModel:
    program = code if code is not None else _GOOD_PROGRAM
    state = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        state["n"] += 1
        prog = _NO_OUTPUT_PROGRAM if (fail_first and state["n"] == 1) else program
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"reasoning": "scripted offline program", "code": prog},
                )
            ]
        )

    return FunctionModel(respond)
