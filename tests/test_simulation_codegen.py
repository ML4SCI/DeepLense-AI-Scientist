"""V2 simulation code-generation agent — fully offline (LocalSandbox + scripted model)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from dlens.agents._scripted_codegen import make_scripted_codegen_model
from dlens.agents._simulation_codegen import SimulationCodegenAgent, validate_output
from dlens.data import SYNTHETIC_PROMPTS
from dlens.schemas._codegen import CodegenResult, SimSpec
from dlens.tools._sandbox import DockerSandbox, LocalSandbox, get_sandbox

_GOOD = (
    "import os\nimport numpy as np\n"
    "np.save(os.environ['DLENS_OUTPUT'], np.arange(64 * 64).reshape(64, 64).astype(float))\n"
)
_NO_OUTPUT = "print('nothing written')\n"


def test_local_sandbox_runs_and_produces_output():
    r = LocalSandbox().run(_GOOD)
    assert r.ok and r.output_path
    assert np.load(r.output_path).shape == (64, 64)


def test_validate_output_pass_and_fail():
    good = validate_output(LocalSandbox().run(_GOOD))
    assert good.passed and good.image_shape == (64, 64)
    assert all(good.checks.values())

    bad = validate_output(LocalSandbox().run(_NO_OUTPUT))
    assert not bad.passed
    assert bad.checks["produced_output"] is False


def test_get_sandbox():
    assert get_sandbox("local").name == "local"
    assert isinstance(get_sandbox("docker"), DockerSandbox)
    with pytest.raises(ValueError):
        get_sandbox("nope")


def test_agent_generate_and_validate_offline():
    agent = SimulationCodegenAgent(model=make_scripted_codegen_model(), sandbox=LocalSandbox())
    res = asyncio.run(agent.generate_and_validate(SimSpec(description="Model I, no substructure")))
    assert isinstance(res, CodegenResult)
    assert res.ok and res.validation.passed
    assert res.attempts == 1
    assert res.validation.image_shape == (64, 64)
    # The model's reasoning is carried through to the final result (framework convention).
    assert res.reasoning == "scripted offline program"


def test_agent_retries_then_succeeds():
    agent = SimulationCodegenAgent(
        model=make_scripted_codegen_model(fail_first=True), sandbox=LocalSandbox(), max_retries=3
    )
    res = asyncio.run(agent.generate_and_validate(SimSpec(description="x")))
    assert res.ok and res.attempts == 2


def test_agent_fails_after_max_retries():
    agent = SimulationCodegenAgent(
        model=make_scripted_codegen_model(code=_NO_OUTPUT), sandbox=LocalSandbox(), max_retries=2
    )
    res = asyncio.run(agent.generate_and_validate(SimSpec(description="x")))
    assert not res.ok and res.attempts == 2 and not res.validation.passed
    assert res.reasoning  # reasoning is carried even when validation fails


def test_synthetic_prompts():
    assert len(SYNTHETIC_PROMPTS) == 9
    for p in SYNTHETIC_PROMPTS:
        assert {"name", "model", "substructure", "description"} <= set(p)
        assert p["model"] in {"Model_I", "Model_II", "Model_III"}
        assert p["substructure"] in {"no_sub", "cdm", "axion"}

def test_validate_output_size_check():
    r = LocalSandbox().run(_GOOD)  # writes a 64x64 image
    ok = validate_output(r, expected_shape=(64, 64))
    assert ok.passed and ok.checks["matches_requested_size"] is True
    bad = validate_output(r, expected_shape=(150, 150))
    assert not bad.passed
    assert bad.checks["matches_requested_size"] is False
    # without an expected shape the check is absent (structural checks only)
    assert "matches_requested_size" not in validate_output(r).checks
