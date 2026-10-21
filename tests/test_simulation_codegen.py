"""V2 simulation code-generation agent — fully offline (LocalSandbox + scripted model)."""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import tempfile

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
    assert r.ok and r.output
    assert np.load(io.BytesIO(r.output)).shape == (64, 64)


def test_local_sandbox_deletes_its_job_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    LocalSandbox().run(_GOOD)
    LocalSandbox().run(_NO_OUTPUT)
    LocalSandbox().run("import sys; sys.exit(3)\n")
    assert list(tmp_path.iterdir()) == []


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
def test_exit_zero_but_no_output_reports_ran_true():
    # A script that exits 0 but writes nothing must be reported as "ran" with no
    # output — not as a process failure (that wrong signal used to reach the
    # retry prompt).
    r = LocalSandbox().run(_NO_OUTPUT)
    assert r.ok is True and r.returncode == 0 and r.output is None
    v = validate_output(r)
    assert not v.passed
    assert v.checks["ran"] is True
    assert v.checks["produced_output"] is False
    assert "wrote no output" in v.message


def test_nonzero_exit_reports_ran_false():
    r = LocalSandbox().run("import sys; sys.exit(3)\n")
    assert r.ok is False and r.returncode == 3
    v = validate_output(r)
    assert not v.passed
    assert v.checks["ran"] is False
    assert "exited with code 3" in v.message


def test_execresult_returncode_defaults_to_sentinel():
    from dlens.tools._sandbox import ExecResult

    # No subprocess ran (docker missing / timeout): returncode must not look like
    # a successful exit.
    assert ExecResult(ok=False).returncode == -1


def test_docker_job_root_defaults_under_home(monkeypatch, tmp_path):
    from dlens.tools._sandbox import _docker_job_root

    # $HOME is shared with the Docker VM by Colima and Docker Desktop; /var/folders isn't.
    monkeypatch.delenv("DLENS_SANDBOX_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = _docker_job_root()
    assert root == str(tmp_path / ".cache" / "dlens" / "sandbox")
    assert os.path.isdir(root)

    monkeypatch.setenv("DLENS_SANDBOX_DIR", str(tmp_path / "shared"))
    assert _docker_job_root() == str(tmp_path / "shared")


def _fake_docker_run(seen):
    """Stand-in for `docker run`: records the bind-mounted job dir and writes the image."""

    def run(cmd, **kwargs):
        job_dir = cmd[cmd.index("-v") + 1].removesuffix(":/work")
        seen["job_dir"] = job_dir
        seen["prog_written"] = os.path.exists(os.path.join(job_dir, "prog.py"))
        np.save(os.path.join(job_dir, "output.npy"), np.arange(64.0).reshape(8, 8))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return run


def test_docker_sandbox_mounts_a_job_dir_under_the_job_root(monkeypatch, tmp_path):
    from dlens.tools import _sandbox

    monkeypatch.setenv("DLENS_SANDBOX_DIR", str(tmp_path))
    seen = {}
    monkeypatch.setattr(_sandbox.subprocess, "run", _fake_docker_run(seen))
    r = DockerSandbox().run(_GOOD)
    assert r.ok
    assert os.path.dirname(seen["job_dir"]) == str(tmp_path)
    assert seen["prog_written"]
    # The image comes back in memory and the job dir is gone.
    assert np.load(io.BytesIO(r.output)).shape == (8, 8)
    assert validate_output(r).passed
    assert list(tmp_path.iterdir()) == []

