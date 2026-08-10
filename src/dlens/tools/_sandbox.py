# tools/_sandbox.py
"""Sandbox execution backends for the V2 simulation code-generation agent.

Agreed 2026-07-11: generated lenstronomy code runs in a self-hosted Docker image
(lenstronomy pre-installed) — Pydantic's Monty / code mode can't import native libs.

Convention: the generated script writes its final image as a numpy ``.npy`` file to
the path in the ``DLENS_OUTPUT`` environment variable; the sandbox reads it back for
validation.

Two backends:
* ``DockerSandbox`` — the real, isolated backend (see ../../sandbox/Dockerfile).
* ``LocalSandbox`` — runs the script in a subprocess on the host. **Insecure — for
  OFFLINE TESTS with trusted code only**, never for untrusted LLM output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

_OUTPUT_NAME = "output.npy"


@dataclass
class ExecResult:
    """Result of running a generated script in a sandbox."""

    ok: bool
    # -1 = sentinel: no process return code exists (docker missing, timeout, never ran)
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    output_path: Optional[str] = None  # host path to the produced .npy, if any
    error: Optional[str] = None


@runtime_checkable
class Sandbox(Protocol):
    name: str

    def run(self, code: str, *, timeout: float = 120.0) -> ExecResult:
        ...


class LocalSandbox:
    """Run code in a host subprocess. INSECURE — offline tests / trusted code only."""

    name = "local"

    def run(self, code: str, *, timeout: float = 120.0) -> ExecResult:
        workdir = tempfile.mkdtemp(prefix="dlens_local_sandbox_")
        prog = os.path.join(workdir, "prog.py")
        out = os.path.join(workdir, _OUTPUT_NAME)
        with open(prog, "w") as fh:
            fh.write(code)
        env = {**os.environ, "DLENS_OUTPUT": out}
        try:
            proc = subprocess.run(
                [sys.executable, prog],
                capture_output=True, text=True, timeout=timeout, env=env, cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(ok=False, error=f"timed out after {timeout}s")
        produced = out if os.path.exists(out) else None
        # `ok` means the process exited successfully; whether it produced output is
        # reported separately via `output_path` (validation checks both).
        ok = proc.returncode == 0
        return ExecResult(
            ok=ok, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
            output_path=produced,
            error=None if ok else f"process exited with code {proc.returncode}",
        )


class DockerSandbox:
    """Run code in an isolated lenstronomy container (see sandbox/Dockerfile)."""

    name = "docker"

    def __init__(self, image: str = "dlens-lenstronomy:latest") -> None:
        self.image = image

    def run(self, code: str, *, timeout: float = 120.0) -> ExecResult:
        workdir = tempfile.mkdtemp(prefix="dlens_docker_sandbox_")
        with open(os.path.join(workdir, "prog.py"), "w") as fh:
            fh.write(code)
        host_out = os.path.join(workdir, _OUTPUT_NAME)
        cmd = [
            "docker", "run", "--rm",
            "--network=none", "--memory=2g", "--cpus=1", "--pids-limit=256",
            "-v", f"{workdir}:/work", "-w", "/work",
            "-e", "DLENS_OUTPUT=/work/" + _OUTPUT_NAME,
            self.image, "timeout", str(int(timeout)), "python", "/work/prog.py",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        except FileNotFoundError:
            return ExecResult(ok=False, error="docker not found on PATH")
        except subprocess.TimeoutExpired:
            return ExecResult(ok=False, error=f"timed out after {timeout}s")
        produced = host_out if os.path.exists(host_out) else None
        # `ok` means the container process exited successfully; output existence is
        # reported separately via `output_path` (validation checks both).
        ok = proc.returncode == 0
        return ExecResult(
            ok=ok, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
            output_path=produced,
            error=None if ok else f"docker run exited with code {proc.returncode}",
        )


def get_sandbox(name: str = "docker", *, image: str = "dlens-lenstronomy:latest") -> Sandbox:
    """Return a sandbox backend. Default 'docker' (the real one); 'local' for tests."""
    name = (name or "docker").lower()
    if name == "local":
        return LocalSandbox()
    if name == "docker":
        return DockerSandbox(image=image)
    raise ValueError(f"Unknown sandbox: {name!r} (expected docker | local)")
