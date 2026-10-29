"""Defensive provenance capture for training results."""

from __future__ import annotations

import platform
import re
import subprocess

from dlens.schemas._downstream import ProvenanceRecord
from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _git_commit() -> str | None:
    """Return the current commit hash, or None outside a Git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip().lower()
    return commit if _COMMIT_RE.fullmatch(commit) else None


def _torch_version() -> str | None:
    """Return the installed PyTorch version without making it a requirement."""
    try:
        import torch
    except (ImportError, OSError):
        return None
    version = getattr(torch, "__version__", None)
    return version if isinstance(version, str) else None


def make_provenance(
    *,
    backend: str,
    device: str,
    seed: int | None,
    architecture: ArchitectureSpec,
    training_config: TrainingConfig,
) -> ProvenanceRecord:
    """Capture reproducibility metadata without recording process secrets."""
    return ProvenanceRecord(
        git_commit=_git_commit(),
        python_version=platform.python_version(),
        pytorch_version=_torch_version(),
        device=device,
        seed=seed,
        backend=backend,
        architecture=architecture,
        training_config=training_config,
    )
