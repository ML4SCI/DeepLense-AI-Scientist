"""Slurm sbatch generation — offline; nothing is ever submitted."""

from __future__ import annotations

from pathlib import Path

from dlens.tools import _slurm
from dlens.tools._slurm import SlurmJobConfig, check_sbatch, render_sbatch, write_job


def test_render_contains_config_not_hardcoded_defaults():
    cfg = SlurmJobConfig(
        job_name="lens_batch", partition="physics", time="02:30:00",
        cpus_per_task=4, mem="16G",
        modules=["python/3.10", "cuda/12.1"],
        env_setup="source ~/venvs/lens/bin/activate",
        output_dir="results/run1",
        extra_directives=["--gres=gpu:1"],
    )
    script = render_sbatch(cfg)
    assert script.startswith("#!/bin/bash\n")
    for expected in (
        "#SBATCH --job-name=lens_batch", "#SBATCH --partition=physics",
        "#SBATCH --time=02:30:00", "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=16G", "#SBATCH --gres=gpu:1",
        "module load python/3.10", "module load cuda/12.1",
        "source ~/venvs/lens/bin/activate",
        'export DLENS_OUTPUT="results/run1/image.npy"',
    ):
        assert expected in script, expected
    # No partition directive appears when unset (config-driven, not hardcoded).
    assert "--partition" not in render_sbatch(SlurmJobConfig())


def test_render_array_mode_gives_per_task_outputs():
    script = render_sbatch(SlurmJobConfig(array_size=50))
    assert "#SBATCH --array=0-49" in script
    assert "image_${SLURM_ARRAY_TASK_ID}.npy" in script
    # Single-job mode has neither.
    single = render_sbatch(SlurmJobConfig())
    assert "--array" not in single and "SLURM_ARRAY_TASK_ID" not in single


def test_write_job_generates_and_never_submits(tmp_path: Path, monkeypatch):
    # Force the structural-check path even if a real sbatch exists somewhere.
    monkeypatch.setattr(_slurm.shutil, "which", lambda _: None)

    prog, script = write_job(str(tmp_path), "print('hi')\n", SlurmJobConfig())
    assert Path(prog).read_text() == "print('hi')\n"
    ok, problems = check_sbatch(script)
    assert ok, problems
    # The module exposes no submission entry point at all.
    assert not hasattr(_slurm, "submit")
    assert "sbatch " not in Path(script).read_text()  # script doesn't self-submit


def test_check_sbatch_structural_failures(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(_slurm.shutil, "which", lambda _: None)
    bad = tmp_path / "bad.sbatch"
    bad.write_text("echo no directives\n")
    ok, problems = check_sbatch(str(bad))
    assert not ok
    assert any("shebang" in p for p in problems)
    assert any("#SBATCH" in p for p in problems)
