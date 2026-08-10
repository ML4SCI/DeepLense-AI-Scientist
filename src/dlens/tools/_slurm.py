# tools/_slurm.py
"""Slurm job-script generation for validated simulation programs.

Scope item from Michael (he currently hand-holds cluster submission): given a
validated parameter set + generated program, emit a ready-to-submit sbatch
script. Cluster specifics (partition, modules, env activation) live in
``SlurmJobConfig`` — nothing site-specific is hardcoded.

DELIBERATELY NEVER SUBMITS. There is no submit function in this module: the
workflow is generate-and-show (``write_job``), plus a dry-run check
(``check_sbatch`` — ``sbatch --test-only`` where sbatch exists, otherwise a
structural check). Actual submission is a human's decision on the cluster.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from pydantic import BaseModel, Field

_PROGRAM_NAME = "prog.py"
_SCRIPT_NAME = "job.sbatch"


class SlurmJobConfig(BaseModel):
    """Cluster-specific knobs. Defaults are deliberately generic — configure
    per site rather than editing the template."""

    job_name: str = Field(default="dlens_sim", description="Slurm job name.")
    partition: Optional[str] = Field(
        default=None, description="Partition/queue; omitted from the script when None."
    )
    time: str = Field(default="01:00:00", description="Wall time (HH:MM:SS).")
    cpus_per_task: int = Field(default=1, ge=1, description="CPUs per task.")
    mem: str = Field(default="4G", description="Memory per node, Slurm syntax (e.g. 4G).")
    array_size: Optional[int] = Field(
        default=None, ge=1,
        description="N > 1 emits --array=0-(N-1) for multi-image runs; each task "
        "writes its own DLENS_OUTPUT file.",
    )
    modules: list[str] = Field(
        default_factory=list, description="`module load` lines (e.g. ['python/3.10'])."
    )
    env_setup: str = Field(
        default="",
        description="Environment activation line(s), e.g. 'source ~/venvs/lens/bin/activate'.",
    )
    output_dir: str = Field(
        default="dlens_out", description="Directory for images and Slurm logs."
    )
    extra_directives: list[str] = Field(
        default_factory=list,
        description="Raw additional #SBATCH directives (e.g. '--gres=gpu:1').",
    )


def render_sbatch(config: SlurmJobConfig, *, program: str = _PROGRAM_NAME) -> str:
    """Render the sbatch script text for a generated program file."""
    is_array = config.array_size is not None and config.array_size > 1
    log_stem = f"{config.output_dir}/%x_%A_%a" if is_array else f"{config.output_dir}/%x_%j"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={config.job_name}",
        f"#SBATCH --time={config.time}",
        f"#SBATCH --cpus-per-task={config.cpus_per_task}",
        f"#SBATCH --mem={config.mem}",
        f"#SBATCH --output={log_stem}.out",
        f"#SBATCH --error={log_stem}.err",
    ]
    if config.partition:
        lines.append(f"#SBATCH --partition={config.partition}")
    if is_array:
        lines.append(f"#SBATCH --array=0-{config.array_size - 1}")
    for directive in config.extra_directives:
        lines.append(f"#SBATCH {directive}")

    lines.append("")
    lines.extend(f"module load {m}" for m in config.modules)
    if config.env_setup:
        lines.append(config.env_setup)
    lines.append("")
    lines.append(f'mkdir -p "{config.output_dir}"')
    if is_array:
        lines.append(
            f'export DLENS_OUTPUT="{config.output_dir}/image_${{SLURM_ARRAY_TASK_ID}}.npy"'
        )
    else:
        lines.append(f'export DLENS_OUTPUT="{config.output_dir}/image.npy"')
    lines.append(f"python {program}")
    return "\n".join(lines) + "\n"


def write_job(workdir: str, code: str, config: SlurmJobConfig) -> tuple[str, str]:
    """Write the generated program + its sbatch script into ``workdir``.

    Generate-and-show only: returns (program_path, script_path); nothing is
    submitted.
    """
    os.makedirs(workdir, exist_ok=True)
    program_path = os.path.join(workdir, _PROGRAM_NAME)
    script_path = os.path.join(workdir, _SCRIPT_NAME)
    with open(program_path, "w") as fh:
        fh.write(code)
    with open(script_path, "w") as fh:
        fh.write(render_sbatch(config))
    return program_path, script_path


def check_sbatch(script_path: str) -> tuple[bool, list[str]]:
    """Dry-run validation of an sbatch script — never submits.

    Uses ``sbatch --test-only`` when sbatch is on PATH (validates against the
    real cluster config without queueing); otherwise falls back to a structural
    check of the script text.
    """
    if shutil.which("sbatch"):
        proc = subprocess.run(
            ["sbatch", "--test-only", script_path], capture_output=True, text=True
        )
        # --test-only reports the would-be schedule on stderr and exits 0.
        detail = (proc.stderr or proc.stdout).strip()
        return proc.returncode == 0, [detail] if detail else []

    problems: list[str] = []
    with open(script_path) as fh:
        text = fh.read()
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#!"):
        problems.append("missing shebang on line 1")
    directives = [ln for ln in lines if ln.startswith("#SBATCH ")]
    if not directives:
        problems.append("no #SBATCH directives found")
    for required in ("--job-name=", "--time=", "--output="):
        if not any(required in d for d in directives):
            problems.append(f"missing #SBATCH {required.rstrip('=')} directive")
    first_command = next(
        (i for i, ln in enumerate(lines)
         if ln.strip() and not ln.startswith("#")), None
    )
    if first_command is not None and any(
        ln.startswith("#SBATCH ") for ln in lines[first_command:]
    ):
        problems.append("#SBATCH directives after the first command are ignored by Slurm")
    if "DLENS_OUTPUT" not in text:
        problems.append("script does not set DLENS_OUTPUT (the program's output convention)")
    return not problems, problems
