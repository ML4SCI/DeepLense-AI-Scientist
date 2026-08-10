# tools/_sim_runner.py
"""The pure simulation runner: config in, files + structured output out.

LLM-agnostic. ``execute_simulation`` is what the agent's ``run_simulation`` tool
calls, but it can be used directly too. Output layout per run:

    <output_root>/<run_id>/
        <substructure>_0000.npy ... _NNNN.npy
        metadata.json
        preview.png            (best-effort; skipped if matplotlib is unavailable)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dlens.schemas import SimConfig, SimOutput
from dlens.tools._sim_backends import SimBackend, get_backend

# 8-hex-char run ids are convenient to type but not collision-proof; a handful
# of fresh draws is plenty (collisions are ~1-in-4e9 per draw).
_RUN_ID_ATTEMPTS = 5


def execute_simulation(
    config: SimConfig,
    *,
    backend: SimBackend | None = None,
    output_root: str | os.PathLike[str] = "simulations",
    make_preview: bool = True,
) -> SimOutput:
    """Run ``config`` through ``backend`` and persist images + metadata.

    Args:
        config: A validated :class:`SimConfig`.
        backend: Simulation backend; defaults to ``get_backend("auto")``.
        output_root: Base directory under which a per-run folder is created.
        make_preview: If True, write a ``preview.png`` grid (best-effort).

    Returns:
        A :class:`SimOutput` (also serialized to ``metadata.json``).
    """
    backend = backend or get_backend("auto")

    # Short run_ids are convenient but can collide; exist_ok=False makes a
    # collision explicit and we retry with a fresh id rather than silently
    # mixing two runs' outputs in one directory.
    for _ in range(_RUN_ID_ATTEMPTS):
        run_id = uuid.uuid4().hex[:8]
        output_dir = Path(output_root) / run_id
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(
            f"Could not allocate a unique run directory under {output_root!s} "
            f"after {_RUN_ID_ATTEMPTS} attempts."
        )

    images = backend.generate(config)
    if not images:
        raise RuntimeError("Backend returned no images.")

    filenames: list[str] = []
    for i, img in enumerate(images):
        fname = f"{config.substructure_type.value}_{i:04d}.npy"
        np.save(output_dir / fname, img)
        filenames.append(fname)

    all_pixels = np.concatenate([np.asarray(img).ravel() for img in images])
    shape = tuple(int(s) for s in np.asarray(images[0]).shape)

    output = SimOutput(
        run_id=run_id,
        config=config,
        num_generated=len(images),
        image_shape=(shape[0], shape[1]),
        pixel_value_range=(float(all_pixels.min()), float(all_pixels.max())),
        timestamp=datetime.now(timezone.utc).isoformat(),
        output_dir=str(output_dir),
        filenames=filenames,
        backend=backend.name,
    )

    (output_dir / "metadata.json").write_text(
        json.dumps(output.model_dump(mode="json"), indent=2, default=str)
    )

    if make_preview:
        _write_preview(images, output, output_dir / "preview.png")

    return output


def _write_preview(images: list[np.ndarray], output: SimOutput, path: Path) -> None:
    """Write a sqrt-scaled grid preview. Best-effort: never fails the run."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless / no display
        import matplotlib.pyplot as plt
    except Exception:
        return

    try:
        n = min(len(images), 8)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_2d(axes)
        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.axis("off")
            if idx < n:
                ax.imshow(np.sqrt(np.abs(images[idx])), cmap="viridis")
                ax.set_title(f"Image {idx + 1}", fontsize=10)

        cfg = output.config
        sub = cfg.substructure_type.value.replace("_", " ").title()
        fig.suptitle(
            f"{sub} | {cfg.model_config_name.value} | "
            f"z_halo={cfg.z_halo}, z_source={cfg.z_source}\n"
            f"Run {output.run_id} | {output.num_generated} images | "
            f"shape {output.image_shape} | backend={output.backend}",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        return
