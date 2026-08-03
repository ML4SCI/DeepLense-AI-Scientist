# tools/_simulation.py
"""The ``run_simulation`` agent tool and its dependency object.

Registered with ``requires_approval=True`` so a model's call to it pauses the run
for human sign-off (the plan-approval HITL gate) before any images are generated.
The work is delegated to the LLM-agnostic :func:`dlens.tools._sim_runner.execute_simulation`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from dlens.schemas import SimConfig, SimOutput
from dlens.tools._sim_backends import SimBackend
from dlens.tools._sim_runner import execute_simulation


@dataclass
class SimDeps:
    """Runtime dependencies injected into the simulation agent (and its tool)."""

    backend: SimBackend
    output_root: str = "simulations"
    make_preview: bool = True
    # Populated by the tool so the agent wrapper can surface the authoritative
    # result instead of trusting the model to echo it.
    last_output: SimOutput | None = None


def register_simulation_tool(agent: Agent) -> None:
    """Register ``run_simulation`` (approval-gated) onto ``agent``."""

    @agent.tool(requires_approval=True)
    def run_simulation(ctx: RunContext[SimDeps], config: SimConfig) -> dict:
        """Generate strong gravitational-lensing images with DeepLenseSim.

        Call this once you have a complete, physically valid configuration. The
        run pauses for human approval before this executes, so pass your best
        parameters directly rather than confirming in text first.

        Substructure types: ``no_sub`` (smooth), ``cdm`` (cold-dark-matter
        subhalos), ``vortex`` (axion vortex — ``axion_mass`` in eV is required,
        typically 1e-24 to 1e-22). Model_I -> 150x150 simple PSF; Model_II ->
        64x64 Euclid-realistic; Model_III -> 64x64 HST-realistic. Model_IV is
        NOT supported (it needs real galaxy source images outside this
        pipeline). The source must sit behind the lens
        (``z_source`` > ``z_halo``).

        Args:
            config: The full, validated simulation specification.

        Returns:
            A compact summary of the completed run (run_id, image count, shape,
            pixel range, output directory, backend). Images, ``metadata.json`` and
            a ``preview.png`` are written under the output directory.
        """
        output = execute_simulation(
            config,
            backend=ctx.deps.backend,
            output_root=ctx.deps.output_root,
            make_preview=ctx.deps.make_preview,
        )
        ctx.deps.last_output = output
        return {
            "run_id": output.run_id,
            "num_generated": output.num_generated,
            "image_shape": list(output.image_shape),
            "pixel_value_range": list(output.pixel_value_range),
            "output_dir": output.output_dir,
            "backend": output.backend,
        }
