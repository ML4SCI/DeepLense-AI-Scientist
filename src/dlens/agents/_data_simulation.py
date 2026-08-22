# agents/_data_simulation.py
"""DataSimulationAgent — agent #1 of the DLens pipeline (see docs/WORKFLOW_DESIGN.md).

Wraps DeepLenseSim (API-wrapping, not code-generation) to turn a natural-language
request into a validated simulation run, with two human-in-the-loop gates built on
the framework's patterns:

* **Clarification** — a typed output (``SimClarification`` in the output union);
  the agent asks rather than guessing an ambiguous request.
* **Plan approval** — the ``run_simulation`` tool is ``requires_approval=True``,
  so a call returns ``DeferredToolRequests`` and the run resumes only after the
  human approves/rejects (``DeferredToolResults``).

It subclasses :class:`dlens.agents.DLensConversationalAgent` (session memory) and
adds the deferred-tool resume cycle that the base class does not expose.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field
from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.messages import ModelMessage

from dlens.agents._base import BaseAgentConfig, DLensConversationalAgent, OutputSchema
from dlens.config import build_model_from_env
from dlens.prompts._simulation import SIM_SYSTEM_PROMPT
from dlens.schemas import SimConfig, SimOutput
from dlens.tools._sim_backends import SimBackend, get_backend
from dlens.tools._simulation import SimDeps, register_simulation_tool


class SimClarification(OutputSchema):
    """Emitted when the request is too ambiguous to simulate safely (HITL gate 1)."""

    question: str = Field(description="The clarifying question to ask the user.")
    options: list[str] = Field(
        default_factory=list, description="Optional suggested answers to present."
    )


class SimReport(OutputSchema):
    """The agent's final structured answer after a run (or after it was declined)."""

    message: str = Field(description="Human-readable summary of the outcome.")
    run_id: Optional[str] = Field(default=None, description="Run id, if a run completed.")
    num_generated: Optional[int] = Field(default=None, description="Images generated, if any.")
    output_dir: Optional[str] = Field(default=None, description="Output directory, if any.")


class DataSimulationAgent(DLensConversationalAgent):
    """Conversational, human-in-the-loop wrapper around the simulation tool."""

    OUTPUT_TYPES = [SimClarification, SimReport, DeferredToolRequests]

    def __init__(
        self,
        *,
        model: Any | None = None,
        config: BaseAgentConfig | None = None,
        backend: SimBackend | None = None,
        output_root: str = "simulations",
        make_preview: bool = True,
        debug: bool = False,
        retries: int = 2,
    ) -> None:
        if config is None:
            config = BaseAgentConfig(
                name="DataSimulationAgent",
                description=(
                    "Translates natural-language requests into validated DeepLenseSim "
                    "runs, with clarification and plan-approval human-in-the-loop gates."
                ),
                custom_system_prompt=SIM_SYSTEM_PROMPT,
                # Local-first, env-configurable (defaults to Ollama qwen3:8b).
                model=model or build_model_from_env(),
                debug=debug,
            )
        elif model is not None:
            config = config.model_copy(update={"model": model})

        super().__init__(
            config=config,
            deps_type=SimDeps,
            output_type=self.OUTPUT_TYPES,
            register_tools=register_simulation_tool,
            retries=retries,
        )

        self._deps = SimDeps(
            backend=backend or get_backend("auto"),
            output_root=output_root,
            make_preview=make_preview,
        )
        # Per-session stash of the pending approval (request + full history to resume from).
        self._pending: dict[str, tuple[DeferredToolRequests, list[ModelMessage]]] = {}
        # A default session so callers can use the agent without juggling ids.
        self._session: str = self.create_session()

    # -- HITL conversation API --------------------------------------------- #

    async def arun(self, query, *, session_id: Optional[str] = None, deps=None, **kwargs):
        """Send a user message. Returns the raw pydantic-ai result; inspect
        ``result.output`` (``SimClarification`` | ``SimReport`` | ``DeferredToolRequests``)."""
        sid = session_id or self._session
        if sid in self._pending:
            raise RuntimeError(
                "A plan is awaiting approval for this session; call approve() or "
                "reject() before sending a new message."
            )
        result = await super().arun(query, session_id=sid, deps=deps or self._deps, **kwargs)
        if isinstance(result.output, DeferredToolRequests):
            self._pending[sid] = (result.output, result.all_messages())
        return result

    async def approve(self, *, session_id: Optional[str] = None):
        """Approve the pending plan and resume; the simulation runs."""
        return await self._resume(session_id or self._session, approved=True)

    async def reject(
        self,
        reason: str = "The human declined the proposed plan.",
        *,
        session_id: Optional[str] = None,
    ):
        """Reject the pending plan and resume; no simulation runs."""
        return await self._resume(session_id or self._session, approved=False, reason=reason)

    async def _resume(self, session_id: str, *, approved: bool, reason: str = ""):
        if session_id not in self._pending:
            raise RuntimeError(
                "No plan is awaiting approval for this session; call arun() first."
            )
        requests, history = self._pending.pop(session_id)
        results = DeferredToolResults()
        for call in requests.approvals:
            results.approvals[call.tool_call_id] = True if approved else ToolDenied(reason)
        result = await self._agent.run(
            message_history=history, deferred_tool_results=results, deps=self._deps
        )
        self._session_store.add_messages(session_id, result.new_messages())
        return result

    def reset(self) -> None:
        """Clear the default session, pending plan, and last output (handy in tests)."""
        self.clear_session(self._session)
        self._pending.pop(self._session, None)
        self._deps.last_output = None
        self._session = self.create_session()

    @property
    def last_output(self) -> SimOutput | None:
        """The most recent authoritative simulation output, if any."""
        return self._deps.last_output


# --------------------------------------------------------------------------- #
# Plan helpers (for rendering the approval checkpoint)
# --------------------------------------------------------------------------- #


def extract_plan(requests: DeferredToolRequests) -> SimConfig | None:
    """Recover and validate the proposed SimConfig from a pending approval request."""
    for call in requests.approvals:
        if call.tool_name == "run_simulation":
            args = call.args
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return None
            payload = args.get("config", args) if isinstance(args, dict) else args
            try:
                return SimConfig.model_validate(payload)
            except Exception:
                return None
    return None


def format_plan(cfg: SimConfig) -> str:
    """Human-readable plan box for the approval checkpoint."""
    sub = {
        "no_sub": "No substructure (smooth lens)",
        "cdm": "CDM subhalos (cold dark matter)",
        "vortex": f"Axion vortex (axion_mass={cfg.axion_mass} eV, vortex_mass={cfg.vortex_mass:.2e} M_sun)",
    }[cfg.substructure_type.value]
    model = {
        "Model_I": "Model_I (150x150 px, Gaussian PSF)",
        "Model_II": "Model_II (64x64 px, Euclid instrument)",
        "Model_III": "Model_III (64x64 px, HST instrument)",
    }[cfg.model_config_name.value]
    bar = "=" * 52
    return (
        f"{bar}\n  PROPOSED SIMULATION PLAN (awaiting your approval)\n{bar}\n"
        f"  Substructure:  {sub}\n"
        f"  Configuration: {model}\n"
        f"  Num images:    {cfg.num_images}\n"
        f"  Halo mass:     {cfg.halo_mass:.2e} M_sun\n"
        f"  z_halo:        {cfg.z_halo}\n"
        f"  z_source:      {cfg.z_source}\n"
        f"  Cosmology:     H0={cfg.cosmology.H0}, Om0={cfg.cosmology.Om0}, Ob0={cfg.cosmology.Ob0}\n"
        f"{bar}"
    )
