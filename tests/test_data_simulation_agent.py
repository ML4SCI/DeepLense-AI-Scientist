"""Human-in-the-loop behavior of DataSimulationAgent, fully offline.

Uses Pydantic AI's FunctionModel (via dlens.agents.make_scripted_sim_model) + the
mock backend, so no GPU, network, or live LLM is touched. Async agent methods are
driven with asyncio.run() to avoid extra test-only dependencies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic_ai import DeferredToolRequests

from dlens.agents._data_simulation import (
    DataSimulationAgent,
    SimClarification,
    SimReport,
    extract_plan,
)
from dlens.agents._scripted import make_scripted_sim_model
from dlens.schemas import SimModelConfig, SubstructureType
from dlens.tools._sim_backends import MockBackend


def _agent(tmp_path: Path, **kw) -> DataSimulationAgent:
    model = kw.pop("model", None) or make_scripted_sim_model()
    return DataSimulationAgent(
        model=model, backend=MockBackend(seed=3), output_root=str(tmp_path),
        make_preview=False, **kw,
    )


def test_clarification_then_approval_then_report(tmp_path: Path):
    async def run():
        agent = _agent(tmp_path)
        r1 = await agent.arun("Generate some lensing images for me.")
        assert isinstance(r1.output, SimClarification)
        assert r1.output.options

        r2 = await agent.arun("CDM substructure, Euclid model, 4 images.")
        assert isinstance(r2.output, DeferredToolRequests)
        plan = extract_plan(r2.output)
        assert plan is not None
        assert plan.substructure_type == SubstructureType.CDM
        assert plan.model_config_name == SimModelConfig.MODEL_II
        assert agent.last_output is None  # approval is gating execution

        r3 = await agent.approve()
        assert isinstance(r3.output, SimReport)
        assert agent.last_output is not None
        assert agent.last_output.num_generated == 4
        assert agent.last_output.image_shape == (64, 64)

    asyncio.run(run())


def test_plan_approval_is_required_before_running(tmp_path: Path):
    async def run():
        model = make_scripted_sim_model(
            plan={"substructure_type": "no_sub", "model_config_name": "Model_I", "num_images": 2},
            clarify_after=0,
        )
        agent = _agent(tmp_path, model=model)
        turn = await agent.arun("Smooth lens, Model_I, 2 images.")
        assert isinstance(turn.output, DeferredToolRequests)
        assert agent.last_output is None  # paused, not executed
        final = await agent.approve()
        assert isinstance(final.output, SimReport)
        assert agent.last_output is not None and agent.last_output.num_generated == 2

    asyncio.run(run())


def test_reject_does_not_run_simulation(tmp_path: Path):
    async def run():
        agent = _agent(tmp_path)
        await agent.arun("Generate some lensing images for me.")
        turn = await agent.arun("CDM, Euclid, 4 images.")
        assert isinstance(turn.output, DeferredToolRequests)
        await agent.reject("On second thought, not now.")
        assert agent.last_output is None
        assert list(tmp_path.iterdir()) == []  # no output dirs created

    asyncio.run(run())


def test_approve_without_pending_raises(tmp_path: Path):
    async def run():
        agent = _agent(tmp_path)
        with pytest.raises(RuntimeError, match="awaiting approval"):
            await agent.approve()

    asyncio.run(run())


def test_send_while_plan_pending_raises(tmp_path: Path):
    async def run():
        agent = _agent(tmp_path)
        await agent.arun("Generate some lensing images for me.")
        turn = await agent.arun("CDM, Euclid, 4 images.")
        assert isinstance(turn.output, DeferredToolRequests)
        with pytest.raises(RuntimeError, match="awaiting approval"):
            await agent.arun("actually, make it 10 images")

    asyncio.run(run())
