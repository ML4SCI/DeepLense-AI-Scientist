"""End-to-end demo for the Data Simulation Agent (agent #1 of WORKFLOW_DESIGN).

Runs the full flow — clarification, plan approval, and a real simulation through
the (mock) backend — with NO GPU and NO live LLM by default.

Offline (default; scripted model + mock backend):
    uv run python scripts/data_simulation_demo.py

Live against a local Ollama model (qwen3:8b by default):
    uv run python scripts/data_simulation_demo.py --live
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pydantic_ai import DeferredToolRequests

from dlens.agents._data_simulation import (
    DataSimulationAgent,
    SimClarification,
    SimReport,
    extract_plan,
    format_plan,
)
from dlens.agents._scripted import make_scripted_sim_model
from dlens.tools._sim_backends import deeplense_available, get_backend

RULE = "-" * 64


def banner(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


async def drive(agent: DataSimulationAgent, opening: str, answers: list[str]):
    """Drive a conversation: feed clarification answers, then approve the plan."""
    print(f"\n[user] {opening}")
    result = await agent.arun(opening)
    ai = 0
    while isinstance(result.output, SimClarification):
        print(f"[agent asks] {result.output.question}  options={result.output.options}")
        answer = answers[ai] if ai < len(answers) else "Use sensible defaults."
        ai += 1
        print(f"[user] {answer}")
        result = await agent.arun(answer)

    if isinstance(result.output, DeferredToolRequests):
        plan = extract_plan(result.output)
        print(f"[agent proposes a plan]\n{format_plan(plan) if plan else result.output}")
        print("[human-in-the-loop] approving the plan")
        result = await agent.approve()

    if isinstance(result.output, SimReport):
        print(f"[agent] {result.output.message}")
    else:
        print(f"[agent | {type(result.output).__name__}] {result.output}")
    return result


def show_outputs(out) -> None:
    if out is None:
        print("  (no simulation output)")
        return
    print(f"  run_id ........ {out.run_id}")
    print(f"  backend ....... {out.backend}")
    print(f"  images ........ {out.num_generated} x {out.image_shape}")
    print(f"  pixel range ... {out.pixel_value_range}")
    print(f"  output_dir .... {out.output_dir}")
    run_dir = Path(out.output_dir)
    if run_dir.exists():
        print(f"  files ......... {sorted(p.name for p in run_dir.iterdir())}")
        assert (run_dir / "metadata.json").exists(), "metadata.json missing"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Data Simulation Agent demo")
    parser.add_argument(
        "--live", action="store_true",
        help="Use the configured local model (Ollama qwen3:8b) instead of the scripted offline model.",
    )
    args = parser.parse_args()

    banner("Data Simulation Agent — end-to-end demo")
    print(f"  DeepLenseSim importable: {deeplense_available()} "
          f"-> backend = '{get_backend('auto').name}'")

    if args.live:
        print("  Mode: LIVE — calling the configured local model.")
        model = None  # DataSimulationAgent builds OllamaModel(qwen3:8b) by default
    else:
        print("  Mode: OFFLINE — scripted FunctionModel (no LLM, no network).")
        model = make_scripted_sim_model(
            plan={"substructure_type": "cdm", "model_config_name": "Model_II", "num_images": 4}
        )

    # Force the mock backend for a deterministic, dependency-free demo.
    agent = DataSimulationAgent(model=model, backend=get_backend("mock", seed=7), output_root="outputs")

    banner("Ambiguous request -> clarification -> plan approval -> simulation")
    await drive(
        agent,
        opening="Generate some strong gravitational lensing images for me.",
        answers=["Use CDM substructure with the Euclid (Model_II) configuration, 4 images."],
    )

    banner("Structured SimOutput (authoritative)")
    show_outputs(agent.last_output)

    ok = agent.last_output is not None and agent.last_output.num_generated == 4
    banner("RESULT")
    print("  PASSED" if ok else "  FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
