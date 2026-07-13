"""Demo for the V2 simulation code-generation agent.

Offline (default): scripted model + LocalSandbox (trusted trivial code) — no LLM, no Docker.
    uv run python scripts/simulation_codegen_demo.py

Live: OpenAI gpt-4o-mini + Docker sandbox (build the image first; needs OPENAI_API_KEY):
    docker build -t dlens-lenstronomy:latest sandbox/
    OPENAI_API_KEY=... uv run python scripts/simulation_codegen_demo.py --live
"""

from __future__ import annotations

import argparse
import asyncio

from dlens.agents._scripted_codegen import make_scripted_codegen_model
from dlens.agents._simulation_codegen import SimulationCodegenAgent
from dlens.data import SYNTHETIC_PROMPTS
from dlens.schemas._codegen import SimSpec
from dlens.tools._sandbox import LocalSandbox, get_sandbox


async def main() -> int:
    parser = argparse.ArgumentParser(description="V2 simulation code-generation demo")
    parser.add_argument("--live", action="store_true", help="Use OpenAI + the Docker sandbox.")
    args = parser.parse_args()

    spec = SimSpec(description=SYNTHETIC_PROMPTS[0]["description"])
    print(f"PROMPT ({SYNTHETIC_PROMPTS[0]['name']}):\n  {spec.description}\n")

    if args.live:
        print("Mode: LIVE — OpenAI gpt-4o-mini + Docker sandbox (dlens-lenstronomy:latest)")
        agent = SimulationCodegenAgent(sandbox=get_sandbox("docker"))
    else:
        print("Mode: OFFLINE — scripted model + LocalSandbox (no LLM, no Docker)")
        agent = SimulationCodegenAgent(model=make_scripted_codegen_model(), sandbox=LocalSandbox())

    res = await agent.generate_and_validate(spec)

    print("\n--- generated code ---")
    print(res.code)
    print("--- validation ---")
    print(f"  passed: {res.validation.passed} | attempts: {res.attempts}")
    print(f"  checks: {res.validation.checks}")
    print(f"  image_shape: {res.validation.image_shape}")
    print("\nRESULT:", "PASSED" if res.ok else "FAILED")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
