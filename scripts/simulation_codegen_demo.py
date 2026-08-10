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


async def run_two_stage(args) -> int:
    """The two-stage path (NL -> validated parameters -> code), opt-in via
    --two-stage. The direct path below stays the default and unchanged."""
    from dlens.agents._param_extraction import ParamExtractionAgent
    from dlens.agents._scripted_param_extraction import make_scripted_extraction_model
    from dlens.agents._two_stage_codegen import TwoStageSimulationAgent

    spec = SimSpec(description=SYNTHETIC_PROMPTS[0]["description"])
    print(f"PROMPT ({SYNTHETIC_PROMPTS[0]['name']}):\n  {spec.description}\n")

    if args.live:
        from dlens.agents._models import OpenAIModel

        model = OpenAIModel(model_name=args.model) if args.model else None
        print("Mode: LIVE two-stage — OpenAI + Docker sandbox")
        agent = TwoStageSimulationAgent(
            extractor=ParamExtractionAgent(model=model),
            codegen=SimulationCodegenAgent(model=model, sandbox=get_sandbox("docker")),
        )
    else:
        print("Mode: OFFLINE two-stage — scripted models + LocalSandbox")
        agent = TwoStageSimulationAgent(
            extractor=ParamExtractionAgent(model=make_scripted_extraction_model()),
            codegen=SimulationCodegenAgent(
                model=make_scripted_codegen_model(), sandbox=LocalSandbox()
            ),
        )

    res = await agent.run(spec)
    print("--- extracted parameters ---")
    print(res.params.model_dump_json(indent=2, exclude_none=True) if res.params else "  (none)")
    print("--- parameter validation ---")
    print(f"  passed: {res.param_validation.passed} | extraction attempts: {res.extraction_attempts}")
    for m in res.param_validation.messages:
        print(f"  - {m}")
    if res.codegen:
        print("--- codegen ---")
        print(f"  passed: {res.codegen.validation.passed} | attempts: {res.codegen.attempts}")
    print("\nRESULT:", "PASSED" if res.ok else "FAILED")
    return 0 if res.ok else 1


async def main() -> int:
    parser = argparse.ArgumentParser(description="V2 simulation code-generation demo")
    parser.add_argument("--live", action="store_true", help="Use OpenAI + the Docker sandbox.")
    parser.add_argument(
        "--model", default=None,
        help="OpenAI model for --live (default: the agent's default; gpt-5.2 "
        "recommended for reliable code generation).",
    )
    parser.add_argument(
        "--two-stage", action="store_true",
        help="Opt into the two-stage path (NL -> validated parameters -> code). "
        "The direct NL -> code path remains the default.",
    )
    args = parser.parse_args()

    if args.two_stage:
        return await run_two_stage(args)

    spec = SimSpec(description=SYNTHETIC_PROMPTS[0]["description"])
    print(f"PROMPT ({SYNTHETIC_PROMPTS[0]['name']}):\n  {spec.description}\n")

    if args.live:
        from dlens.agents._models import OpenAIModel

        model = OpenAIModel(model_name=args.model) if args.model else None
        print(f"Mode: LIVE — OpenAI {args.model or 'gpt-4o-mini'} + Docker sandbox (dlens-lenstronomy:latest)")
        agent = SimulationCodegenAgent(model=model, sandbox=get_sandbox("docker"))
    else:
        print("Mode: OFFLINE — scripted model + LocalSandbox (no LLM, no Docker)")
        agent = SimulationCodegenAgent(model=make_scripted_codegen_model(), sandbox=LocalSandbox())

    res = await agent.generate_and_validate(spec)

    print("\n--- agent reasoning ---")
    print(f"  {res.reasoning}")
    print("--- generated code ---")
    print(res.code)
    print("--- validation ---")
    print(f"  passed: {res.validation.passed} | attempts: {res.attempts}")
    print(f"  checks: {res.validation.checks}")
    print(f"  image_shape: {res.validation.image_shape}")
    print("\nRESULT:", "PASSED" if res.ok else "FAILED")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
