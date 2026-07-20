"""Compare code-generation models on the V2 simulation agent's real workflow.

For each model and each synthetic prompt, drives the agent's own primitives —
generate (LLM) -> run (sandbox) -> validate — recording per-attempt metrics:
structured-output (schema) success, syntactic validity (ast.parse), clean
execution, image validation, retries, and latency. Real LLM calls + real
execution; nothing mocked.

Usage:
    OPENAI_API_KEY=... uv run python scripts/model_comparison_eval.py \
        --models gpt-5.2,gpt-5.6-luna --sandbox docker --out docs/model_comparison_results.json
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import time

from dlens.agents._models import OpenAIModel
from dlens.agents._simulation_codegen import SimulationCodegenAgent, validate_output
from dlens.data import SYNTHETIC_PROMPTS
from dlens.schemas._codegen import SimSpec
from dlens.tools._sandbox import get_sandbox


def build_model(model_id: str):
    """Build the pydantic-ai model for an id.

    Prefix with ``responses:`` to use OpenAI's /v1/responses API — required by some
    newer models (e.g. gpt-5.6-*) which reject function tools on /v1/chat/completions
    unless reasoning is disabled.
    """
    if model_id.startswith("responses:"):
        from pydantic_ai.models.openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_id.removeprefix("responses:"))
    return OpenAIModel(model_name=model_id)


async def eval_prompt(agent: SimulationCodegenAgent, spec: SimSpec, max_retries: int, timeout: float) -> dict:
    rec: dict = {"attempts": [], "ok": False}
    last_error = None
    for n in range(1, max_retries + 1):
        att: dict = {"n": n}
        t0 = time.monotonic()
        try:
            program = (await agent.generate(spec, last_error=last_error)).output
            att["schema_ok"] = True
        except Exception as exc:  # structured output failed even after pydantic-ai retries
            att["schema_ok"] = False
            att["error"] = f"schema: {type(exc).__name__}: {exc}"[:300]
            att["gen_s"] = round(time.monotonic() - t0, 1)
            rec["attempts"].append(att)
            last_error = "Your previous response was not valid structured output. Return the schema exactly."
            continue
        att["gen_s"] = round(time.monotonic() - t0, 1)

        try:
            ast.parse(program.code)
            att["parses"] = True
        except SyntaxError as exc:
            att["parses"] = False
            att["error"] = f"syntax: {exc}"[:300]

        t1 = time.monotonic()
        exec_result = agent.sandbox.run(program.code, timeout=timeout)
        att["exec_s"] = round(time.monotonic() - t1, 1)
        att["ran"] = exec_result.ok
        validation = validate_output(exec_result)
        att["validated"] = validation.passed
        if not validation.passed:
            att["error"] = (validation.message + " | " + (exec_result.stderr or exec_result.error or "").strip().splitlines()[-1:][0] if (exec_result.stderr or exec_result.error) else validation.message)[:300]
        rec["attempts"].append(att)
        if validation.passed:
            rec["ok"] = True
            rec["image_shape"] = validation.image_shape
            break
        last_error = f"{validation.message}\n{exec_result.stderr or exec_result.error or ''}"[:2000]
    rec["n_attempts"] = len(rec["attempts"])
    return rec


async def main() -> int:
    p = argparse.ArgumentParser(description="V2 code-gen model comparison")
    p.add_argument("--models", required=True, help="Comma-separated OpenAI model ids.")
    p.add_argument("--sandbox", default="docker", choices=["docker", "local"])
    p.add_argument("--out", default="docs/model_comparison_results.json")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=240.0)
    args = p.parse_args()

    sandbox = get_sandbox(args.sandbox)
    results: dict = {"sandbox": args.sandbox, "max_retries": args.max_retries, "models": {}}

    for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n===== MODEL: {model_id} =====", flush=True)
        agent = SimulationCodegenAgent(
            model=build_model(model_id), sandbox=sandbox, max_retries=args.max_retries
        )
        per_prompt: dict = {}
        for prompt in SYNTHETIC_PROMPTS:
            t0 = time.monotonic()
            rec = await eval_prompt(agent, SimSpec(description=prompt["description"]), args.max_retries, args.timeout)
            rec["wall_s"] = round(time.monotonic() - t0, 1)
            per_prompt[prompt["name"]] = rec
            status = "PASS" if rec["ok"] else "FAIL"
            print(f"  {prompt['name']:<22} {status}  attempts={rec['n_attempts']}  wall={rec['wall_s']}s", flush=True)
        results["models"][model_id] = per_prompt
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)

    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
