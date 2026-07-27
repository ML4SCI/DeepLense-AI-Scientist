"""Paper-2 evaluation harness: grounding ablation + expanded suite.

Runs the code-generation agent (gpt-5.2) against a prompt suite in one or both
arms (grounded = with the lenstronomy 1.9.2 cheat-sheet; ungrounded = identical
prompt minus the cheat-sheet block), executing every attempt in the Docker
sandbox, with a failure taxonomy and per-phase latency. Real runs, nothing mocked.

  Experiment A:  --prompts canonical --arms grounded,ungrounded --repeats 1
  Experiment B:  --prompts expanded  --arms grounded             --repeats 3
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import time

from dlens.agents._models import OpenAIModel
from dlens.agents._simulation_codegen import SimulationCodegenAgent, validate_output
from dlens.schemas._codegen import SimSpec
from dlens.tools._sandbox import get_sandbox

FAILURE_PATTERNS = [
    ("wrong_signature", r"unexpected keyword argument|takes \d+ positional|got multiple values"),
    ("wrong_api_import", r"ImportError|ModuleNotFoundError|cannot import"),
    ("wrong_api_attribute", r"AttributeError|has no attribute"),
    ("timeout", r"timed out"),
    ("syntax", r"SyntaxError"),
]


def categorize(stderr: str, validation_msg: str) -> str:
    blob = f"{stderr}\n{validation_msg}"
    for cat, pat in FAILURE_PATTERNS:
        if re.search(pat, blob):
            return cat
    if "failed checks" in validation_msg:
        return "bad_output"
    return "other"


async def eval_one(agent: SimulationCodegenAgent, prompt: dict, max_retries: int, timeout: float) -> dict:
    spec = SimSpec(description=prompt["description"])
    expected = tuple(prompt["expected_shape"]) if prompt.get("expected_shape") else None
    rec: dict = {"attempts": [], "ok": False}
    last_error = None
    for n in range(1, max_retries + 1):
        att: dict = {"n": n}
        t0 = time.monotonic()
        try:
            program = (await agent.generate(spec, last_error=last_error)).output
            att["schema_ok"] = True
        except Exception as exc:  # structured output failed after pydantic-ai retries
            att.update(schema_ok=False, error=f"schema: {type(exc).__name__}"[:200],
                       gen_s=round(time.monotonic() - t0, 1), category="schema")
            rec["attempts"].append(att)
            last_error = "Previous response was not valid structured output."
            continue
        att["gen_s"] = round(time.monotonic() - t0, 1)
        try:
            ast.parse(program.code)
            att["parses"] = True
        except SyntaxError:
            att["parses"] = False
        t1 = time.monotonic()
        ex = agent.sandbox.run(program.code, timeout=timeout)
        att["exec_s"] = round(time.monotonic() - t1, 1)
        v = validate_output(ex, expected_shape=expected)
        att["ran"], att["validated"] = ex.ok, v.passed
        if not v.passed:
            att["category"] = categorize(ex.stderr or ex.error or "", v.message)
            att["error"] = ((ex.stderr or ex.error or "").strip().splitlines()[-1:] or [v.message])[0][:200]
        rec["attempts"].append(att)
        if v.passed:
            rec.update(ok=True, image_shape=v.image_shape)
            break
        last_error = f"{v.message}\n{ex.stderr or ex.error or ''}"[:2000]
    rec["n_attempts"] = len(rec["attempts"])
    rec["wall_s"] = round(sum(a.get("gen_s", 0) + a.get("exec_s", 0) for a in rec["attempts"]), 1)
    if not rec["ok"]:
        rec["final_category"] = rec["attempts"][-1].get("category", "other")
    return rec


async def main() -> int:
    p = argparse.ArgumentParser(description="Paper-2 evaluation harness")
    p.add_argument("--prompts", choices=["canonical", "expanded"], required=True)
    p.add_argument("--arms", default="grounded", help="Comma list: grounded,ungrounded")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--model", default="gpt-5.2")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.prompts == "canonical":
        from dlens.data.sim_prompts import SYNTHETIC_PROMPTS
        prompts = [{"name": q["name"], "category": "canonical",
                    "description": q["description"], "expected_shape": None}
                   for q in SYNTHETIC_PROMPTS]
    else:
        from dlens.data.eval_prompts import EXPANDED_PROMPTS
        prompts = EXPANDED_PROMPTS

    sandbox = get_sandbox("docker")
    results: dict = {"model": args.model, "max_retries": args.max_retries,
                     "repeats": args.repeats, "arms": {}}
    for arm in [a.strip() for a in args.arms.split(",")]:
        grounded = arm == "grounded"
        agent = SimulationCodegenAgent(
            model=OpenAIModel(model_name=args.model), sandbox=sandbox,
            max_retries=args.max_retries, grounded=grounded,
        )
        print(f"\n===== ARM: {arm} (model={args.model}, prompts={len(prompts)}, "
              f"repeats={args.repeats}) =====", flush=True)
        arm_out: dict = {}
        for prompt in prompts:
            runs = []
            for rep in range(args.repeats):
                rec = await eval_one(agent, prompt, args.max_retries, args.timeout)
                runs.append(rec)
                status = "PASS" if rec["ok"] else f"FAIL({rec.get('final_category')})"
                print(f"  {prompt['name']:<24} rep{rep + 1} {status:<18} "
                      f"attempts={rec['n_attempts']} wall={rec['wall_s']}s", flush=True)
            arm_out[prompt["name"]] = {"category": prompt["category"], "runs": runs}
            results["arms"][arm] = arm_out
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=2)
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
