"""Two-stage path evaluation harness with PERSISTED artifacts.

Runs NL -> parameter extraction -> deterministic validation -> code generation ->
Docker sandbox -> AST binding verification over a list of prompts, and writes
everything needed to reconstruct each run to disk. Real runs, nothing mocked
(unless --offline, which uses the scripted models for a plumbing check).

The existing scripts/simulation_codegen_demo.py --two-stage runs a SINGLE prompt
and only prints; this harness exists because those printed results were lost.
Per prompt it persists:

  * the extracted + validated LensParameterSet
  * every parameter-validation check and its outcome, plus the SNR estimate
  * the full generated program (also written as a standalone .py)
  * the sandbox result (exit status, image shape, validation checks)
  * the complete field-by-field AST comparison: matched / diverged / missing /
    unresolved, with expected-vs-actual for every divergence
  * extraction attempts and codegen passes, and per-phase wall time

    DLENS_LLM=gpt-5.2 OPENAI_API_KEY=... .venv/bin/python scripts/run_two_stage_eval.py \\
        --out-dir docs/two_stage --prompts Model_I_no_sub,var_redshifts_1,var_hst_cdm_z

Field counts are NOT comparable across prompts unless the extracted parameter
shape matches: the denominator is 2 model lists + one entry per lens/source
kwargs key + numPix/deltaPix/exposure_time/background_rms/psf_fwhm. A single
SIE lens with a single SERSIC_ELLIPSE source gives 19.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time


def _prompt_table() -> dict[str, dict]:
    """All known prompts by name: the 9 canonical + the 28-prompt expanded suite."""
    from dlens.data.eval_prompts import EXPANDED_PROMPTS
    from dlens.data.sim_prompts import SYNTHETIC_PROMPTS

    table: dict[str, dict] = {}
    for q in SYNTHETIC_PROMPTS:
        table[q["name"]] = {"name": q["name"], "category": "canonical",
                            "description": q["description"]}
    for q in EXPANDED_PROMPTS:
        table.setdefault(q["name"], {"name": q["name"], "category": q["category"],
                                     "description": q["description"]})
    return table


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _comparison_summary(cmp_) -> dict:
    """Counts plus the full lists — the denominator is shape-dependent, so both
    the counts and the field names are recorded."""
    if cmp_ is None:
        return {"ran": False}
    total = len(cmp_.matched) + len(cmp_.diverged) + len(cmp_.missing) + len(cmp_.unresolved)
    return {
        "ran": True,
        "passed": cmp_.passed,
        "n_fields_compared": total,
        "n_matched": len(cmp_.matched),
        "n_diverged": len(cmp_.diverged),
        "n_missing": len(cmp_.missing),
        "n_unresolved": len(cmp_.unresolved),
        "matched": list(cmp_.matched),
        "diverged": [d.model_dump() for d in cmp_.diverged],
        "missing": list(cmp_.missing),
        "unresolved": list(cmp_.unresolved),
        "messages": list(cmp_.messages),
    }


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompts", required=True,
                   help="Comma list of prompt names (see dlens.data.sim_prompts / eval_prompts)")
    p.add_argument("--model", default=os.environ.get("DLENS_LLM", "gpt-5.2"))
    p.add_argument("--max-extraction-retries", type=int, default=3)
    p.add_argument("--max-codegen-passes", type=int, default=2)
    p.add_argument("--max-codegen-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--offline", action="store_true",
                   help="Scripted models + LocalSandbox: plumbing check only, no LLM/Docker.")
    args = p.parse_args()

    from dlens.agents._param_extraction import ParamExtractionAgent
    from dlens.agents._simulation_codegen import SimulationCodegenAgent
    from dlens.agents._two_stage_codegen import TwoStageSimulationAgent
    from dlens.schemas._codegen import SimSpec

    table = _prompt_table()
    names = [n.strip() for n in args.prompts.split(",") if n.strip()]
    unknown = [n for n in names if n not in table]
    if unknown:
        print(f"unknown prompt name(s): {unknown}\nknown: {sorted(table)}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    if args.offline:
        from dlens.agents._scripted_codegen import make_scripted_codegen_model
        from dlens.agents._scripted_param_extraction import (
            GOOD_PARAMS, make_scripted_extraction_model, render_offline_program,
        )
        from dlens.tools._sandbox import LocalSandbox
        sandbox_name = "local(offline)"
        def make_agent() -> TwoStageSimulationAgent:
            return TwoStageSimulationAgent(
                extractor=ParamExtractionAgent(model=make_scripted_extraction_model()),
                codegen=SimulationCodegenAgent(
                    model=make_scripted_codegen_model(code=render_offline_program(GOOD_PARAMS)),
                    sandbox=LocalSandbox(), max_retries=args.max_codegen_retries,
                    timeout=args.timeout),
                max_extraction_retries=args.max_extraction_retries,
                max_codegen_passes=args.max_codegen_passes,
            )
    else:
        from dlens.agents._models import OpenAIModel
        from dlens.tools._sandbox import get_sandbox
        sandbox = get_sandbox("docker")
        sandbox_name = sandbox.name
        def make_agent() -> TwoStageSimulationAgent:
            model = OpenAIModel(model_name=args.model)
            return TwoStageSimulationAgent(
                extractor=ParamExtractionAgent(model=model),
                codegen=SimulationCodegenAgent(
                    model=model, sandbox=sandbox,
                    max_retries=args.max_codegen_retries, timeout=args.timeout),
                max_extraction_retries=args.max_extraction_retries,
                max_codegen_passes=args.max_codegen_passes,
            )

    meta = {
        "model": args.model if not args.offline else "scripted(offline)",
        "sandbox": sandbox_name,
        "max_extraction_retries": args.max_extraction_retries,
        "max_codegen_passes": args.max_codegen_passes,
        "max_codegen_retries": args.max_codegen_retries,
        "timeout_s": args.timeout,
        "git_rev": _git_rev(),
        "platform": platform.platform(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompts": names,
    }
    print(f"=== two-stage eval: {len(names)} prompt(s), model={meta['model']}, "
          f"sandbox={sandbox_name} ===", flush=True)

    per_prompt: dict[str, dict] = {}
    for name in names:
        q = table[name]
        spec = SimSpec(description=q["description"])
        print(f"\n--- {name} ({q['category']}) ---", flush=True)
        agent = make_agent()
        t0 = time.monotonic()
        error = None
        try:
            res = await agent.run(spec)
        except Exception as exc:  # persist the failure rather than losing the run
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR {error}", flush=True)
            per_prompt[name] = {"prompt": q, "error": error,
                               "wall_s": round(time.monotonic() - t0, 1)}
            with open(os.path.join(args.out_dir, "two_stage_eval.json"), "w") as fh:
                json.dump({"meta": meta, "runs": per_prompt}, fh, indent=2)
            continue
        wall = round(time.monotonic() - t0, 1)

        cmp_summary = _comparison_summary(res.code_param_comparison)
        cg = res.codegen
        record = {
            "prompt": q,
            "ok": res.ok,
            "wall_s": wall,
            "extraction_attempts": res.extraction_attempts,
            "codegen_passes": res.codegen_passes,
            "codegen_attempts": (cg.attempts if cg else None),
            "param_validation": res.param_validation.model_dump(),
            "params": (res.params.model_dump(exclude_none=True) if res.params else None),
            "sandbox": {
                "ok": (cg.validation.passed if cg else None),
                "image_shape": (cg.validation.image_shape if cg else None),
                "checks": (cg.validation.checks if cg else None),
                "message": (cg.validation.message if cg else None),
            },
            "code_param_comparison": cmp_summary,
            # full model dump so nothing is lost even if the summary shape changes
            "raw_two_stage_result": res.model_dump(exclude_none=True),
        }
        per_prompt[name] = record

        # Final accepted program, plus EVERY pass including ones L3 rejected —
        # a rejected program is the evidence that L3 catches what L1 does not.
        if cg and cg.code:
            with open(os.path.join(args.out_dir, f"{name}.generated.py"), "w") as fh:
                fh.write(cg.code)
        record["codegen_pass_log"] = [
            {k: v for k, v in rec.model_dump().items() if k != "code"}
            for rec in res.codegen_pass_log
        ]
        for rec in res.codegen_pass_log:
            if rec.code and not rec.accepted:
                path = os.path.join(args.out_dir, f"{name}.pass{rec.pass_index}.REJECTED.py")
                with open(path, "w") as fh:
                    fh.write(rec.code)

        v = res.param_validation
        print(f"  L2 validation : passed={v.passed} checks={len(v.checks)} "
              f"snr={v.snr_estimate if v.snr_estimate is None else round(v.snr_estimate, 1)} "
              f"extraction_attempts={res.extraction_attempts}", flush=True)
        if cg:
            print(f"  L1 sandbox    : passed={cg.validation.passed} "
                  f"shape={cg.validation.image_shape} attempts={cg.attempts}", flush=True)
        if cmp_summary.get("ran"):
            print(f"  L3 binding    : {cmp_summary['n_matched']}/"
                  f"{cmp_summary['n_fields_compared']} matched "
                  f"(diverged {cmp_summary['n_diverged']}, missing "
                  f"{cmp_summary['n_missing']}, unresolved "
                  f"{cmp_summary['n_unresolved']}) passes={res.codegen_passes}", flush=True)
            for d in cmp_summary["diverged"]:
                print(f"     diverged {d['field']}: script={d['actual']!r} "
                      f"validated={d['expected']!r}", flush=True)
            for f in cmp_summary["unresolved"]:
                print(f"     unresolved {f}", flush=True)
        for rec in res.codegen_pass_log:
            c = rec.comparison
            verdict = "ACCEPTED" if rec.accepted else "REJECTED"
            detail = (f"{len(c.matched)} matched, {len(c.diverged)} diverged, "
                      f"{len(c.missing)} missing, {len(c.unresolved)} unresolved"
                      if c else f"L1 failed: {rec.sandbox_message}")
            print(f"     pass {rec.pass_index}: {verdict} — {detail}", flush=True)
            if c and not rec.accepted:
                for d in c.diverged:
                    print(f"        diverged {d.field}: script={d.actual!r} "
                          f"validated={d.expected!r}", flush=True)
                for f in c.missing + c.unresolved:
                    print(f"        unmatched {f}", flush=True)
        print(f"  RESULT        : {'PASSED' if res.ok else 'FAILED'} ({wall}s)", flush=True)

        with open(os.path.join(args.out_dir, "two_stage_eval.json"), "w") as fh:
            json.dump({"meta": meta, "runs": per_prompt}, fh, indent=2)

    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = os.path.join(args.out_dir, "two_stage_eval.json")
    with open(out, "w") as fh:
        json.dump({"meta": meta, "runs": per_prompt}, fh, indent=2)

    print("\n=== summary ===")
    for name, r in per_prompt.items():
        if "error" in r:
            print(f"  {name:<22} ERROR {r['error'][:60]}")
            continue
        c = r["code_param_comparison"]
        matched = f"{c['n_matched']}/{c['n_fields_compared']}" if c.get("ran") else "n/a"
        print(f"  {name:<22} {'PASSED' if r['ok'] else 'FAILED':<7} "
              f"fields {matched:<8} extraction={r['extraction_attempts']} "
              f"codegen_passes={r['codegen_passes']} codegen_attempts={r['codegen_attempts']}")
    print(f"\nartifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
