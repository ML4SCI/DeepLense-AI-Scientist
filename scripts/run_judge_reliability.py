"""E2 — LLM-as-judge ranking stability under repeated identical trials.

Fixes the candidate set and the prompt, varies only sampling: the judge ranks the
same 10 architectures N times. Measures whether the pre-training filter is a
stable function of its input.

Candidates are taken verbatim from a recorded search (default: the multi-seed
pilot's experiment_b run, 10 candidates), so the input is exactly what the judge
saw in production, including the params-per-candidate the prompt exposes.

Reported: pairwise Kendall tau across all N(N-1)/2 ranking pairs, how often each
candidate lands in the top-4 (the cut that actually matters, since only the top-4
are trained), and the full reasoning text for every trial.

    DLENS_LLM=gpt-5.2 .venv/bin/python scripts/run_judge_reliability.py \\
        --out docs/agent_reliability/e2_judge.json --trials 20
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import statistics
import time
from collections import Counter

PILOT = os.path.expanduser(
    "~/Personal/GSoC/deeplense_artifacts/pilot/experiment_b/expB_run0_seed0/metrics.json"
)
TASK = (
    "Classify 150x150 grayscale strong-lensing images into 3 dark-matter "
    "substructure classes (no_sub, cdm, axion). 7200 training images, 1800 validation."
)


def kendall_tau(a: list[int], b: list[int]) -> float:
    """tau over the rank vectors of two orderings of the same items."""
    items = sorted(set(a) & set(b))
    ra = {v: i for i, v in enumerate(a)}
    rb = {v: i for i, v in enumerate(b)}
    c = d = 0
    for x, y in itertools.combinations(items, 2):
        s = (ra[x] - ra[y]) * (rb[x] - rb[y])
        if s > 0:
            c += 1
        elif s < 0:
            d += 1
    n = len(items)
    return (c - d) / (0.5 * n * (n - 1)) if n > 1 else float("nan")


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--model", default=os.environ.get("DLENS_LLM", "gpt-5.2"))
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--source", default=PILOT)
    args = p.parse_args()

    from dlens.agents._architecture_search import ArchitectureJudge, _params_m
    from dlens.agents._models import OpenAIModel
    from dlens.schemas._model_design import ArchitectureSpec

    rec = json.load(open(args.source))
    cands = [ArchitectureSpec.model_validate(c) if "family" in c and "input_shape" in c else c
             for c in rec["candidates"]]
    # The recorded candidate dicts carry name/family/depths/widths/params_m; rebuild
    # specs so the listing is byte-identical in shape to production.
    specs = []
    for c in rec["candidates"]:
        specs.append(ArchitectureSpec.model_validate({
            "name": c["name"], "family": c["family"], "input_shape": [150, 150],
            "channels": 1, "num_classes": 3, "depths": c["depths"], "widths": c["widths"],
            "physics_informed": False, "rationale": "",
        }))
    listing = "\n".join(
        f"[{i}] name={c.name} family={c.family.value} depths={c.depths} "
        f"widths={c.widths} params={_params_m(c)}M" for i, c in enumerate(specs)
    )
    prompt = f"{TASK}\n\nCandidates:\n{listing}\n\nRank all candidate indices."
    judge = ArchitectureJudge(model=OpenAIModel(model_name=args.model))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"=== E2: {len(specs)} fixed candidates, {args.trials} trials, {args.model} ===")
    for i, c in enumerate(specs):
        print(f"   [{i}] {c.name:<44} {_params_m(c)}M")

    out = {"model": args.model, "trials": args.trials, "source": args.source,
           "task": TASK, "candidates": [{"index": i, "name": c.name,
                                         "params_m": _params_m(c)} for i, c in enumerate(specs)],
           "recorded_production_ranking": rec["judge_ranking"], "runs": []}
    rankings = []
    for t in range(args.trials):
        t0 = time.monotonic()
        try:
            v = (await judge.arun(prompt)).output
            ranking = [i for i in v.ranking if 0 <= i < len(specs)]
            row = {"trial": t + 1, "ranking": ranking, "top_k": ranking[: args.top_k],
                   "reasoning": v.reasoning, "seconds": round(time.monotonic() - t0, 1)}
            rankings.append(ranking)
        except Exception as exc:
            row = {"trial": t + 1, "error": f"{type(exc).__name__}: {exc}"[:300]}
        out["runs"].append(row)
        print(f"   trial {t+1:>2}: {row.get('ranking', row.get('error'))}", flush=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)

    taus = [kendall_tau(a, b) for a, b in itertools.combinations(rankings, 2)]
    topk = Counter(i for r in rankings for i in r[: args.top_k])
    first = Counter(r[0] for r in rankings)
    name = {i: c.name for i, c in enumerate(specs)}
    out["summary"] = {
        "n_rankings": len(rankings),
        "n_pairs": len(taus),
        "kendall_tau_mean": round(statistics.mean(taus), 4) if taus else None,
        "kendall_tau_min": round(min(taus), 4) if taus else None,
        "kendall_tau_max": round(max(taus), 4) if taus else None,
        "kendall_tau_stdev": round(statistics.stdev(taus), 4) if len(taus) > 1 else None,
        "identical_rankings": len(set(tuple(r) for r in rankings)),
        "top_k": args.top_k,
        "top_k_frequency": {name[i]: f"{topk[i]}/{len(rankings)}"
                            for i in sorted(topk, key=lambda x: -topk[x])},
        "rank1_frequency": {name[i]: f"{first[i]}/{len(rankings)}"
                            for i in sorted(first, key=lambda x: -first[x])},
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    s = out["summary"]
    print(f"\n  distinct rankings: {s['identical_rankings']} of {s['n_rankings']}")
    print(f"  pairwise Kendall tau over {s['n_pairs']} pairs: mean {s['kendall_tau_mean']} "
          f"(min {s['kendall_tau_min']}, max {s['kendall_tau_max']}, sd {s['kendall_tau_stdev']})")
    print(f"  top-{args.top_k} frequency: {s['top_k_frequency']}")
    print(f"  ranked #1 frequency: {s['rank1_frequency']}")
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
