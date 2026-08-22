"""E1 — planner decision reliability under repeated identical trials.

Holds the observation FIXED and varies only LLM sampling: the planner is asked
to decide from the same ExperimentState N times. Measures what the planner's
decision function actually is, rather than what one run happened to produce.

Four probe states. Provenance is recorded per state and printed in the output,
because two are real recorded observations and two are assembled from recorded
metrics:

  overfit    VERBATIM RECORDED  paper_run resnet34_ref iteration 0
  healthy    VERBATIM RECORDED  paper_run resnet34_ref iteration 1
  underfit   CONSTRUCTED PROBE  real confusion/AUC from ai_scientist_run
                                iteration 0; train set just above val so the
                                signature is "low everywhere, no gap"
  collapsed  CONSTRUCTED PROBE  train/val from the loop-guard regression
                                (chance level for 3 classes); confusion built
                                to be consistent with chance

Every raw PlannerDecision is saved. The scoring rules below are a convenience
for the write-up, not ground truth — the rationale text is kept so the
classification can be re-read by hand.

    DLENS_LLM=gpt-5.2 .venv/bin/python scripts/run_planner_reliability.py \\
        --out docs/agent_reliability/e1_planner.json --trials 20
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import time
from collections import Counter

BASE_STATE = "docs/paper_run.json"


def _base_run_state() -> dict:
    """A real serialized ExperimentState to use as the template."""
    d = json.load(open(BASE_STATE))
    return copy.deepcopy(d["arms"]["resnet34_ref"]["trajectory"])


def _set_metrics(state: dict, *, train_acc, val_acc, auc, confusion, per_class=None) -> dict:
    """Overwrite the single run's metric block; drop any recorded decision so
    the planner is never primed with the answer."""
    st = copy.deepcopy(state)
    st["runs"] = st["runs"][:1]
    st["current_iteration"] = 0
    r = st["runs"][0]
    r["planner_decision"] = None
    # The recorded JSON has the prediction arrays stripped, but InferResult
    # requires `predictions`. Backfill an empty list: planner.decide() excludes
    # these arrays from what it sends, so the observation is unaffected.
    r["infer_result"].setdefault("predictions", [])
    r["train_result"]["metrics"]["train_accuracy"] = train_acc
    r["infer_result"]["accuracy"] = val_acc
    r["analysis_result"]["accuracy"] = val_acc
    r["analysis_result"]["macro_auc"] = auc
    r["analysis_result"]["confusion_matrix"] = confusion
    if per_class:
        r["analysis_result"]["per_class"] = per_class
    return st


def build_states() -> dict[str, dict]:
    base = _base_run_state()
    recorded = base["runs"][0]
    rec_conf = recorded["analysis_result"]["confusion_matrix"]

    healthy_src = json.load(open(BASE_STATE))["arms"]["resnet34_ref"]["trajectory"]["runs"][1]
    ai0 = json.load(open("docs/ai_scientist_run.json"))["tuning"]["runs"][0]

    states = {
        "overfit": {
            "provenance": "VERBATIM RECORDED — paper_run resnet34_ref iteration 0",
            "numbers": "train 0.9839 / val 0.7517 / gap 0.2322 / macro_auc 0.8922",
            "expected": "regularize (augment / dropout / weight decay / early stop)",
            "state": _set_metrics(
                base, train_acc=0.9839, val_acc=0.7516666666666667,
                auc=0.8921652777777779, confusion=rec_conf),
        },
        "healthy": {
            "provenance": "VERBATIM RECORDED — paper_run resnet34_ref iteration 1",
            "numbers": "train 0.8233 / val 0.8167 / gap 0.0066 / macro_auc 0.9283",
            "expected": "stop or report (the gap is closed)",
            "state": _set_metrics(
                base, train_acc=0.8233, val_acc=0.8166666666666667,
                auc=0.9283296296296296,
                confusion=healthy_src["analysis_result"]["confusion_matrix"]),
        },
        "underfit": {
            "provenance": ("CONSTRUCTED PROBE — val/AUC/confusion verbatim from "
                           "ai_scientist_run tuning iteration 0; train set to 0.6500 "
                           "so train and val are both low with no gap"),
            "numbers": "train 0.6500 / val 0.6294 / gap 0.0206 / macro_auc 0.8068",
            "expected": "add capacity or training (more epochs / larger model / higher LR); "
                        "adding regularization would be harmful",
            "state": _set_metrics(
                base, train_acc=0.6500, val_acc=0.6294444444444445,
                auc=0.8068305555555556,
                confusion=ai0["analysis_result"]["confusion_matrix"]),
        },
        "collapsed": {
            "provenance": ("CONSTRUCTED PROBE — train/val from the loop-guard regression "
                           "(chance level for 3 classes); confusion built consistent with "
                           "chance, AUC 0.51"),
            "numbers": "train 0.3740 / val 0.3320 / gap 0.0420 / macro_auc 0.5100",
            "expected": "recognise collapse — redesign, restart, lower LR, or stop; "
                        "reporting success would be harmful",
            "state": _set_metrics(
                base, train_acc=0.3740, val_acc=0.3320, auc=0.5100,
                confusion=[[201, 199, 200], [202, 199, 199], [201, 200, 199]]),
        },
    }
    return states


# --- convenience scoring -------------------------------------------------- #
REGULARIZERS = {"augment", "dropout", "weight_decay", "early_stop_patience", "label_smoothing"}
CAPACITY_UP = {"epochs", "learning_rate", "widths", "depths"}


def knobs_of(updated: dict) -> list[str]:
    out = []
    for section, v in (updated or {}).items():
        if isinstance(v, dict):
            out += [f"{section}.{k}" for k in v]
        else:
            out.append(section)
    return sorted(out)


def leaf_names(updated: dict) -> set[str]:
    names = set()
    for _, v in (updated or {}).items():
        if isinstance(v, dict):
            names |= set(v)
    return names


def judge_quality(state_name: str, action: str, updated: dict) -> str:
    """'ok' | 'noop' | 'harmful' — a convenience label; rationale text is kept."""
    leaves = leaf_names(updated)
    if state_name == "overfit":
        if action in ("report", "stop"):
            return "harmful"          # a 0.23 gap is not a finished experiment
        return "ok" if leaves & REGULARIZERS else ("noop" if not leaves else "ok")
    if state_name == "healthy":
        if action in ("report", "stop"):
            return "ok"
        return "noop" if not leaves else "harmful"   # gap is closed; more churn is not warranted
    if state_name == "underfit":
        if action in ("report", "stop"):
            return "harmful"
        if leaves & REGULARIZERS and not (leaves & CAPACITY_UP):
            return "harmful"          # regularizing an underfit model
        return "ok" if leaves else "noop"
    if state_name == "collapsed":
        if action == "report":
            return "harmful"          # reporting a chance-level model as a result
        return "ok" if (leaves or action in ("design_model", "stop", "simulate")) else "noop"
    return "ok"


def entropy(counter: Counter) -> float:
    n = sum(counter.values())
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--model", default=os.environ.get("DLENS_LLM", "gpt-5.2"))
    p.add_argument("--states", default="overfit,healthy,underfit,collapsed")
    args = p.parse_args()

    from dlens.agents._experiment_planner import ExperimentPlanner
    from dlens.agents._models import OpenAIModel
    from dlens.schemas._experiment import ExperimentState

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    states = build_states()
    wanted = [s.strip() for s in args.states.split(",") if s.strip()]
    planner = ExperimentPlanner(model=OpenAIModel(model_name=args.model))

    out = {"model": args.model, "trials": args.trials,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "states": {}}

    for name in wanted:
        meta = states[name]
        st = ExperimentState.model_validate(meta["state"])
        print(f"\n===== {name.upper()} =====")
        print(f"  {meta['provenance']}")
        print(f"  {meta['numbers']}")
        print(f"  expected: {meta['expected']}")
        trials = []
        for i in range(args.trials):
            t0 = time.monotonic()
            try:
                dec = (await planner.decide(st)).output
                rec = {"trial": i + 1, "action": dec.action.value,
                       "updated_params": dec.updated_params,
                       "knobs": knobs_of(dec.updated_params),
                       "rationale": dec.rationale,
                       "quality": judge_quality(name, dec.action.value, dec.updated_params),
                       "seconds": round(time.monotonic() - t0, 1)}
            except Exception as exc:
                rec = {"trial": i + 1, "error": f"{type(exc).__name__}: {exc}"[:300],
                       "seconds": round(time.monotonic() - t0, 1)}
            trials.append(rec)
            tag = rec.get("quality", "ERR")
            print(f"   trial {i+1:>2}: {rec.get('action','ERR'):<13} "
                  f"{','.join(rec.get('knobs', [])) or '(none)':<46} [{tag}]", flush=True)
            out["states"][name] = {**{k: v for k, v in meta.items() if k != "state"},
                                   "trials": trials}
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=2)

        ok = [t for t in trials if "error" not in t]
        acts = Counter(t["action"] for t in ok)
        knobsets = Counter(",".join(t["knobs"]) or "(none)" for t in ok)
        qual = Counter(t["quality"] for t in ok)
        out["states"][name]["summary"] = {
            "n": len(ok), "actions": dict(acts), "knob_sets": dict(knobsets),
            "quality": dict(qual),
            "action_entropy_bits": round(entropy(acts), 3),
            "knobset_entropy_bits": round(entropy(knobsets), 3),
            "modal_knob_set": knobsets.most_common(1)[0] if knobsets else None,
        }
        s = out["states"][name]["summary"]
        print(f"  -> actions {dict(acts)}")
        print(f"  -> knob sets {dict(knobsets)}")
        print(f"  -> quality {dict(qual)}")
        print(f"  -> entropy: action {s['action_entropy_bits']} bits, "
              f"knob-set {s['knobset_entropy_bits']} bits")
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)

    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
