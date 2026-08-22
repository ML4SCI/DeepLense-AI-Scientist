"""Complete the broadened search: round 3, ReAct tuning of the winner, held-out test.

The 2026-08-15 run finished rounds 1 and 2 and was stopped partway through round 3
because its first candidate, an equivariant model, was projected at ~180 minutes
for a 10-epoch evaluation. This script finishes what is affordable and records
what is not, rather than quietly re-running the whole search.

  --stage round3  train round 3's remaining candidates at the search's own budget
  --stage tune    ReAct-tune the global winner, then evaluate ONCE on the held-out
                  test set

Round 3's equivariant candidate is deliberately NOT trained here. At ~1080 s/epoch
a 10-epoch evaluation costs ~3 hours, roughly 36x the three convolutional
candidates in the same shortlist combined. That is recorded as a result about the
search's affordability, not hidden as a gap.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import time

import numpy as np

# Round 3's shortlist, transcribed from docs/broader_search/run_partial.log.
ROUND3 = [
    ("cnn_s4_mid_w224",      "cnn", [3, 3, 3, 2], [32, 64, 128, 224]),
    ("cnn_s4_w256_deeper",   "cnn", [2, 4, 4, 2], [32, 64, 128, 256]),
    ("cnn_s4_wider_early",   "cnn", [3, 3, 3, 2], [48, 96, 160, 256]),
]
# Not trained; see the module docstring.
ROUND3_DEFERRED = [
    ("equivariant_c4_wider", "equivariant", [4, 3, 2], [24, 48, 96], "~1080 s/epoch"),
]

# Best of rounds 1-2, from the same log.
INCUMBENT = ("cnn_deeper_s4", "cnn", [3, 3, 3, 2], [32, 64, 128, 256], 0.9033)


def _ref(root: str, classes: list[str]):
    from dlens.schemas._downstream import DatasetRef
    n = sum(len(glob.glob(os.path.join(root, f"{c}_*.npy"))) for c in classes)
    sample = np.load(sorted(glob.glob(os.path.join(root, f"{classes[0]}_*.npy")))[0])
    return DatasetRef(root=root, class_names=classes, image_shape=tuple(sample.shape),
                      num_samples=n)


def _spec(name, family, depths, widths, shape, n_classes):
    from dlens.schemas._model_design import ArchitectureSpec
    return ArchitectureSpec.model_validate({
        "name": name, "family": family, "input_shape": list(shape), "channels": 1,
        "num_classes": n_classes, "depths": depths, "widths": widths,
        "physics_informed": family == "equivariant", "rationale": "",
    })


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["round3", "tune"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-root", default=None, help="required for --stage tune")
    p.add_argument("--train-dir", default="train")
    p.add_argument("--val-dir", default="val")
    p.add_argument("--classes", default="no_sub,cdm,axion")
    p.add_argument("--candidate-epochs", type=int, default=10)
    p.add_argument("--tune-epochs", type=int, default=25)
    p.add_argument("--tune-iterations", type=int, default=4)
    p.add_argument("--round3-json", default=None,
                   help="Round-3 results, so the tune stage picks the true global "
                        "winner. Deriving this from --out by string substitution was "
                        "brittle and silently tuned the wrong architecture.")
    p.add_argument("--seed", type=int, default=0,
                   help="Training seed; the pilot protocol uses 0 and 1.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from dlens.agents._architecture_search import _params_m
    from dlens.schemas._model_design import TrainingConfig
    from dlens.tools._analysis import compute_analysis
    from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

    root = os.path.expanduser(args.data_root)
    classes = [c.strip() for c in args.classes.split(",")]
    train_ref = _ref(os.path.join(root, args.train_dir), classes)
    val_ref = _ref(os.path.join(root, args.val_dir), classes)
    train_backend = TorchTrainBackend(
        output_root=os.path.join(root, f"finish_runs_s{args.seed}"), seed=args.seed)
    infer_backend = TorchInferBackend()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.stage == "round3":
        cfg = TrainingConfig(loss="cross_entropy", epochs=args.candidate_epochs, batch_size=32)
        out = {"stage": "round3", "epochs": args.candidate_epochs,
               "n_train": train_ref.num_samples, "n_val": val_ref.num_samples,
               "results": [],
               "deferred": [{"name": n, "family": f, "depths": d, "widths": w,
                             "reason": f"projected {c} -> ~3 h for a 10-epoch evaluation"}
                            for n, f, d, w, c in ROUND3_DEFERRED]}
        print(f"=== ROUND 3 (remaining candidates, {args.candidate_epochs} epochs) ===", flush=True)
        for name, fam, depths, widths in ROUND3:
            spec = _spec(name, fam, depths, widths, train_ref.image_shape, len(classes))
            t0 = time.monotonic()
            tr = train_backend.train(train_ref, spec, cfg)
            ir = infer_backend.infer(tr.weights_path, val_ref)
            ar = compute_analysis(ir, val_ref.class_names)
            dt = round(time.monotonic() - t0, 1)
            ta = float(tr.metrics.get("train_accuracy", float("nan")))
            va = float(ir.accuracy or float("nan"))
            out["results"].append({
                "name": name, "family": fam, "params_m": _params_m(spec),
                "train_accuracy": ta, "val_accuracy": va, "gap": round(ta - va, 4),
                "macro_auc": ar.macro_auc, "train_seconds": dt,
                "weights_path": tr.weights_path,
            })
            print(f"  {name:<22}{fam:<8}{_params_m(spec):>7}M  val={va:.4f} "
                  f"gap={ta-va:+.4f} auc={ar.macro_auc:.4f}  {dt}s", flush=True)
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=2)
        best = max(out["results"], key=lambda r: r["val_accuracy"])
        out["round3_winner"] = best["name"]
        out["beats_incumbent"] = best["val_accuracy"] > INCUMBENT[4]
        print(f"\n  round-3 winner : {best['name']} ({best['val_accuracy']:.4f})")
        print(f"  incumbent      : {INCUMBENT[0]} ({INCUMBENT[4]:.4f})")
        print(f"  new global best: {'round 3' if out['beats_incumbent'] else INCUMBENT[0]}")
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        return 0

    # ---------------------------------------------------------------- tune --
    assert args.test_root, "--test-root is required for --stage tune"
    from dlens.agents._experiment_loop import ExperimentLoop
    from dlens.agents._experiment_planner import ExperimentPlanner, ReActPlannerStrategy
    from dlens.config import build_llm

    # Which architecture won overall: read round 3's result if present.
    winner = _spec(*INCUMBENT[:4], train_ref.image_shape, len(classes))
    winner_val = INCUMBENT[4]
    r3_path = args.round3_json
    if r3_path and os.path.exists(r3_path):
        r3 = json.load(open(r3_path))
        best = max(r3["results"], key=lambda r: r["val_accuracy"])
        if best["val_accuracy"] > winner_val:
            winner = _spec(best["name"], best["family"],
                           next(c[2] for c in ROUND3 if c[0] == best["name"]),
                           next(c[3] for c in ROUND3 if c[0] == best["name"]),
                           train_ref.image_shape, len(classes))
            winner_val = best["val_accuracy"]
    if args.round3_json and not os.path.exists(args.round3_json):
        raise SystemExit(f"--round3-json {args.round3_json} does not exist; refusing to "
                         "guess the winner")
    print(f"=== TUNING {winner.name} (search val {winner_val:.4f}) seed={args.seed} ===",
          flush=True)

    llm = build_llm(None)
    loop = ExperimentLoop(
        strategy=ReActPlannerStrategy(ExperimentPlanner(model=llm)),
        train_backend=train_backend, infer_backend=infer_backend,
    )
    cfg = TrainingConfig(loss="cross_entropy", optimizer="adamw", learning_rate=3e-4,
                         batch_size=32, epochs=args.tune_epochs, weight_decay=1e-4,
                         lr_scheduler="cosine")
    state = await loop.run(
        hypothesis=(f"Tune the search winner '{winner.name}' to close any generalization "
                    "gap on DeepLense Model_I."),
        train_ref=train_ref, val_ref=val_ref, architecture=winner,
        training_config=cfg, max_iterations=args.tune_iterations,
    )

    # Select on validation only, then touch the test set exactly once.
    best_run = max(state.runs, key=lambda r: r.infer_result.accuracy or -1.0)
    test_ref = _ref(os.path.expanduser(args.test_root), classes)
    ir = infer_backend.infer(best_run.train_result.weights_path, test_ref)
    ar = compute_analysis(ir, classes)

    out = {
        "stage": "tune", "seed": args.seed, "winner": winner.model_dump(mode="json"),
        "search_val_accuracy": winner_val,
        "trajectory": [{
            "iteration": r.iteration,
            "train_accuracy": r.train_result.metrics.get("train_accuracy"),
            "val_accuracy": r.infer_result.accuracy,
            "gap": round((r.train_result.metrics.get("train_accuracy") or 0)
                         - (r.infer_result.accuracy or 0), 4),
            "macro_auc": r.analysis_result.macro_auc,
            "planner_action": (r.planner_decision.action.value if r.planner_decision else None),
            "planner_rationale": (r.planner_decision.rationale if r.planner_decision else None),
            "planner_updated_params": (r.planner_decision.updated_params
                                       if r.planner_decision else None),
        } for r in state.runs],
        "best_iteration": best_run.iteration,
        "best_val_accuracy": best_run.infer_result.accuracy,
        "test": {"n_test": test_ref.num_samples, "test_accuracy": ir.accuracy,
                 "test_macro_auc": ar.macro_auc, "per_class": ar.per_class,
                 "confusion_matrix": ar.confusion_matrix},
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n  trajectory:")
    for r in out["trajectory"]:
        print(f"    it{r['iteration']}: train={r['train_accuracy']:.4f} "
              f"val={r['val_accuracy']:.4f} gap={r['gap']:+.4f} -> {r['planner_action']}")
    print(f"\n  best iteration : it{out['best_iteration']} "
          f"val={out['best_val_accuracy']:.4f}")
    print(f"  HELD-OUT TEST  : acc={ir.accuracy:.4f} auc={ar.macro_auc:.4f} "
          f"(n={test_ref.num_samples}, evaluated once)")
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
