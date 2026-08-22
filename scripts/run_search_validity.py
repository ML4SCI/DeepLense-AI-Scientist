"""E3b / E4 — is the search's ranking machinery actually valid?

Two questions the recorded runs cannot answer, because both are about
architectures or epoch budgets the search never evaluated.

E3b (judge filter validity). The judge ranks N candidates and only its top-k are
ever trained, so E3's correlation is restricted to the set the judge already
liked. This trains the candidates the judge REJECTED under the SAME short
protocol as the shortlist, and asks: would any rejected architecture have beaten
the judge's top-k?

E4 (low-fidelity ranking bias). The shortlist is ranked on a short training run.
This retrains those same candidates to the full tuning budget and asks whether
the short-run winner is the full-run winner.

Protocol is copied from ArchitectureSearch._evaluate so the numbers are
comparable to the recorded ones: TrainingConfig(cross_entropy, batch_size=32),
epochs from --epochs, torch backend, seed 0, val accuracy from a fresh inference
pass. Nothing is mocked.

    .venv/bin/python scripts/run_search_validity.py --mode e3b \\
        --source docs/paper_run.json --data-root ~/Personal/GSoC/deeplense_data/model1 \\
        --train-dir train_small --val-dir val --epochs 10 \\
        --out docs/agent_reliability/e3b_paper_run.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def _ref(root: str, classes: list[str]):
    """Same flat {class}_{index}.npy layout the recorded runs used."""
    import glob

    from dlens.schemas._downstream import DatasetRef
    n = sum(len(glob.glob(os.path.join(root, f"{c}_*.npy"))) for c in classes)
    sample = np.load(sorted(glob.glob(os.path.join(root, f"{classes[0]}_*.npy")))[0])
    return DatasetRef(root=root, class_names=classes, image_shape=tuple(sample.shape),
                      num_samples=n)


def load_search(path: str) -> dict:
    d = json.load(open(os.path.expanduser(path)))
    return d["search"] if "search" in d else d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["e3b", "e4"], required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-dir", default="train_small")
    p.add_argument("--val-dir", default="val")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classes", default="no_sub,cdm,axion")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from dlens.agents._architecture_search import _params_m
    from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig
    from dlens.tools._analysis import compute_analysis
    from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

    sec = load_search(args.source)
    proposed = sec.get("proposed") or sec.get("candidates")
    ranking = sec["judge_ranking"]
    rounds = sec.get("rounds") or sec.get("shortlist_rounds")
    shortlist = {e["name"]: e for e in rounds[0]}
    names = [c["name"] for c in proposed]

    if args.mode == "e3b":
        pick = [c for c in proposed if c["name"] not in shortlist]
        label = "REJECTED by the judge (never trained in the recorded run)"
    else:
        pick = [c for c in proposed if c["name"] in shortlist]
        label = "the recorded shortlist, retrained to full budget"

    root = os.path.expanduser(args.data_root)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    train_ref = _ref(os.path.join(root, args.train_dir), classes)
    val_ref = _ref(os.path.join(root, args.val_dir), classes)

    print(f"=== {args.mode.upper()} — {label} ===")
    print(f"  source {args.source}")
    print(f"  train {train_ref.num_samples} from {args.train_dir} | "
          f"val {val_ref.num_samples} from {args.val_dir} | epochs {args.epochs} | seed {args.seed}")
    for c in pick:
        pos = ranking.index(names.index(c["name"]))
        print(f"    judge#{pos+1:<2} {c['name']}")

    train_backend = TorchTrainBackend(output_root=os.path.join(root, "e_runs"), seed=args.seed)
    infer_backend = TorchInferBackend()
    cfg = TrainingConfig(loss="cross_entropy", epochs=args.epochs, batch_size=32)

    out = {"mode": args.mode, "source": args.source, "epochs": args.epochs, "seed": args.seed,
           "train_dir": args.train_dir, "val_dir": args.val_dir,
           "n_train": train_ref.num_samples, "n_val": val_ref.num_samples,
           "judge_ranking": ranking, "shortlist_recorded": rounds[0], "results": []}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for c in pick:
        spec = ArchitectureSpec.model_validate({
            "name": c["name"], "family": c["family"], "input_shape": [150, 150],
            "channels": 1, "num_classes": 3, "depths": c["depths"], "widths": c["widths"],
            "physics_informed": False, "rationale": "",
        })
        pos = ranking.index(names.index(c["name"]))
        t0 = time.monotonic()
        tr = train_backend.train(train_ref, spec, cfg)
        ir = infer_backend.infer(tr.weights_path, val_ref)
        ar = compute_analysis(ir, val_ref.class_names)
        dt = round(time.monotonic() - t0, 1)
        train_acc = float(tr.metrics.get("train_accuracy", float("nan")))
        val_acc = float(ir.accuracy or float("nan"))
        row = {"name": spec.name, "judge_rank": pos + 1, "params_m": _params_m(spec),
               "train_accuracy": train_acc, "val_accuracy": val_acc,
               "gap": round(train_acc - val_acc, 4), "macro_auc": ar.macro_auc,
               "train_seconds": dt,
               "recorded_short_val": (shortlist.get(spec.name) or {}).get("val_accuracy")}
        out["results"].append(row)
        print(f"    {spec.name:<24} judge#{pos+1:<2} val={val_acc:.4f} "
              f"gap={row['gap']:+.4f} auc={ar.macro_auc:.4f} ({dt}s)", flush=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)

    print("\n=== comparison ===")
    if args.mode == "e3b":
        best_short = max(rounds[0], key=lambda e: e["val_accuracy"])
        print(f"  best SHORTLISTED (recorded): {best_short['name']} val "
              f"{best_short['val_accuracy']:.4f}")
        beat = [r for r in out["results"] if r["val_accuracy"] > best_short["val_accuracy"]]
        allshort = sorted(rounds[0], key=lambda e: -e["val_accuracy"])
        worst_short = allshort[-1]
        beat_any = [r for r in out["results"] if r["val_accuracy"] > worst_short["val_accuracy"]]
        out["comparison"] = {
            "best_shortlisted": {"name": best_short["name"],
                                 "val_accuracy": best_short["val_accuracy"]},
            "rejected_beating_best_shortlisted": [r["name"] for r in beat],
            "rejected_beating_worst_shortlisted": [r["name"] for r in beat_any],
            "n_rejected_trained": len(out["results"]),
        }
        print(f"  rejected candidates beating the BEST shortlisted: "
              f"{[r['name'] for r in beat] or 'none'}")
        print(f"  rejected candidates beating the WORST shortlisted "
              f"({worst_short['name']} {worst_short['val_accuracy']:.4f}): "
              f"{[r['name'] for r in beat_any] or 'none'}")
    else:
        short_rank = [e["name"] for e in sorted(rounds[0], key=lambda e: -e["val_accuracy"])]
        full_rank = [r["name"] for r in sorted(out["results"], key=lambda r: -r["val_accuracy"])]
        import itertools
        def tau(a, b):
            ra = {v: i for i, v in enumerate(a)}; rb = {v: i for i, v in enumerate(b)}
            c = d = 0
            for x, y in itertools.combinations(a, 2):
                s = (ra[x]-ra[y])*(rb[x]-rb[y])
                if s > 0: c += 1
                elif s < 0: d += 1
            n = len(a)
            return (c-d)/(0.5*n*(n-1))
        out["comparison"] = {
            "short_epoch_ranking": short_rank, "full_epoch_ranking": full_rank,
            "kendall_tau": round(tau(short_rank, full_rank), 4),
            "short_winner": short_rank[0], "full_winner": full_rank[0],
            "winner_preserved": short_rank[0] == full_rank[0],
        }
        print(f"  short-budget ranking: {short_rank}")
        print(f"  full-budget  ranking: {full_rank}")
        print(f"  Kendall tau = {out['comparison']['kendall_tau']:+.4f}")
        print(f"  short winner == full winner? {out['comparison']['winner_preserved']}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
