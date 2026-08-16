"""E4 for the broadened search: does the 10-epoch proxy rank families fairly?

The search prunes its shortlist on a 10-epoch training run. A prior audit
(docs/AGENT_RELIABILITY.md) showed that proxy inverted the ranking in two
convolution-only searches. With transformer and group-equivariant families now in
the space the risk is sharper and more specific: those families were observed to
need noticeably more optimizer steps to leave their initial plateau, so a short
budget may be measuring how fast a family STARTS rather than how good it is.

This retrains a given shortlist at the full budget under the identical protocol
and reports the rank correlation, broken down by family. If the short-budget
ranking systematically under-rates non-convolutional families, the search result
is a budget artifact and must be re-run at a higher candidate_epochs before it
supports any claim about which architecture the search "converged on".

    .venv/bin/python scripts/run_budget_validity.py \\
        --data-root ~/Personal/GSoC/deeplense_data/model1 \\
        --train-dir train --val-dir val --epochs 25 \\
        --out docs/broader_search/e4_budget_validity.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import time

import numpy as np

# The round-1 shortlist of the 2026-08-15 broadened search, with the validation
# accuracy each reached at the 10-epoch budget. Specs are transcribed from the
# run log (docs/broader_search/run_partial.log) so this is reproducible without
# re-running the search.
SHORTLIST = [
    # name, family, depths, widths, val_accuracy @10 epochs, train_seconds @10
    ("cnn_deeper_s4",       "cnn",         [3, 3, 3, 2], [32, 64, 128, 256], 0.9033, 285.4),
    ("resnet_deep_narrow",  "resnet",      [4, 4, 3, 2], [32, 64, 128, 256], 0.8778, 95.4),
    ("resnet_micro_4stage", "resnet",      [1, 1, 1, 1], [32, 64, 128, 256], 0.8433, 48.3),
    ("equivariant_c4_mid",  "equivariant", [2, 2, 2],    [16, 32, 64],       0.7206, 695.1),
]

# The three non-convolutional candidates the judge ranked 5th, 6th and 7th in
# round 1 — just outside the top-4 cut, so the search never trained them and the
# shortlist above contains no transformer or hybrid model at all. Without these
# the question "were transformer/hybrid families under-rated by the short budget?"
# cannot be answered: it is a restricted-range comparison over what the judge
# already promoted. Their short-budget accuracy is unknown (never measured), so
# it is recorded as None and both budgets are run here.
REJECTED_NONCONV = [
    ("hybrid_conv2_attn6",     "hybrid",   [2, 2, 6],    [32, 64, 256],      None, None),
    ("vit_small_8b_d192",      "vit",      [2, 2, 2, 2], [192, 192, 192, 192], None, None),
    ("mlpmixer_small_8b_d256", "mlpmixer", [4, 4],       [256, 256],         None, None),
]


def _ref(root: str, classes: list[str]):
    from dlens.schemas._downstream import DatasetRef
    n = sum(len(glob.glob(os.path.join(root, f"{c}_*.npy"))) for c in classes)
    sample = np.load(sorted(glob.glob(os.path.join(root, f"{classes[0]}_*.npy")))[0])
    return DatasetRef(root=root, class_names=classes, image_shape=tuple(sample.shape),
                      num_samples=n)


def kendall_tau(a: list[float], b: list[float]) -> float:
    c = d = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0:
            c += 1
        elif s < 0:
            d += 1
    n = len(a)
    return (c - d) / (0.5 * n * (n - 1)) if n > 1 else float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-dir", default="train")
    p.add_argument("--val-dir", default="val")
    p.add_argument("--classes", default="no_sub,cdm,axion")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--set", choices=["shortlist", "rejected"], default="shortlist")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from dlens.agents._architecture_search import _params_m
    from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig
    from dlens.tools._analysis import compute_analysis
    from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

    root = os.path.expanduser(args.data_root)
    classes = [c.strip() for c in args.classes.split(",")]
    train_ref = _ref(os.path.join(root, args.train_dir), classes)
    val_ref = _ref(os.path.join(root, args.val_dir), classes)

    # This branch predates the multi-seed backend: training is seeded internally
    # with torch.manual_seed(0), which is exactly what the search itself used, so
    # the comparison stays like-for-like.
    train_backend = TorchTrainBackend(output_root=os.path.join(root, "e4_runs"))
    infer_backend = TorchInferBackend()
    # Identical to ArchitectureSearch._evaluate except for the epoch budget.
    cfg = TrainingConfig(loss="cross_entropy", epochs=args.epochs, batch_size=32)

    print(f"=== E4: shortlist retrained at {args.epochs} epochs "
          f"({train_ref.num_samples} train / {val_ref.num_samples} val, seed 0) ===",
          flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {"epochs": args.epochs, "seed": 0, "n_train": train_ref.num_samples,
           "n_val": val_ref.num_samples, "results": []}

    chosen = SHORTLIST if args.set == "shortlist" else REJECTED_NONCONV
    for name, family, depths, widths, short_val, short_s in chosen:
        spec = ArchitectureSpec.model_validate({
            "name": name, "family": family, "input_shape": list(train_ref.image_shape),
            "channels": 1, "num_classes": len(classes), "depths": depths, "widths": widths,
            "physics_informed": family == "equivariant", "rationale": "",
        })
        t0 = time.monotonic()
        tr = train_backend.train(train_ref, spec, cfg)
        ir = infer_backend.infer(tr.weights_path, val_ref)
        ar = compute_analysis(ir, val_ref.class_names)
        dt = round(time.monotonic() - t0, 1)
        train_acc = float(tr.metrics.get("train_accuracy", float("nan")))
        val_acc = float(ir.accuracy or float("nan"))
        row = {
            "name": name, "family": family, "params_m": _params_m(spec),
            "short_val_accuracy": short_val, "short_train_seconds": short_s,
            "full_val_accuracy": val_acc, "full_train_accuracy": train_acc,
            "gap": round(train_acc - val_acc, 4), "macro_auc": ar.macro_auc,
            "full_train_seconds": dt,
            "delta": (round(val_acc - short_val, 4) if short_val is not None else None),
        }
        out["results"].append(row)
        short_txt = f"@10ep {short_val:.4f} -> " if short_val is not None else "never trained -> "
        delta_txt = f"({row['delta']:+.4f})  " if row["delta"] is not None else ""
        print(f"  {name:<22} {family:<12} {row['params_m']:>7}M  "
              f"{short_txt}@{args.epochs}ep {val_acc:.4f}  "
              f"{delta_txt}gap {row['gap']:+.4f}  {dt}s", flush=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)

    res = out["results"]
    if any(r["short_val_accuracy"] is None for r in res):
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nartifacts -> {args.out}")
        return 0
    short_rank = [r["name"] for r in sorted(res, key=lambda r: -r["short_val_accuracy"])]
    full_rank = [r["name"] for r in sorted(res, key=lambda r: -r["full_val_accuracy"])]
    s = [r["short_val_accuracy"] for r in res]
    f = [r["full_val_accuracy"] for r in res]
    conv = [r for r in res if r["family"] in ("cnn", "resnet")]
    other = [r for r in res if r["family"] not in ("cnn", "resnet")]
    out["comparison"] = {
        "short_budget_ranking": short_rank,
        "full_budget_ranking": full_rank,
        "kendall_tau": round(kendall_tau(s, f), 4),
        "winner_preserved": short_rank[0] == full_rank[0],
        "mean_delta_convolutional": (
            round(sum(r["delta"] for r in conv) / len(conv), 4) if conv else None),
        "mean_delta_non_convolutional": (
            round(sum(r["delta"] for r in other) / len(other), 4) if other else None),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    c = out["comparison"]
    print(f"\n  short-budget ranking : {short_rank}")
    print(f"  full-budget ranking  : {full_rank}")
    print(f"  Kendall tau          : {c['kendall_tau']:+.4f}")
    print(f"  winner preserved     : {c['winner_preserved']}")
    print(f"  mean gain, conv      : {c['mean_delta_convolutional']:+.4f}")
    print(f"  mean gain, non-conv  : {c['mean_delta_non_convolutional']:+.4f}")
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
