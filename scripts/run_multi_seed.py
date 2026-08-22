"""Multi-seed replication + full artifact capture for the AI-Scientist paper.

Two experiments, both on the FULL dataset (7,200 train / 1,800 val) with the
held-out test set touched exactly once per final model:

  EXPERIMENT A — is the performance gap robust?
      Fixed architectures (cnn_medium_s4, resnet34) x N seeds, each tuned under
      the identical ReAct planner protocol, then evaluated once on TEST.

  EXPERIMENT B — does the agent independently converge on small architectures?
      The complete pipeline N times (generate -> judge -> shortlist training ->
      prune -> refine -> winner -> ReAct tuning -> TEST). Fresh LLM sampling
      each run; the nondeterminism is the measurement.

Test-set discipline is enforced in code: the test DatasetRef is only constructed
inside ``evaluate_on_test()``, which refuses to run unless the caller has marked
selection and tuning complete for that run.

    DLENS_LLM=gpt-5.2 OPENAI_API_KEY=... uv run python scripts/run_multi_seed.py \\
        --data-root ~/Personal/GSoC/deeplense_data/model1 \\
        --test-root ~/Personal/GSoC/deeplense_data/model1_test \\
        --artifact-root ~/Personal/GSoC/deeplense_artifacts \\
        --seeds 0,1 --pipeline-runs 1
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np

from dlens.agents._architecture_search import (
    ArchitectureGenerator,
    ArchitectureJudge,
    ArchitectureSearch,
    _params_m,
)
from dlens.agents._experiment_loop import ExperimentLoop, best_run_by_val_accuracy
from dlens.agents._experiment_planner import ExperimentPlanner, ReActPlannerStrategy
from dlens.config import DEFAULT_LLM, build_llm
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig
from dlens.tools._analysis import compute_analysis
from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

RULE = "=" * 78

# The two architectures Experiment A takes as given: the agent's pick from the
# definitive run (docs/paper_run.json search winner) and the manual baseline.
CNN_MEDIUM_S4 = dict(
    name="cnn_medium_s4", family=ArchFamily.CNN, depths=[2, 3, 3, 4],
    widths=[32, 64, 128, 256],
)
RESNET34 = dict(name="resnet34", family=ArchFamily.RESNET, depths=None, widths=None)


class Tee:
    """Duplicate stdout into a per-run log file (full training logs are artifacts)."""

    def __init__(self, path: str) -> None:
        self._fh = open(path, "w", buffering=1)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self):
        self._stdout.flush()
        self._fh.flush()

    def close(self):
        self._fh.close()


@contextmanager
def tee_stdout(path: str):
    t = Tee(path)
    sys.stdout = t
    try:
        yield
    finally:
        sys.stdout = t._stdout
        t.close()


def dataset_ref(root: str, classes: list[str]) -> DatasetRef:
    n = sum(len(glob.glob(os.path.join(root, f"{c}_*.npy"))) for c in classes)
    if n == 0:
        raise SystemExit(f"no .npy images found under {root}")
    sample = np.load(sorted(glob.glob(os.path.join(root, f"{classes[0]}_*.npy")))[0])
    return DatasetRef(root=root, class_names=classes,
                      image_shape=tuple(sample.shape), num_samples=n)


def spec_of(kind: dict, shape: tuple[int, int], n_classes: int) -> ArchitectureSpec:
    return ArchitectureSpec(
        name=kind["name"], family=kind["family"], input_shape=shape, channels=1,
        num_classes=n_classes, depths=kind["depths"], widths=kind["widths"],
    )


# --------------------------------------------------------------------------- #
# Test-set quarantine
# --------------------------------------------------------------------------- #

class TestSetGuard:
    """Allows exactly one test evaluation per run, only after selection+tuning."""

    def __init__(self) -> None:
        self._unlocked: set[str] = set()
        self._used: set[str] = set()

    def unlock(self, run_id: str) -> None:
        self._unlocked.add(run_id)

    def check(self, run_id: str) -> None:
        if run_id not in self._unlocked:
            raise AssertionError(
                f"TEST SET ACCESS DENIED for {run_id}: selection/tuning not complete."
            )
        if run_id in self._used:
            raise AssertionError(
                f"TEST SET ACCESS DENIED for {run_id}: already evaluated once."
            )
        self._used.add(run_id)

    @property
    def evaluations(self) -> int:
        return len(self._used)


GUARD = TestSetGuard()


def evaluate_on_test(run_id: str, weights_path: str, test_root: str,
                     classes: list[str]) -> dict:
    """THE ONLY place the test set is loaded. One shot per run, post-selection."""
    GUARD.check(run_id)
    test_ref = dataset_ref(test_root, classes)
    ir = TorchInferBackend().infer(weights_path, test_ref)
    ar = compute_analysis(ir, classes)
    print(f"    TEST (once, n={test_ref.num_samples}): acc={ir.accuracy:.4f} "
          f"auc={ar.macro_auc:.4f}", flush=True)
    return {
        "n_test": test_ref.num_samples,
        "test_accuracy": ir.accuracy,
        "test_macro_auc": ar.macro_auc,
        "per_class": ar.per_class,
        "confusion_matrix": ar.confusion_matrix,
        "predictions": ir.predictions,
        "probabilities": ir.probabilities,
        "true_labels": ir.true_labels,
    }


# --------------------------------------------------------------------------- #
# Artifact helpers
# --------------------------------------------------------------------------- #

def write_json(path: str, payload) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def save_predictions(run_dir: str, name: str, result: dict) -> None:
    """Predictions + probabilities live beside the metrics, not inside them."""
    preds = {k: result.pop(k) for k in ("predictions", "probabilities", "true_labels")
             if k in result}
    if preds:
        write_json(os.path.join(run_dir, f"predictions_{name}.json"), preds)


def collect_checkpoints(run_dir: str, backend_root: str) -> list[dict]:
    """Copy every checkpoint the backend wrote into the run's artifact dir."""
    dest = os.path.join(run_dir, "checkpoints")
    os.makedirs(dest, exist_ok=True)
    out = []
    for src in sorted(glob.glob(os.path.join(backend_root, "*", "model.pt"))):
        tag = os.path.basename(os.path.dirname(src))
        target = os.path.join(dest, f"{tag}.pt")
        shutil.copy2(src, target)
        out.append({"run_id": tag, "file": f"checkpoints/{tag}.pt",
                    "bytes": os.path.getsize(target)})
    return out


def environment_info(llm_id: str) -> dict:
    import torch
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "git_commit": commit,
        "llm": llm_id,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "platform": platform.platform(),
    }


def trajectory_json(state) -> list[dict]:
    """Every planner decision with its reasoning, plus the metrics it saw."""
    out = []
    for r in state.runs:
        train_acc = r.train_result.metrics.get("train_accuracy")
        val_acc = r.infer_result.accuracy
        d = r.planner_decision
        out.append({
            "iteration": r.iteration,
            "architecture": r.architecture.model_dump(mode="json"),
            "training_config": r.training_config.model_dump(mode="json"),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "gap": (train_acc - val_acc) if (train_acc is not None and val_acc is not None)
                   else None,
            "val_macro_auc": r.analysis_result.macro_auc,
            "confusion_matrix": r.analysis_result.confusion_matrix,
            "weights_path": r.train_result.weights_path,
            "planner_action": d.action.value if d else None,
            "planner_reasoning": d.rationale if d else None,
            "planner_updated_params": d.updated_params if d else None,
        })
    return out


def stop_reason_of(state) -> Optional[str]:
    last = state.runs[-1].planner_decision if state.runs else None
    return last.rationale if last else None


# --------------------------------------------------------------------------- #
# Tuning protocol (shared by both experiments)
# --------------------------------------------------------------------------- #

async def tune_and_test(*, run_id: str, run_dir: str, arch: ArchitectureSpec,
                        train_ref: DatasetRef, val_ref: DatasetRef, test_root: str,
                        classes: list[str], llm, seed: int, tune_epochs: int,
                        tune_iterations: int) -> dict:
    """ReAct tuning on train/val, then a single test evaluation of the best-val run."""
    backend_root = os.path.join(run_dir, "_train_runs")
    train_backend = TorchTrainBackend(output_root=backend_root, seed=seed)
    infer_backend = TorchInferBackend()
    loop = ExperimentLoop(
        strategy=ReActPlannerStrategy(ExperimentPlanner(model=llm)),
        train_backend=train_backend, infer_backend=infer_backend,
    )
    cfg = TrainingConfig(loss="cross_entropy", optimizer="adamw", learning_rate=3e-4,
                         batch_size=32, epochs=tune_epochs, weight_decay=1e-4,
                         lr_scheduler="cosine")
    t0 = time.monotonic()
    state = await loop.run(
        hypothesis=(f"Tune '{arch.name}' (seed {seed}) to close any generalization "
                    "gap on DeepLense Model_I."),
        train_ref=train_ref, val_ref=val_ref, architecture=arch, training_config=cfg,
        max_iterations=tune_iterations,
    )
    tune_s = time.monotonic() - t0

    best = best_run_by_val_accuracy(state)
    best_val = best.infer_result.accuracy
    print(f"    best-val iteration: it{best.iteration} val={best_val:.4f} "
          f"(of {len(state.runs)} iterations, {tune_s/60:.1f} min)", flush=True)

    # Selection and tuning are done -> the test set may be opened, once.
    GUARD.unlock(run_id)
    test = evaluate_on_test(run_id, best.train_result.weights_path, test_root, classes)
    save_predictions(run_dir, "test", test)

    # Val predictions of the reported model are artifacts too.
    val_ir = infer_backend.infer(best.train_result.weights_path, val_ref)
    save_predictions(run_dir, "val", {
        "predictions": val_ir.predictions, "probabilities": val_ir.probabilities,
        "true_labels": val_ir.true_labels,
    })

    return {
        "tuning_seconds": round(tune_s, 1),
        "iterations": len(state.runs),
        "best_iteration": best.iteration,
        "val_accuracy": best_val,
        "val_macro_auc": best.analysis_result.macro_auc,
        "train_accuracy": best.train_result.metrics.get("train_accuracy"),
        "gap": (best.train_result.metrics.get("train_accuracy") or 0) - (best_val or 0),
        "stop_reason": stop_reason_of(state),
        "trajectory": trajectory_json(state),
        "test": test,
        "checkpoints": collect_checkpoints(run_dir, backend_root),
    }


# --------------------------------------------------------------------------- #
# Experiment A
# --------------------------------------------------------------------------- #

async def experiment_a(args, classes, train_ref, val_ref, llm, llm_id, art_root) -> list[dict]:
    print(f"\n{RULE}\n  EXPERIMENT A — fixed architectures x {len(args.seeds)} seeds\n{RULE}",
          flush=True)
    rows = []
    for kind in (CNN_MEDIUM_S4, RESNET34):
        arch = spec_of(kind, train_ref.image_shape, len(classes))
        for seed in args.seeds:
            run_id = f"expA_{arch.name}_seed{seed}"
            run_dir = os.path.join(art_root, "experiment_a", run_id)
            os.makedirs(run_dir, exist_ok=True)
            print(f"\n--- {run_id} ---", flush=True)
            with tee_stdout(os.path.join(run_dir, "train_log.txt")):
                write_json(os.path.join(run_dir, "config.json"), {
                    "experiment": "A", "run_id": run_id, "seed": seed,
                    "architecture": arch.model_dump(mode="json"),
                    "protocol": {"tune_epochs": args.tune_epochs,
                                 "tune_iterations": args.tune_iterations,
                                 "train_dir": args.train_dir, "val_dir": args.val_dir,
                                 "n_train": train_ref.num_samples,
                                 "n_val": val_ref.num_samples},
                    "environment": environment_info(llm_id),
                })
                res = await tune_and_test(
                    run_id=run_id, run_dir=run_dir, arch=arch, train_ref=train_ref,
                    val_ref=val_ref, test_root=args.test_root, classes=classes, llm=llm,
                    seed=seed, tune_epochs=args.tune_epochs,
                    tune_iterations=args.tune_iterations,
                )
            res.update({"run_id": run_id, "architecture": arch.name, "seed": seed})
            write_json(os.path.join(run_dir, "metrics.json"), res)
            rows.append(res)
    return rows


# --------------------------------------------------------------------------- #
# Experiment B
# --------------------------------------------------------------------------- #

async def experiment_b(args, classes, train_ref, val_ref, llm, llm_id, art_root) -> list[dict]:
    print(f"\n{RULE}\n  EXPERIMENT B — full pipeline x {args.pipeline_runs} runs\n{RULE}",
          flush=True)
    task = (
        f"Task: {len(classes)}-class image classification of simulated strong "
        f"gravitational-lensing images (classes: {', '.join(classes)}). Images are "
        f"{train_ref.image_shape[0]}x{train_ref.image_shape[1]}, 1 channel. "
        f"{train_ref.num_samples} training / {val_ref.num_samples} validation samples. "
        f"Both underfitting and overfitting have been observed on this data depending "
        f"on configuration — judge capacity on the numbers, not by assumption."
    )
    rows = []
    for i in range(args.pipeline_runs):
        seed = args.seeds[i % len(args.seeds)]
        run_id = f"expB_run{i}_seed{seed}"
        run_dir = os.path.join(art_root, "experiment_b", run_id)
        os.makedirs(run_dir, exist_ok=True)
        search_root = os.path.join(run_dir, "_search_runs")
        print(f"\n--- {run_id} ---", flush=True)
        with tee_stdout(os.path.join(run_dir, "train_log.txt")):
            write_json(os.path.join(run_dir, "config.json"), {
                "experiment": "B", "run_id": run_id, "seed": seed,
                "protocol": {"num_candidates": args.num_candidates, "top_k": args.top_k,
                             "rounds": args.rounds, "refine_variants": args.refine_variants,
                             "candidate_epochs": args.candidate_epochs,
                             "tune_epochs": args.tune_epochs,
                             "tune_iterations": args.tune_iterations,
                             "n_train": train_ref.num_samples,
                             "n_val": val_ref.num_samples},
                "environment": environment_info(llm_id),
            })
            search = ArchitectureSearch(
                generator=ArchitectureGenerator(model=llm),
                judge=ArchitectureJudge(model=llm),
                train_backend=TorchTrainBackend(output_root=search_root, seed=seed),
                infer_backend=TorchInferBackend(),
                num_candidates=args.num_candidates, top_k=args.top_k, rounds=args.rounds,
                refine_variants=args.refine_variants, candidate_epochs=args.candidate_epochs,
            )
            t0 = time.monotonic()
            sr = await search.search(task_description=task, train_ref=train_ref,
                                     val_ref=val_ref)
            search_s = time.monotonic() - t0
            search_payload = sr.model_dump(mode="json")
            write_json(os.path.join(run_dir, "search_trajectory.json"), search_payload)
            search_ckpts = collect_checkpoints(run_dir, search_root)

            print(f"    winner: {sr.winner.name} (~{sr.winner_eval.params_m}M params, "
                  f"val={sr.winner_eval.val_accuracy:.4f}) in {search_s/60:.1f} min",
                  flush=True)

            res = await tune_and_test(
                run_id=run_id, run_dir=run_dir, arch=sr.winner, train_ref=train_ref,
                val_ref=val_ref, test_root=args.test_root, classes=classes, llm=llm,
                seed=seed, tune_epochs=args.tune_epochs,
                tune_iterations=args.tune_iterations,
            )
        res.update({
            "run_id": run_id, "seed": seed, "architecture": sr.winner.name,
            "winner_params_m": sr.winner_eval.params_m,
            "search_seconds": round(search_s, 1),
            "candidates": [
                {"name": c.name, "family": c.family.value, "params_m": _params_m(c),
                 "depths": c.depths, "widths": c.widths, "buildable": c.is_buildable()}
                for c in sr.proposed
            ],
            "dropped_unbuildable": search_payload.get("dropped_unbuildable"),
            "judge_ranking": search_payload.get("judge_ranking"),
            "judge_reasoning": search_payload.get("judge_reasoning"),
            "shortlist_rounds": search_payload.get("rounds"),
            "search_checkpoints": search_ckpts,
        })
        write_json(os.path.join(run_dir, "metrics.json"), res)
        rows.append(res)
    return rows


# --------------------------------------------------------------------------- #

MANIFEST_README = """# DeepLense AI-Scientist — multi-seed artifacts

Generated by `scripts/run_multi_seed.py`. Everything needed to check the numbers
in `docs/MULTI_SEED_RESULTS.md` is here; nothing in this tree is required to run
the code.

## Layout

    manifest.json                  index of every run: seed, winner, headline metrics
    README.md                      this file
    experiment_a/<run_id>/         fixed architecture x seed (is the gap robust?)
    experiment_b/<run_id>/         full pipeline run (what does the agent pick?)

Each `<run_id>/` directory is self-describing:

    config.json                    resolved config: architecture, protocol, seed,
                                   dataset sizes, git commit, LLM id, torch/device
    metrics.json                   all metrics for the run, incl. the single TEST
                                   evaluation and the tuning trajectory
    train_log.txt                  complete stdout: per-epoch losses/accuracies,
                                   planner decisions, guard messages
    predictions_test.json          predictions + probabilities + true labels (TEST)
    predictions_val.json           the same for VAL, for the reported model
    checkpoints/<id>.pt            every checkpoint written during the run
    search_trajectory.json         (experiment B only) candidates proposed, judge
                                   ranking + reasoning, per-round training results,
                                   prune/refine outcome, winner
    _train_runs/, _search_runs/    raw backend output the checkpoints were copied from

## Reading a trajectory

`metrics.json -> trajectory[]` has one entry per tuning iteration with the metrics
the planner saw (`train_accuracy`, `val_accuracy`, `gap`, `val_macro_auc`,
`confusion_matrix`), the decision it made (`planner_action`,
`planner_updated_params`) and its stated `planner_reasoning`. `stop_reason` records
why the loop terminated.

## Test-set discipline

The held-out test set is opened exactly once per run, after all selection and
tuning, by `evaluate_on_test()`. A guard object refuses a second evaluation or any
evaluation before tuning completes, so accidental reuse fails loudly rather than
silently leaking. `manifest.json -> test_evaluations` records the total count.
"""


async def main() -> int:
    p = argparse.ArgumentParser(description="Multi-seed replication + artifact capture")
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--train-dir", default="train")
    p.add_argument("--val-dir", default="val")
    p.add_argument("--classes", default="no_sub,cdm,axion")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--pipeline-runs", type=int, default=5)
    p.add_argument("--num-candidates", type=int, default=10)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--refine-variants", type=int, default=3)
    p.add_argument("--candidate-epochs", type=int, default=10)
    p.add_argument("--tune-epochs", type=int, default=25)
    p.add_argument("--tune-iterations", type=int, default=4)
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-b", action="store_true")
    p.add_argument("--tag", default="", help="Suffix for the artifact directory.")
    args = p.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    classes = args.classes.split(",")
    train_ref = dataset_ref(os.path.join(args.data_root, args.train_dir), classes)
    val_ref = dataset_ref(os.path.join(args.data_root, args.val_dir), classes)

    llm = build_llm()  # DLENS_LLM
    llm_id = os.getenv("DLENS_LLM") or DEFAULT_LLM
    import torch

    art_root = os.path.expanduser(args.artifact_root)
    if args.tag:
        art_root = os.path.join(art_root, args.tag)
    os.makedirs(art_root, exist_ok=True)

    print(RULE)
    print("  MULTI-SEED REPLICATION — full dataset, held-out test once per model")
    print(f"  effective LLM   : {type(llm).__name__} model_name="
          f"{getattr(llm, 'model_name', '?')}  (DLENS_LLM={os.getenv('DLENS_LLM')})")
    print(f"  data            : {train_ref.num_samples} train / {val_ref.num_samples} val "
          f"| {train_ref.image_shape}px | classes={classes}")
    print(f"  torch           : {torch.__version__} | device="
          f"{'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print(f"  seeds           : {args.seeds} | pipeline runs: {args.pipeline_runs}")
    print(f"  protocol        : tune {args.tune_epochs} epochs x <= {args.tune_iterations} "
          f"iterations | candidates {args.num_candidates} -> top {args.top_k}")
    print(f"  artifacts       : {art_root}")
    print(RULE, flush=True)
    if "5.6" in str(getattr(llm, "model_name", "")) or "luna" in str(
        getattr(llm, "model_name", "")
    ):
        raise SystemExit("refusing to run: Luna detected. Set DLENS_LLM=gpt-5.2.")

    t_start = time.monotonic()
    rows_a = [] if args.skip_a else await experiment_a(
        args, classes, train_ref, val_ref, llm, llm_id, art_root)
    rows_b = [] if args.skip_b else await experiment_b(
        args, classes, train_ref, val_ref, llm, llm_id, art_root)
    total_s = time.monotonic() - t_start

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": environment_info(llm_id),
        "protocol": vars(args) | {"seeds": args.seeds},
        "dataset": {"n_train": train_ref.num_samples, "n_val": val_ref.num_samples,
                    "train_dir": args.train_dir, "test_root": args.test_root},
        "total_seconds": round(total_s, 1),
        "test_evaluations": GUARD.evaluations,
        "experiment_a": [
            {k: r.get(k) for k in ("run_id", "architecture", "seed", "iterations",
                                   "best_iteration", "val_accuracy", "val_macro_auc",
                                   "gap", "stop_reason", "tuning_seconds")}
            | {"test_accuracy": r["test"]["test_accuracy"],
               "test_macro_auc": r["test"]["test_macro_auc"]}
            for r in rows_a
        ],
        "experiment_b": [
            {k: r.get(k) for k in ("run_id", "architecture", "winner_params_m", "seed",
                                   "iterations", "best_iteration", "val_accuracy",
                                   "val_macro_auc", "gap", "stop_reason",
                                   "search_seconds", "tuning_seconds")}
            | {"test_accuracy": r["test"]["test_accuracy"],
               "test_macro_auc": r["test"]["test_macro_auc"]}
            for r in rows_b
        ],
    }
    write_json(os.path.join(art_root, "manifest.json"), manifest)
    with open(os.path.join(art_root, "README.md"), "w") as fh:
        fh.write(MANIFEST_README)
    write_json(os.path.join(art_root, "experiment_a_full.json"), rows_a)
    write_json(os.path.join(art_root, "experiment_b_full.json"), rows_b)

    print(f"\n{RULE}\n  DONE in {total_s/60:.1f} min | test evaluations: "
          f"{GUARD.evaluations} | artifacts -> {art_root}\n{RULE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
