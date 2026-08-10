"""THE definitive paper run — one clean end-to-end experiment, gpt-5.2 everywhere.

Protocol (all on train/val; the held-out TEST set is touched exactly once, at the end):

  A.  Tree architecture search (generate 10 -> LLM-judge top-4 -> 10-epoch real
      training -> prune -> one refine round) ................ selection on VAL
  B1. ReAct loop tunes the search winner ................... selection on VAL
  B2. ReAct loop tunes the resnet34 reference (same protocol, same LLM)
  C.  FINAL TEST — the only step that loads model1_test: evaluate the best-val
      checkpoint of BOTH arms once. No retraining, no selection after this.

The test set is quarantined in code: its DatasetRef is constructed inside
final_test_evaluation(), which asserts both arms are complete first.

    DLENS_LLM=gpt-5.2 OPENAI_API_KEY=... uv run python scripts/run_paper_experiment.py \
        --data-root ~/Personal/GSoC/deeplense_data/model1 \
        --test-root ~/Personal/GSoC/deeplense_data/model1_test
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import time

import numpy as np

from dlens.agents._architecture_search import ArchitectureGenerator, ArchitectureJudge, ArchitectureSearch
from dlens.agents._experiment_loop import ExperimentLoop
from dlens.agents._experiment_planner import ExperimentPlanner, ReActPlannerStrategy
from dlens.config import DEFAULT_LLM, build_llm
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._experiment import ExperimentState
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig
from dlens.tools._analysis import compute_analysis
from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

RULE = "=" * 78
_ARMS_COMPLETE = {"done": False}  # set only after both tuning arms finish


def _ref(root: str, classes: list[str]) -> DatasetRef:
    n = sum(len(glob.glob(os.path.join(root, f"{c}_*.npy"))) for c in classes)
    sample = np.load(sorted(glob.glob(os.path.join(root, f"{classes[0]}_*.npy")))[0])
    return DatasetRef(root=root, class_names=classes, image_shape=tuple(sample.shape), num_samples=n)


def _best_val_run(state: ExperimentState):
    """Select the iteration with the best VAL accuracy (selection happens on val only)."""
    return max(state.runs, key=lambda r: r.infer_result.accuracy or -1.0)


def _print_state(tag: str, state: ExperimentState) -> None:
    for r in state.runs:
        tr = r.train_result.metrics.get("train_accuracy", float("nan"))
        va = r.infer_result.accuracy or float("nan")
        d = r.planner_decision
        print(f"    [{tag}] it{r.iteration}: train={tr:.4f} val={va:.4f} gap={tr - va:+.4f} "
              f"auc={r.analysis_result.macro_auc:.4f} -> {d.action.value if d else '-'} "
              f"{d.updated_params if d else ''}", flush=True)


def final_test_evaluation(test_root: str, classes: list[str], arms: dict[str, dict]) -> dict:
    """THE ONLY place the test set is loaded. Asserts both arms are complete."""
    assert _ARMS_COMPLETE["done"], "test evaluation attempted before both arms completed"
    test_ref = _ref(test_root, classes)
    print(f"\n{RULE}\n  PHASE C — FINAL TEST (single evaluation; {test_ref.num_samples} images)\n{RULE}")
    infer = TorchInferBackend()
    out: dict = {}
    for arm, info in arms.items():
        ir = infer.infer(info["weights_path"], test_ref)
        ar = compute_analysis(ir, classes)
        out[arm] = {"test_accuracy": ir.accuracy, "test_macro_auc": ar.macro_auc,
                    "per_class": ar.per_class, "confusion": ar.confusion_matrix}
        print(f"  {arm:<16} val={info['val_accuracy']:.4f}  TEST acc={ir.accuracy:.4f}  "
              f"TEST auc={ar.macro_auc:.4f}", flush=True)
    return out


async def main() -> int:
    p = argparse.ArgumentParser(description="Definitive paper run (gpt-5.2 end to end)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--train-dir", default="train_small")
    p.add_argument("--val-dir", default="val")
    p.add_argument("--classes", default="no_sub,cdm,axion")
    p.add_argument("--num-candidates", type=int, default=10)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--refine-variants", type=int, default=3)
    p.add_argument("--candidate-epochs", type=int, default=10)
    p.add_argument("--tune-epochs", type=int, default=25)
    p.add_argument("--tune-iterations", type=int, default=4)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    classes = args.classes.split(",")
    train_ref = _ref(os.path.join(args.data_root, args.train_dir), classes)
    val_ref = _ref(os.path.join(args.data_root, args.val_dir), classes)

    llm = build_llm()  # from DLENS_LLM
    llm_id = os.getenv("DLENS_LLM") or DEFAULT_LLM
    import torch
    print(RULE)
    print("  DEFINITIVE PAPER RUN — gpt-5.2 end to end, held-out test evaluated once")
    print(f"  effective LLM: {type(llm).__name__} model_name={getattr(llm, 'model_name', '?')}")
    print(f"  torch {torch.__version__} | mps={torch.backends.mps.is_available()}")
    print(f"  data: {train_ref.num_samples} train / {val_ref.num_samples} val "
          f"(test loaded only in Phase C) | {train_ref.image_shape}px")
    print(RULE, flush=True)
    assert getattr(llm, "model_name", "") == "gpt-5.2", "this run requires DLENS_LLM=gpt-5.2"

    train_backend = TorchTrainBackend(output_root=os.path.join(args.data_root, "paper_runs"))
    infer_backend = TorchInferBackend()
    timings: dict[str, float] = {}

    # ---------- Phase A: tree search (selection on val) ----------
    print(f"\n{RULE}\n  PHASE A — TREE ARCHITECTURE SEARCH (gpt-5.2)\n{RULE}", flush=True)
    t0 = time.monotonic()
    search = ArchitectureSearch(
        generator=ArchitectureGenerator(model=llm), judge=ArchitectureJudge(model=llm),
        train_backend=train_backend, infer_backend=infer_backend,
        num_candidates=args.num_candidates, top_k=args.top_k, rounds=args.rounds,
        refine_variants=args.refine_variants, candidate_epochs=args.candidate_epochs,
    )
    task = (
        f"Task: {len(classes)}-class image classification of simulated strong "
        f"gravitational-lensing images (classes: {', '.join(classes)}). Images are "
        f"{train_ref.image_shape[0]}x{train_ref.image_shape[1]}, 1 channel. "
        f"{train_ref.num_samples} training / {val_ref.num_samples} validation samples. "
        f"Both underfitting and overfitting have been observed on this data depending "
        f"on configuration — judge capacity on the numbers, not by assumption."
    )
    result = await search.search(task_description=task, train_ref=train_ref, val_ref=val_ref)
    timings["phase_a_s"] = round(time.monotonic() - t0, 1)

    # ---------- Phase B: two tuning arms, same protocol ----------
    base_cfg = TrainingConfig(loss="cross_entropy", optimizer="adamw", learning_rate=3e-4,
                              batch_size=32, epochs=args.tune_epochs, weight_decay=1e-4,
                              lr_scheduler="cosine")
    arms_specs = {
        "search_winner": result.winner,
        "resnet34_ref": ArchitectureSpec(
            name="resnet34", family=ArchFamily.RESNET,
            input_shape=train_ref.image_shape, channels=1, num_classes=len(classes),
        ),
    }
    arm_states: dict[str, ExperimentState] = {}
    arms_final: dict[str, dict] = {}
    for arm, spec in arms_specs.items():
        print(f"\n{RULE}\n  PHASE B — TUNING ARM: {arm} ({spec.name})\n{RULE}", flush=True)
        t0 = time.monotonic()
        loop = ExperimentLoop(
            strategy=ReActPlannerStrategy(ExperimentPlanner(model=llm)),
            train_backend=train_backend, infer_backend=infer_backend,
        )
        state = await loop.run(
            hypothesis=f"Tune '{spec.name}' to close any generalization gap (arm: {arm}).",
            train_ref=train_ref, val_ref=val_ref,
            architecture=spec, training_config=base_cfg,
            max_iterations=args.tune_iterations,
        )
        timings[f"phase_b_{arm}_s"] = round(time.monotonic() - t0, 1)
        arm_states[arm] = state
        best = _best_val_run(state)
        arms_final[arm] = {
            "weights_path": best.train_result.weights_path,
            "val_accuracy": best.infer_result.accuracy,
            "val_macro_auc": best.analysis_result.macro_auc,
            "val_per_class": best.analysis_result.per_class,
            "val_confusion": best.analysis_result.confusion_matrix,
            "selected_iteration": best.iteration,
            "architecture": best.architecture.model_dump(mode="json"),
            "training_config": best.training_config.model_dump(mode="json"),
        }
        print(f"  arm '{arm}': selected iteration {best.iteration} by val "
              f"(val={best.infer_result.accuracy:.4f})", flush=True)

    _ARMS_COMPLETE["done"] = True

    # ---------- Phase C: the single test evaluation ----------
    t0 = time.monotonic()
    test_results = final_test_evaluation(args.test_root, classes, arms_final)
    timings["phase_c_s"] = round(time.monotonic() - t0, 1)
    timings["total_s"] = round(sum(v for k, v in timings.items() if k != "total_s"), 1)

    # ---------- summary ----------
    print(f"\n{RULE}\n  SUMMARY (val = selection metric; test = reported once)\n{RULE}")
    print(f"  {'arm':<16} {'val acc':>8} {'test acc':>9} {'val auc':>8} {'test auc':>9}")
    for arm, info in arms_final.items():
        t = test_results[arm]
        print(f"  {arm:<16} {info['val_accuracy']:>8.4f} {t['test_accuracy']:>9.4f} "
              f"{info['val_macro_auc']:>8.4f} {t['test_macro_auc']:>9.4f}")
    print(f"  timings: {timings}")

    out = args.out_json or os.path.join(args.data_root, "paper_run.json")
    payload = {
        "llm": llm_id,
        "search": result.model_dump(mode="json"),
        "arms": {
            arm: {
                "final": arms_final[arm],
                "test": test_results[arm],
                "trajectory": arm_states[arm].model_dump(mode="json", exclude={
                    "runs": {"__all__": {"infer_result": {"predictions", "probabilities", "true_labels"}}}
                }),
            } for arm in arms_final
        },
        "timings_s": timings,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  full log -> {out}")
    for arm, st in arm_states.items():
        print(f"\n  trajectory [{arm}]:")
        _print_state(arm, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
