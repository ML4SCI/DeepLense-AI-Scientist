"""REAL closed-loop experiment-planner run on DeepLense Model_I data — no mocks.

Starts from the configuration that overfit in the baseline classification run
(resnet34, no regularization) and lets the ReAct planner (gpt-5.2) iterate:
observe train-vs-val metrics -> decide ONE change -> retrain -> re-score.

Usage:
    OPENAI_API_KEY=... uv run python scripts/run_experiment_loop.py \
        --data-root ~/data/model1 --train-dir train_small --model gpt-5.2
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import time

import numpy as np

from dlens.agents._experiment_loop import ExperimentLoop
from dlens.agents._experiment_planner import ExperimentPlanner, ReActPlannerStrategy
from dlens.agents._models import OpenAIModel
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig
from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend


def _ref(root: str, classes: list[str]) -> DatasetRef:
    n = sum(len(glob.glob(os.path.join(root, f"{c}_*.npy"))) for c in classes)
    sample = np.load(sorted(glob.glob(os.path.join(root, f"{classes[0]}_*.npy")))[0])
    return DatasetRef(root=root, class_names=classes, image_shape=tuple(sample.shape), num_samples=n)


async def main() -> int:
    p = argparse.ArgumentParser(description="Closed-loop experiment planner (real run)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-dir", default="train_small", help="Subdir with the training subset.")
    p.add_argument("--val-dir", default="val")
    p.add_argument("--classes", default="no_sub,cdm,axion")
    p.add_argument("--model", default="gpt-5.2")
    p.add_argument("--max-iterations", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12, help="Baseline epochs (iteration 0).")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    classes = args.classes.split(",")
    train_ref = _ref(os.path.join(args.data_root, args.train_dir), classes)
    val_ref = _ref(os.path.join(args.data_root, args.val_dir), classes)

    # Iteration-0 config: what overfit in the baseline run (no regularization).
    arch = ArchitectureSpec(
        name="resnet34", family=ArchFamily.RESNET,
        input_shape=train_ref.image_shape, channels=1, num_classes=len(classes),
    )
    cfg = TrainingConfig(
        loss="cross_entropy", optimizer="adamw", learning_rate=3e-4,
        batch_size=32, epochs=args.epochs, weight_decay=1e-4, lr_scheduler="cosine",
    )

    print(f"data: {train_ref.num_samples} train / {val_ref.num_samples} val "
          f"({len(classes)} classes, {train_ref.image_shape}px) | planner: {args.model}")

    loop = ExperimentLoop(
        strategy=ReActPlannerStrategy(ExperimentPlanner(model=OpenAIModel(model_name=args.model))),
        train_backend=TorchTrainBackend(output_root=os.path.join(args.data_root, "loop_runs")),
        infer_backend=TorchInferBackend(),
    )
    t0 = time.monotonic()
    state = await loop.run(
        hypothesis=(
            "Starting from a resnet34 that overfits DeepLense Model_I (train>>val), "
            "can iterative planner-driven changes close the generalization gap?"
        ),
        train_ref=train_ref, val_ref=val_ref,
        architecture=arch, training_config=cfg,
        max_iterations=args.max_iterations,
    )
    wall = time.monotonic() - t0

    # ---- trajectory table ----
    print("\n" + "=" * 78)
    print(f"TRAJECTORY ({len(state.runs)} iterations, {wall/60:.1f} min total)")
    print(f"{'it':>2} | {'change chosen':<38} | {'train':>6} | {'val':>6} | {'gap':>6} | {'auc':>6}")
    for run in state.runs:
        tr_acc = run.train_result.metrics.get("train_accuracy", float("nan"))
        val_acc = run.infer_result.accuracy or float("nan")
        auc = run.analysis_result.macro_auc or float("nan")
        d = run.planner_decision
        change = "(baseline)" if run.iteration == 0 else "?"
        # the change shown on row i is what the PREVIOUS decision set up
        prev = state.runs[run.iteration - 1].planner_decision if run.iteration > 0 else None
        if prev is not None:
            change = json.dumps(prev.updated_params)[:38]
        print(f"{run.iteration:>2} | {change:<38} | {tr_acc:.4f} | {val_acc:.4f} | "
              f"{tr_acc - val_acc:.4f} | {auc:.4f}")
        if d is not None:
            print(f"     decision: {d.action.value} {json.dumps(d.updated_params)}")
            print(f"     reasoning: {d.rationale[:220]}")
    print("=" * 78)

    out = args.out_json or os.path.join(args.data_root, "experiment_loop_trajectory.json")
    compact = state.model_dump(
        mode="json",
        exclude={"runs": {"__all__": {"infer_result": {"predictions", "probabilities", "true_labels"}}}},
    )
    with open(out, "w") as fh:
        json.dump(compact, fh, indent=2)
    print(f"trajectory -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
