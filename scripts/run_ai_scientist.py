"""AI Scientist — full combined run: tree architecture search + closed-loop tuning.

Phase A (tree search): LLM generates ~N candidates -> LLM-as-judge takes top-k ->
short REAL training ranks them -> prune (optional one refinement round) -> winner.
Phase B (closed loop): the EXISTING ReAct hyperparameter tuner takes the winning
architecture and tunes it (dropout/augmentation/early-stop/...) until the
train/val gap closes.

Every phase is timed; output is sectioned for the demo recording.

    OPENAI_API_KEY=... uv run python scripts/run_ai_scientist.py \
        --data-root ~/Personal/GSoC/deeplense_data/model1
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
from dlens.schemas._model_design import TrainingConfig
from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

RULE = "=" * 78


def _ref(root: str, classes: list[str]) -> DatasetRef:
    n = sum(len(glob.glob(os.path.join(root, f"{c}_*.npy"))) for c in classes)
    sample = np.load(sorted(glob.glob(os.path.join(root, f"{classes[0]}_*.npy")))[0])
    return DatasetRef(root=root, class_names=classes, image_shape=tuple(sample.shape), num_samples=n)


async def main() -> int:
    p = argparse.ArgumentParser(description="AI Scientist: architecture search + tuning loop")
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-dir", default="train_small")
    p.add_argument("--val-dir", default="val")
    p.add_argument("--classes", default="no_sub,cdm,axion")
    p.add_argument("--llm", default=None, help=f"LLM id (default env DLENS_LLM or {DEFAULT_LLM})")
    p.add_argument("--num-candidates", type=int, default=10)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2, help="Search rounds (1 = no refinement).")
    p.add_argument("--refine-variants", type=int, default=3)
    p.add_argument("--candidate-epochs", type=int, default=4)
    p.add_argument("--tune-epochs", type=int, default=25, help="Epochs for tuning-loop iterations.")
    p.add_argument("--tune-iterations", type=int, default=4)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    classes = args.classes.split(",")
    train_ref = _ref(os.path.join(args.data_root, args.train_dir), classes)
    val_ref = _ref(os.path.join(args.data_root, args.val_dir), classes)
    llm = build_llm(args.llm)
    llm_id = args.llm or os.getenv("DLENS_LLM") or DEFAULT_LLM

    task = (
        f"Task: {len(classes)}-class image classification of simulated strong "
        f"gravitational-lensing images (classes: {', '.join(classes)}). Images are "
        f"{train_ref.image_shape[0]}x{train_ref.image_shape[1]}, 1 channel. "
        f"{train_ref.num_samples} training / {val_ref.num_samples} validation samples. "
        f"Both underfitting and overfitting have been observed on this data depending "
        f"on configuration — judge capacity on the numbers, not by assumption."
    )

    print(RULE)
    print("  AI SCIENTIST — architecture tree search + closed-loop tuning")
    print(f"  llm={llm_id} | data={train_ref.num_samples} train / {val_ref.num_samples} val "
          f"| {train_ref.image_shape}px")
    print(RULE, flush=True)

    train_backend = TorchTrainBackend(output_root=os.path.join(args.data_root, "search_runs"))
    infer_backend = TorchInferBackend()

    # ---------------- Phase A: tree search ----------------
    print(f"\n{RULE}\n  PHASE A — TREE-BASED ARCHITECTURE SEARCH\n{RULE}", flush=True)
    tA = time.monotonic()
    search = ArchitectureSearch(
        generator=ArchitectureGenerator(model=llm),
        judge=ArchitectureJudge(model=llm),
        train_backend=train_backend,
        infer_backend=infer_backend,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        rounds=args.rounds,
        refine_variants=args.refine_variants,
        candidate_epochs=args.candidate_epochs,
    )
    result = await search.search(task_description=task, train_ref=train_ref, val_ref=val_ref)
    phase_a = time.monotonic() - tA
    print(f"\n  PHASE A winner: {result.winner.name} "
          f"(val={result.winner_eval.val_accuracy:.4f}, ~{result.winner_eval.params_m}M params) "
          f"in {phase_a/60:.1f} min", flush=True)

    # ---------------- Phase B: closed-loop tuning ----------------
    print(f"\n{RULE}\n  PHASE B — CLOSED-LOOP HYPERPARAMETER TUNING (ReAct)\n{RULE}", flush=True)
    tB = time.monotonic()
    loop = ExperimentLoop(
        strategy=ReActPlannerStrategy(ExperimentPlanner(model=llm)),
        train_backend=train_backend,
        infer_backend=infer_backend,
    )
    cfg = TrainingConfig(loss="cross_entropy", optimizer="adamw", learning_rate=3e-4,
                         batch_size=32, epochs=args.tune_epochs, weight_decay=1e-4,
                         lr_scheduler="cosine")
    state = await loop.run(
        hypothesis=(f"Tune the search winner '{result.winner.name}' to close any "
                    "generalization gap on DeepLense Model_I."),
        train_ref=train_ref, val_ref=val_ref,
        architecture=result.winner, training_config=cfg,
        max_iterations=args.tune_iterations,
    )
    phase_b = time.monotonic() - tB

    # ---------------- summary ----------------
    final = state.runs[-1]
    final_train = final.train_result.metrics.get("train_accuracy", float("nan"))
    final_val = final.infer_result.accuracy or float("nan")
    print(f"\n{RULE}\n  SUMMARY\n{RULE}")
    print(f"  winner architecture : {result.winner.name} "
          f"(family={result.winner.family.value}, depths={result.winner.depths}, "
          f"widths={result.winner.widths}, ~{result.winner_eval.params_m}M params)")
    print(f"  final (tuned)       : train={final_train:.4f} val={final_val:.4f} "
          f"gap={final_train - final_val:.4f} auc={final.analysis_result.macro_auc:.4f}")
    print(f"  runtime             : phase A {phase_a/60:.1f} min | phase B {phase_b/60:.1f} min "
          f"| TOTAL {(phase_a + phase_b)/60:.1f} min")
    print(f"  search timings (s)  : { {k: v for k, v in result.timings.items()} }")

    out = args.out_json or os.path.join(args.data_root, "ai_scientist_run.json")
    payload = {
        "llm": llm_id,
        "search": result.model_dump(mode="json"),
        "tuning": state.model_dump(mode="json", exclude={
            "runs": {"__all__": {"infer_result": {"predictions", "probabilities", "true_labels"}}}
        }),
        "runtime_s": {"phase_a": round(phase_a, 1), "phase_b": round(phase_b, 1),
                      "total": round(phase_a + phase_b, 1)},
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  full log -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
