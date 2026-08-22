"""REAL end-to-end classification pipeline on DeepLense data — no mocks.

Agents (LLM reasoning) run on the given OpenAI model; training/inference run
locally with torch (MPS/CPU):

    dataset -> ModelDesignAgent (recommends architecture + hyperparameters)
            -> TrainAgent   (real CNN training, torch)
            -> InferAgent   (real predictions on the held-out val split)
            -> AnalysisAgent (accuracy, macro AUC, confusion matrix, per-class P/R/F1)

Authoritative numbers are taken from each agent's deps-backed ``last_result``
(computed by the tools), never from LLM-transcribed text.

Usage:
    OPENAI_API_KEY=... uv run python scripts/run_classification.py \
        --data-root ~/data/model1 --classes no_sub,cdm,axion --model gpt-5.2
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import time

import numpy as np

from dlens.agents._analysis import AnalysisAgent
from dlens.agents._inference import InferAgent
from dlens.agents._model_design import ModelDesignAgent
from dlens.agents._models import OpenAIModel
from dlens.agents._train import TrainAgent
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import DatasetCharacteristics, TaskType
from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

RULE = "-" * 68


def _ref(root: str, classes: list[str]) -> DatasetRef:
    n = sum(len(glob.glob(os.path.join(root, f"{c}_*.npy"))) for c in classes)
    sample = np.load(sorted(glob.glob(os.path.join(root, f"{classes[0]}_*.npy")))[0])
    return DatasetRef(
        root=root, class_names=classes, image_shape=tuple(sample.shape), num_samples=n
    )


async def main() -> int:
    p = argparse.ArgumentParser(description="Real DeepLense classification pipeline")
    p.add_argument("--data-root", required=True, help="Dir containing train/ and val/.")
    p.add_argument("--classes", default="no_sub,cdm,axion")
    p.add_argument("--model", default="gpt-5.2", help="OpenAI model for the agents' reasoning.")
    p.add_argument("--out-json", default=None, help="Where to write the results JSON.")
    args = p.parse_args()

    classes = args.classes.split(",")
    train_ref = _ref(os.path.join(args.data_root, "train"), classes)
    val_ref = _ref(os.path.join(args.data_root, "val"), classes)
    llm = OpenAIModel(model_name=args.model)

    print(RULE)
    print(f"  dataset: {train_ref.num_samples} train / {val_ref.num_samples} val, "
          f"{len(classes)} classes {classes}, {train_ref.image_shape} px")
    print(f"  agents on: {args.model} | training: torch")
    print(RULE)

    # 1. Model design (gpt reasons over dataset characteristics)
    chars = DatasetCharacteristics(
        task=TaskType.CLASSIFICATION,
        image_shape=train_ref.image_shape,
        channels=1,
        num_classes=len(classes),
        num_samples=train_ref.num_samples,
    )
    t0 = time.monotonic()
    design = await ModelDesignAgent(model=llm).design(chars)
    arch = design.output.architecture
    cfg = design.output.training_config
    print(f"\n[1] ModelDesign ({time.monotonic()-t0:.0f}s): {arch.name} ({arch.family}), "
          f"lr={cfg.learning_rate}, batch={cfg.batch_size}, epochs={cfg.epochs}, loss={cfg.loss}")
    print(f"    reasoning: {design.output.reasoning[:200]}")

    # 2. Real training
    t0 = time.monotonic()
    train_agent = TrainAgent(
        model=llm, backend=TorchTrainBackend(output_root=os.path.join(args.data_root, "runs"))
    )
    await train_agent.train(train_ref, arch, cfg)
    tr = train_agent.last_result
    print(f"\n[2] Train ({time.monotonic()-t0:.0f}s): run={tr.run_id} backend={tr.backend} "
          f"metrics={tr.metrics}")

    # 3. Real inference on held-out val
    t0 = time.monotonic()
    infer_agent = InferAgent(model=llm, backend=TorchInferBackend())
    await infer_agent.infer(tr.weights_path, val_ref)
    ir = infer_agent.last_result
    print(f"\n[3] Infer ({time.monotonic()-t0:.0f}s): {ir.num_samples} samples, "
          f"val accuracy={ir.accuracy:.4f}")

    # 4. Analysis
    t0 = time.monotonic()
    analysis_agent = AnalysisAgent(model=llm)
    report = await analysis_agent.analyze(ir, class_names=classes)
    ar = analysis_agent.last_result
    print(f"\n[4] Analysis ({time.monotonic()-t0:.0f}s): {ar.summary}")
    print(f"    accuracy={ar.accuracy:.4f}  macro_auc={ar.macro_auc:.4f}")
    print(f"    confusion={ar.confusion_matrix}")
    for name, m in ar.per_class.items():
        print(f"    {name:<8} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={int(m['support'])}")
    print(f"    agent message: {report.output.message}")

    payload = {
        "dataset": {"train": train_ref.num_samples, "val": val_ref.num_samples,
                     "classes": classes, "image_shape": list(train_ref.image_shape)},
        "design": design.output.model_dump(mode="json"),
        "train": tr.model_dump(mode="json"),
        "infer_summary": {"num_samples": ir.num_samples, "accuracy": ir.accuracy},
        "analysis": ar.model_dump(mode="json"),
    }
    out = args.out_json or os.path.join(args.data_root, "pipeline_results.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nresults -> {out}")
    print(RULE)
    print("  PIPELINE COMPLETE (real data, real training, real metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
