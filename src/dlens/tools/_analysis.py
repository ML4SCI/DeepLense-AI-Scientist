# tools/_analysis.py
"""Analysis tool for the Analysis Agent.

Pure, deterministic numpy evaluation of an ``InferResult`` against its ground-truth
labels — accuracy, confusion matrix, per-class precision/recall/F1/support, and a
macro one-vs-rest ROC AUC. Fully offline by construction (no model, no randomness),
so no separate mock backend is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic_ai import Agent, RunContext

from dlens.schemas._downstream import AnalysisResult, InferResult


def _macro_auc(probs: np.ndarray, y: np.ndarray, k: int) -> Optional[float]:
    """Macro one-vs-rest ROC AUC via the rank (Mann-Whitney) statistic."""
    aucs: list[float] = []
    for c in range(k):
        pos = y == c
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        order = probs[:, c].argsort()
        ranks = np.empty(len(y), dtype=np.float64)
        ranks[order] = np.arange(1, len(y) + 1)
        aucs.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else None


def compute_analysis(infer: InferResult, class_names: Optional[list[str]] = None) -> AnalysisResult:
    """Evaluate predictions against ground truth into an :class:`AnalysisResult`."""
    if infer.true_labels is None:
        raise ValueError("AnalysisResult requires InferResult.true_labels to be set.")
    k = infer.num_classes
    names = class_names or [str(i) for i in range(k)]
    y = np.asarray(infer.true_labels)
    pred = np.asarray(infer.predictions)

    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1

    per_class: dict[str, dict[str, float]] = {}
    for c in range(k):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[names[c]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": float(int(cm[c, :].sum())),
        }

    accuracy = float((y == pred).mean())
    macro_auc = None
    if infer.probabilities is not None:
        macro_auc = _macro_auc(np.asarray(infer.probabilities), y, k)

    worst = min(per_class.items(), key=lambda kv: kv[1]["recall"])[0] if per_class else "-"
    summary = (
        f"accuracy={accuracy:.3f}"
        + (f", macro_auc={macro_auc:.3f}" if macro_auc is not None else "")
        + f"; weakest recall on class '{worst}'."
    )
    return AnalysisResult(
        accuracy=accuracy,
        macro_auc=macro_auc,
        confusion_matrix=cm.tolist(),
        per_class=per_class,
        summary=summary,
    )


@dataclass
class AnalysisDeps:
    """Runtime dependencies for the analysis agent.

    The inference result is injected here rather than passed as a tool argument:
    for real datasets the prediction arrays are far too large to round-trip through
    the model context, and the numbers must not be LLM-transcribed anyway.
    """

    infer_result: InferResult
    class_names: Optional[list[str]] = None
    last_result: Optional[AnalysisResult] = None


def register_analysis_tool(agent: Agent) -> None:
    """Register the ``analyze_predictions`` tool onto ``agent``."""

    @agent.tool
    def analyze_predictions(ctx: RunContext[AnalysisDeps]) -> dict:
        """Compute accuracy, confusion matrix, per-class metrics and macro AUC for the
        inference result attached to this run. Takes no arguments."""
        result = compute_analysis(ctx.deps.infer_result, ctx.deps.class_names)
        ctx.deps.last_result = result
        return result.model_dump(mode="json")
