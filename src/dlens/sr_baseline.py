"""Validate a saved SR baseline against the current validation set."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from dlens.sr_metrics import METRIC_POLICY as METRIC_POLICY
from dlens.sr_metrics import METRICS, matches_metric_policy, summarize_scores


def load_baseline(
    folder: Path, bicubic_rows: list[dict]
) -> tuple[dict[str, float], dict]:
    """Return validated model means and provenance for the same validation pairs.

    The locally recomputed bicubic scores serve as a fingerprint of the image
    contents and metric implementation, beyond matching sample names alone.
    """
    summary_path = folder / "baseline_summary.json"
    csv_path = folder / "validation_per_image.csv"
    summary = json.loads(summary_path.read_text())
    if not matches_metric_policy(summary.get("metric_policy")):
        raise ValueError("baseline uses different metric settings")
    if summary.get("normalization") != "none":
        raise ValueError("baseline must use unnormalized images")
    if summary.get("model") != "Iterative Auto Encoder":
        raise ValueError("unexpected baseline model")
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or len(rows) != summary.get("validation_pairs"):
        raise ValueError("baseline CSV length differs from its summary")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("baseline contains duplicate sample IDs")
    current_ids = [row["sample_id"] for row in bicubic_rows]
    if current_ids != ids[:len(current_ids)]:
        raise ValueError("baseline validation IDs or order differ from this run")
    if len(current_ids) > len(rows):
        raise ValueError("baseline has fewer validation images than this run")

    for imported, current in zip(rows, bicubic_rows):
        for key, column, tolerance in (("psnr_db", "bicubic_psnr_db", 1e-3),
                                       ("ssim", "bicubic_ssim", 2e-4)):
            value = float(imported[column])
            if not math.isfinite(value) or abs(value - float(current[key])) > tolerance:
                raise ValueError(f"bicubic {key} mismatch for {current['sample_id']}")

    selected = rows[:len(current_ids)]
    stats = summarize_scores([{key: float(row[f"model_{key}"]) for key in METRICS}
                              for row in selected])
    means = {key: value["mean"] for key, value in stats.items()}
    if len(selected) == len(rows):
        for key, value in means.items():
            if not math.isclose(value, float(summary["model_validation"][key]),
                                rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"baseline summary {key} differs from CSV")
    provenance = {"name": summary["model"], "summary": str(summary_path.resolve()),
                  "per_image_csv": str(csv_path.resolve()),
                  "checkpoint": str((folder / "best_model.pt").resolve()),
                  "checkpoint_epoch": summary.get("checkpoint_epoch"),
                  "inference_steps": summary.get("inference_steps"),
                  "validation_pairs_used": len(selected)}
    return means, provenance


load_iterative_baseline = load_baseline
