"""Predeclared validation policy for super resolution experiments."""

from __future__ import annotations

from math import isfinite

from dlens.sr_metrics import METRICS


def passes_baseline(run: dict, baseline: dict) -> bool:
    return all(isfinite(float(run[key])) and float(run[key]) > float(baseline[key])
               for key in METRICS)


def _ranking(runs: list[dict]) -> tuple[list[dict], dict[str, dict[str, int]]]:
    ranks: dict[str, dict[str, int]] = {run["id"]: {} for run in runs}
    for metric in METRICS:
        previous = None
        rank = 0
        for position, run in enumerate(sorted(runs, key=lambda r: -float(r[metric])), start=1):
            value = float(run[metric])
            if value != previous:
                rank = position
                previous = value
            ranks[run["id"]][metric] = rank
    return sorted(runs, key=lambda run: (
        sum(ranks[run["id"]].values()), int(run["parameters"]), str(run["id"]),
    )), ranks


def _eligible(runs: list[dict], baseline: dict) -> list[dict]:
    for key in METRICS:
        if not isfinite(float(baseline[key])):
            raise ValueError(f"baseline {key} must be finite")
    return [run for run in runs if passes_baseline(run, baseline)]


def rank_runs(runs: list[dict], baseline: dict) -> list[dict]:
    """Rank models that beat both validation metrics, with size breaking ties."""
    return _ranking(_eligible(runs, baseline))[0]


def rank_with_fallback(runs: list[dict], baseline: dict) -> list[dict]:
    """List eligible models first, then the remaining finite results for exploration."""
    eligible = rank_runs(runs, baseline)
    eligible_ids = {run["id"] for run in eligible}
    finite = [run for run in runs if all(isfinite(float(run[key])) for key in METRICS)]
    return eligible + [run for run in _ranking(finite)[0] if run["id"] not in eligible_ids]


def ranking_details(runs: list[dict], baseline: dict) -> list[dict]:
    """Return auditable metric ranks and baseline checks in input order."""
    _, ranks = _ranking(_eligible(runs, baseline))
    return [{"id": run["id"], "passes_baseline": run["id"] in ranks,
             "metric_ranks": ranks.get(run["id"], {}),
             "rank_sum": sum(ranks[run["id"]].values()) if run["id"] in ranks else None}
            for run in runs]
