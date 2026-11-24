"""Tree-based architecture search — fully offline (scripted LLMs + tiny training)."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import numpy as np
import pytest

from dlens.agents._architecture_search import (
    ArchitectureGenerator,
    ArchitectureJudge,
    ArchitectureSearch,
    CandidateEval,
)
from dlens.agents._scripted_search import make_scripted_generator, make_scripted_judge
from dlens.schemas._downstream import DatasetRef, InferResult, TrainResult
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec, TrainingConfig

pytest.importorskip("torch")


def _cand(name: str, family: str = "resnet", depths=None, widths=None) -> dict:
    return {
        "name": name, "family": family, "input_shape": (16, 16), "channels": 1,
        "num_classes": 2, "depths": depths or [1, 1], "widths": widths or [8, 16],
        "rationale": "tiny",
    }


def _dataset(tmp_path: Path) -> dict[str, DatasetRef]:
    rng = np.random.default_rng(0)
    refs = {}
    for split in ("train", "val"):
        d = tmp_path / split
        d.mkdir()
        for label, c in enumerate(["a", "b"]):
            for i in range(6):
                base = np.zeros((16, 16)) if label == 0 else np.ones((16, 16)) * 3
                np.save(d / f"{c}_{i:04d}.npy", (base + rng.normal(0, 0.1, (16, 16))).astype(np.float32))
        refs[split] = DatasetRef(root=str(d), class_names=["a", "b"], image_shape=(16, 16), num_samples=12)
    return refs


def test_spec_structure_validation():
    with pytest.raises(Exception):
        ArchitectureSpec(name="x", family=ArchFamily.CNN, input_shape=(8, 8), depths=[2], widths=[8, 16])
    with pytest.raises(Exception):
        ArchitectureSpec(name="x", family=ArchFamily.CNN, input_shape=(8, 8), depths=[2, 9], widths=[8, 16])
    spec = ArchitectureSpec(name="ok", family=ArchFamily.CNN, input_shape=(8, 8),
                            depths=[1, 2], widths=[8, 16])
    assert spec.is_buildable()
    # The buildable space was broadened on 2026-08-13: every declared family now
    # has a real builder, so nothing the generator can name is silently dropped.
    for fam in ArchFamily:
        assert ArchitectureSpec(
            name="x", family=fam, input_shape=(8, 8)
        ).is_buildable(), f"{fam.value} is declared but has no builder"


def test_build_model_families():
    import torch

    from dlens.tools._torch_backends import build_model

    cnn = ArchitectureSpec(name="c", family=ArchFamily.CNN, input_shape=(16, 16),
                           channels=1, num_classes=2, depths=[1, 1], widths=[8, 16])
    res = ArchitectureSpec(name="r", family=ArchFamily.RESNET, input_shape=(16, 16),
                           channels=1, num_classes=2, depths=[1, 1], widths=[8, 16])
    vit = ArchitectureSpec(name="v", family=ArchFamily.VIT, input_shape=(16, 16),
                           channels=1, num_classes=2, depths=[1, 1], widths=[8, 16])
    for spec in (cnn, res, vit):
        out = build_model(spec)(torch.randn(2, 1, 16, 16))
        assert tuple(out.shape) == (2, 2)
    # Per-family build/train coverage lives in tests/test_arch_families.py.


def test_search_offline(tmp_path: Path):
    from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

    refs = _dataset(tmp_path)
    batch1 = [_cand("tiny_resnet"), _cand("tiny_cnn", family="cnn")]
    # Round 2 re-proposes tiny_resnet verbatim (must be dropped as a duplicate,
    # since seeded training would only reproduce the same numbers) plus one novel
    # architecture the search has not seen.
    batch2 = [_cand("tiny_resnet"), _cand("tiny_resnet_v2", widths=[16, 32])]
    search = ArchitectureSearch(
        generator=ArchitectureGenerator(model=make_scripted_generator([batch1, batch2])),
        judge=ArchitectureJudge(model=make_scripted_judge([1, 0])),
        train_backend=TorchTrainBackend(output_root=str(tmp_path / "m"), device="cpu"),
        infer_backend=TorchInferBackend(device="cpu"),
        num_candidates=2, top_k=2, rounds=2, candidate_epochs=1,
    )
    result = asyncio.run(search.search(
        task_description="tiny 2-class test", train_ref=refs["train"], val_ref=refs["val"],
    ))
    assert len(result.rounds) == 2
    assert len(result.rounds[0]) == 2                     # top-2 trained in round 1
    assert result.round_records[1].dropped_duplicate == ["tiny_resnet"]
    assert result.round_records[1].shortlist == ["tiny_resnet_v2"]
    assert result.winner.name in {"tiny_resnet", "tiny_cnn", "tiny_resnet_v2"}
    # winner chosen by REAL val accuracy across all evaluated candidates
    all_evals = [e for r in result.rounds for e in r] + [result.winner_eval]
    assert result.winner_eval.val_accuracy == max(e.val_accuracy for e in all_evals)
    assert result.timings["round1_generate_s"] >= 0 and "search_total_s" in result.timings


def test_search_feeds_history_into_later_rounds(tmp_path: Path):
    """Round 2's generator prompt must carry round 1's measured results."""
    from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

    refs = _dataset(tmp_path)
    seen_prompts: list[str] = []

    class Recording(ArchitectureGenerator):
        async def arun(self, prompt, *a, **kw):          # type: ignore[override]
            seen_prompts.append(prompt)
            return await super().arun(prompt, *a, **kw)

    search = ArchitectureSearch(
        generator=Recording(model=make_scripted_generator(
            [[_cand("r1_a"), _cand("r1_b", family="cnn")], [_cand("r2_a", widths=[16, 32])]]
        )),
        judge=ArchitectureJudge(model=make_scripted_judge([0, 1])),
        train_backend=TorchTrainBackend(output_root=str(tmp_path / "m2"), device="cpu"),
        infer_backend=TorchInferBackend(device="cpu"),
        num_candidates=2, top_k=2, rounds=2, candidate_epochs=1,
    )
    asyncio.run(search.search(
        task_description="tiny 2-class test", train_ref=refs["train"], val_ref=refs["val"],
    ))
    assert len(seen_prompts) == 2
    assert "MEASURED RESULTS SO FAR" not in seen_prompts[0]   # round 1 is blind
    assert "MEASURED RESULTS SO FAR" in seen_prompts[1]       # round 2 is informed
    assert "r1_a" in seen_prompts[1] and "val_accuracy=" in seen_prompts[1]


class _MetricTrainBackend:
    name = "metric-test"

    def train(self, dataset, architecture, config):
        return TrainResult(
            run_id="train",
            weights_path="weights",
            backend=self.name,
            num_classes=2,
            metrics={"train_accuracy": 0.0},
            epochs_run=1,
        )


class _MetricInferBackend:
    name = "metric-test"

    def __init__(self, accuracies, probabilities=None):
        self._accuracies = iter(accuracies)
        self._probabilities = probabilities

    def infer(self, weights_path, dataset):
        accuracy = next(self._accuracies)
        correct = round(accuracy * 4)
        true = [0, 0, 1, 1]
        predictions = [true[i] if i < correct else (true[i] + 1) % 2 for i in range(4)]
        return InferResult(
            run_id="infer",
            num_samples=4,
            num_classes=2,
            predictions=predictions,
            probabilities=self._probabilities,
            true_labels=true,
            accuracy=accuracy,
            backend=self.name,
        )


class _MissingLabelsInferBackend:
    name = "metric-test"

    def infer(self, weights_path, dataset):
        return InferResult(
            run_id="infer", num_samples=4, num_classes=2,
            predictions=[0, 1, 0, 1], true_labels=None, accuracy=None,
            backend=self.name,
        )


def _metric_search(accuracies, probabilities=None):
    return ArchitectureSearch(
        generator=object(),
        judge=object(),
        train_backend=_MetricTrainBackend(),
        infer_backend=_MetricInferBackend(accuracies, probabilities),
        num_candidates=len(accuracies),
        top_k=len(accuracies),
    )


def _missing_labels_search():
    return ArchitectureSearch(
        generator=object(),
        judge=object(),
        train_backend=_MetricTrainBackend(),
        infer_backend=_MissingLabelsInferBackend(),
        num_candidates=1,
        top_k=1,
    )


def _metric_specs(names):
    return [
        ArchitectureSpec(
            name=name,
            family=ArchFamily.CNN,
            input_shape=(8, 8),
            depths=[1, 1],
            widths=[8, 16],
        )
        for name in names
    ]


def test_search_evaluation_preserves_zero_accuracy(monkeypatch):
    search = _metric_search([0.0])
    monkeypatch.setattr("dlens.agents._architecture_search._params_m", lambda spec: 0.1)
    dataset = DatasetRef(root="unused", class_names=["a", "b"], image_shape=(8, 8), num_samples=4)

    evals = asyncio.run(search._evaluate(
        _metric_specs(["zero"]), dataset, dataset, {}, tag="test"
    ))

    assert evals[0].val_accuracy == 0.0
    assert math.isfinite(evals[0].val_accuracy)


def test_search_evaluation_rejects_missing_ground_truth(monkeypatch):
    search = _missing_labels_search()
    monkeypatch.setattr("dlens.agents._architecture_search._params_m", lambda spec: 0.1)
    dataset = DatasetRef(root="unused", class_names=["a", "b"], image_shape=(8, 8), num_samples=4)

    with pytest.raises(ValueError, match="true_labels"):
        asyncio.run(search._evaluate(
            _metric_specs(["missing"]), dataset, dataset, {}, tag="test"
        ))


def test_search_evaluation_preserves_zero_macro_auc_and_logs_missing_auc(
    monkeypatch, capsys
):
    inverse_probabilities = [
        [0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]
    ]
    search = _metric_search([0.0], inverse_probabilities)
    monkeypatch.setattr("dlens.agents._architecture_search._params_m", lambda spec: 0.1)
    dataset = DatasetRef(root="unused", class_names=["a", "b"], image_shape=(8, 8), num_samples=4)

    evals = asyncio.run(search._evaluate(
        _metric_specs(["zero"]), dataset, dataset, {}, tag="test"
    ))

    assert evals[0].macro_auc == 0.0
    assert "auc=0.0000" in capsys.readouterr().out

    no_auc_search = _metric_search([0.0])
    asyncio.run(no_auc_search._evaluate(
        _metric_specs(["zero"]), dataset, dataset, {}, tag="test"
    ))
    assert "auc=n/a" in capsys.readouterr().out


def test_best_positive_accuracy_beats_zero_and_is_order_independent():
    specs = _metric_specs(["zero", "positive"])
    evals = [
        CandidateEval(
            name="zero", params_m=0.1, train_accuracy=0.0, val_accuracy=0.0,
            gap=0.0, macro_auc=None, train_seconds=0.0,
        ),
        CandidateEval(
            name="positive", params_m=0.1, train_accuracy=0.25, val_accuracy=0.25,
            gap=0.0, macro_auc=None, train_seconds=0.0,
        ),
    ]

    winner, _ = ArchitectureSearch._best(specs, evals)
    reverse_winner, _ = ArchitectureSearch._best(specs[::-1], evals[::-1])

    assert winner.name == "positive"
    assert reverse_winner.name == "positive"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_best_rejects_empty_mismatched_and_nonfinite_evaluations(score):
    specs = _metric_specs(["a", "b"])
    finite = CandidateEval(
        name="a", params_m=0.1, train_accuracy=0.0, val_accuracy=0.0,
        gap=0.0, macro_auc=None, train_seconds=0.0,
    )
    invalid = finite.model_copy(update={"name": "b", "val_accuracy": score})

    with pytest.raises(ValueError, match="at least one candidate"):
        ArchitectureSearch._best([], [])
    with pytest.raises(ValueError, match="same length"):
        ArchitectureSearch._best(specs, [finite])
    with pytest.raises(ValueError, match="finite"):
        ArchitectureSearch._best(specs, [invalid, invalid])


def test_best_keeps_first_candidate_for_equal_finite_scores():
    specs = _metric_specs(["first", "second"])
    evals = [
        CandidateEval(
            name="first", params_m=0.1, train_accuracy=0.25, val_accuracy=0.25,
            gap=0.0, macro_auc=None, train_seconds=0.0,
        ),
        CandidateEval(
            name="second", params_m=0.1, train_accuracy=0.25, val_accuracy=0.25,
            gap=0.0, macro_auc=None, train_seconds=0.0,
        ),
    ]

    winner, _ = ArchitectureSearch._best(specs, evals)

    assert winner.name == "first"


def test_search_keeps_global_winner_across_rounds(monkeypatch):
    batches = [[_cand("round1")], [_cand("round2", widths=[16, 32])]]
    search = ArchitectureSearch(
        generator=ArchitectureGenerator(model=make_scripted_generator(batches)),
        judge=ArchitectureJudge(model=make_scripted_judge([0, 0])),
        train_backend=_MetricTrainBackend(),
        infer_backend=_MetricInferBackend([]),
        num_candidates=1, top_k=1, rounds=2,
    )
    monkeypatch.setattr(
        "dlens.agents._architecture_search._params_m", lambda spec: 0.1
    )
    scores = iter([0.0, 0.25])

    async def evaluate(specs, train_ref, val_ref, timings, tag):
        score = next(scores)
        return [CandidateEval(
            name=specs[0].name, params_m=0.1, train_accuracy=score,
            val_accuracy=score, gap=0.0, macro_auc=None, train_seconds=0.0,
        )]

    search._evaluate = evaluate
    dataset = DatasetRef(root="unused", class_names=["a", "b"], image_shape=(8, 8), num_samples=4)
    result = asyncio.run(search.search(
        task_description="round winner test", train_ref=dataset, val_ref=dataset
    ))

    assert result.winner.name == "round2"
    assert result.winner_eval.val_accuracy == 0.25


@pytest.mark.parametrize("accuracy", [None, float("nan"), float("inf"), float("-inf")])
def test_best_run_rejects_invalid_accuracy_values(accuracy):
    from dlens.agents._experiment_loop import best_run_by_val_accuracy
    from dlens.schemas._experiment import ExperimentRun, ExperimentState

    run = ExperimentRun(
        iteration=0,
        infer_result=InferResult(
            run_id="infer", num_samples=1, num_classes=2,
            predictions=[0], true_labels=[0], accuracy=accuracy,
        ),
    )
    with pytest.raises(ValueError, match="finite accuracy"):
        best_run_by_val_accuracy(ExperimentState(
            hypothesis="invalid metrics", max_iterations=1, runs=[run]
        ))


def test_best_run_preserves_zero_and_prefers_higher_finite_accuracy():
    from dlens.agents._experiment_loop import best_run_by_val_accuracy
    from dlens.schemas._experiment import ExperimentRun, ExperimentState

    def run(iteration, accuracy):
        return ExperimentRun(
            iteration=iteration,
            infer_result=InferResult(
                run_id=f"infer-{iteration}", num_samples=1, num_classes=2,
                predictions=[0], true_labels=[0], accuracy=accuracy,
            ),
        )

    state = ExperimentState(
        hypothesis="mixed metrics", max_iterations=4,
        runs=[run(0, float("nan")), run(1, 0.0), run(2, float("inf")), run(3, 0.75)],
    )
    best = best_run_by_val_accuracy(state)

    assert best is state.runs[3]


@pytest.mark.parametrize(
    ("macro_auc", "expected"),
    [(None, "auc=n/a"), (0.0, "auc=0.0000"), (0.375, "auc=0.3750"),
     (float("nan"), "auc=invalid"), (float("inf"), "auc=invalid")],
)
def test_experiment_loop_auc_reporting(monkeypatch, capsys, macro_auc, expected):
    from dlens.agents._experiment_loop import ExperimentLoop
    from dlens.schemas._downstream import AnalysisResult
    from dlens.tools._inference import MockInferBackend
    from dlens.tools._training import MockTrainBackend

    monkeypatch.setattr(
        "dlens.agents._experiment_loop.compute_analysis",
        lambda infer, class_names: AnalysisResult(
            accuracy=0.75, macro_auc=macro_auc, confusion_matrix=[[1, 0], [0, 1]],
            per_class={}, summary="test",
        ),
    )
    loop = ExperimentLoop(
        strategy=object(), train_backend=MockTrainBackend(),
        infer_backend=MockInferBackend(),
    )
    dataset = DatasetRef(root="unused", class_names=["a", "b"], image_shape=(8, 8), num_samples=8)
    asyncio.run(loop.run(
        hypothesis="auc reporting", train_ref=dataset, val_ref=dataset,
        architecture=_metric_specs(["auc"])[0],
        training_config=TrainingConfig(loss="cross_entropy", epochs=1, batch_size=8),
        max_iterations=1,
    ))

    assert expected in capsys.readouterr().out
