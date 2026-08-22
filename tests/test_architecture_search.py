"""Tree-based architecture search — fully offline (scripted LLMs + tiny training)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from dlens.agents._architecture_search import (
    ArchitectureGenerator,
    ArchitectureJudge,
    ArchitectureSearch,
)
from dlens.agents._scripted_search import make_scripted_generator, make_scripted_judge
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import ArchFamily, ArchitectureSpec

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
