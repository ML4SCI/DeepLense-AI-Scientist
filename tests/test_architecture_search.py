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
    assert not ArchitectureSpec(name="v", family=ArchFamily.VIT, input_shape=(8, 8)).is_buildable()


def test_build_model_families():
    import torch

    from dlens.tools._torch_backends import build_model

    cnn = ArchitectureSpec(name="c", family=ArchFamily.CNN, input_shape=(16, 16),
                           channels=1, num_classes=2, depths=[1, 1], widths=[8, 16])
    res = ArchitectureSpec(name="r", family=ArchFamily.RESNET, input_shape=(16, 16),
                           channels=1, num_classes=2, depths=[1, 1], widths=[8, 16])
    for spec in (cnn, res):
        out = build_model(spec)(torch.randn(2, 1, 16, 16))
        assert tuple(out.shape) == (2, 2)
    with pytest.raises(ValueError, match="not buildable"):
        build_model(ArchitectureSpec(name="v", family=ArchFamily.VIT, input_shape=(16, 16)))


def test_search_offline(tmp_path: Path):
    from dlens.tools._torch_backends import TorchInferBackend, TorchTrainBackend

    refs = _dataset(tmp_path)
    # 3 candidates; one unbuildable (vit) must be dropped code-side.
    batch1 = [
        _cand("tiny_resnet"),
        {"name": "a_vit", "family": "vit", "input_shape": (16, 16), "channels": 1, "num_classes": 2},
        _cand("tiny_cnn", family="cnn"),
    ]
    batch2 = [_cand("tiny_resnet_v2", widths=[16, 32])]  # refinement variants
    search = ArchitectureSearch(
        generator=ArchitectureGenerator(model=make_scripted_generator([batch1, batch2])),
        judge=ArchitectureJudge(model=make_scripted_judge([1, 0])),
        train_backend=TorchTrainBackend(output_root=str(tmp_path / "m"), device="cpu"),
        infer_backend=TorchInferBackend(device="cpu"),
        num_candidates=3, top_k=2, rounds=2, refine_variants=1, candidate_epochs=1,
    )
    result = asyncio.run(search.search(
        task_description="tiny 2-class test", train_ref=refs["train"], val_ref=refs["val"],
    ))
    assert result.dropped_unbuildable == ["a_vit"]
    assert len(result.rounds) == 2
    assert len(result.rounds[0]) == 2          # top-2 trained in round 1
    assert result.winner.name in {"tiny_resnet", "tiny_cnn", "tiny_resnet_v2"}
    # winner chosen by REAL val accuracy across all evaluated candidates
    all_evals = [e for r in result.rounds for e in r] + [result.winner_eval]
    assert result.winner_eval.val_accuracy == max(e.val_accuracy for e in all_evals)
    assert result.timings["generate_s"] >= 0 and "search_total_s" in result.timings
