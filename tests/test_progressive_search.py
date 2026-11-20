import math

import pytest

from dlens.agents._progressive_search import select_survivors


def test_selects_top_half():
    scores = {
        "cnn": 0.70,
        "resnet": 0.85,
        "vit": 0.60,
        "mlp": 0.75,
    }

    assert select_survivors(scores) == ["resnet", "mlp"]


def test_ties_are_broken_by_name():
    scores = {"z_model": 0.8, "a_model": 0.8, "b_model": 0.7}

    assert select_survivors(scores, keep_fraction=0.5) == ["a_model", "z_model"]


def test_keeps_at_least_one_candidate():
    assert select_survivors({"only": 0.5}) == ["only"]


def test_keep_fraction_one_keeps_every_candidate():
    scores = {"b": 0.6, "a": 0.9}

    assert select_survivors(scores, keep_fraction=1.0) == ["a", "b"]


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, math.nan])
def test_rejects_invalid_keep_fraction(fraction):
    with pytest.raises(ValueError, match="keep_fraction"):
        select_survivors({"cnn": 0.5}, keep_fraction=fraction)


def test_rejects_empty_scores():
    with pytest.raises(ValueError, match="At least one candidate"):
        select_survivors({})


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_scores(score):
    with pytest.raises(ValueError, match="must be finite"):
        select_survivors({"cnn": score})


def test_rejects_blank_candidate_name():
    with pytest.raises(ValueError, match="names must not be empty"):
        select_survivors({" ": 0.5})


def test_progressive_search_resumes_only_survivors(tmp_path):
    import numpy as np

    from dlens.agents._progressive_search import run_progressive_search
    from dlens.schemas._downstream import DatasetRef
    from dlens.schemas._model_design import (
        ArchFamily,
        ArchitectureSpec,
        TrainingConfig,
    )
    from dlens.tools._torch_backends import (
        TorchInferBackend,
        TorchTrainBackend,
    )

    refs = {}
    rng = np.random.default_rng(0)

    for split in ("train", "val"):
        directory = tmp_path / split
        directory.mkdir()

        for label, class_name in enumerate(("a", "b")):
            for i in range(4):
                base = np.zeros((16, 16)) if label == 0 else np.ones((16, 16)) * 3
                image = base + rng.normal(0, 0.1, (16, 16))
                np.save(
                    directory / f"{class_name}_{i:04d}.npy",
                    image.astype(np.float32),
                )

        refs[split] = DatasetRef(
            root=str(directory),
            class_names=["a", "b"],
            image_shape=(16, 16),
            num_samples=8,
        )

    candidates = [
        ArchitectureSpec(
            name="tiny_cnn",
            family=ArchFamily.CNN,
            input_shape=(16, 16),
            channels=1,
            num_classes=2,
            depths=[1, 1],
            widths=[8, 16],
        ),
        ArchitectureSpec(
            name="tiny_resnet",
            family=ArchFamily.RESNET,
            input_shape=(16, 16),
            channels=1,
            num_classes=2,
            depths=[1, 1],
            widths=[8, 16],
        ),
    ]

    trainer = TorchTrainBackend(
        output_root=str(tmp_path / "models"),
        device="cpu",
        seed=0,
    )
    inferencer = TorchInferBackend(device="cpu")
    config = TrainingConfig(
        loss="cross_entropy",
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
    )

    result = run_progressive_search(
        candidates,
        refs["train"],
        refs["val"],
        trainer,
        inferencer,
        config,
        initial_epochs=1,
        additional_epochs=1,
        keep_fraction=0.5,
    )

    assert result["strategy"] == "successive_halving"
    assert result["winner"] in {"tiny_cnn", "tiny_resnet"}
    assert len(result["candidates"]) == 2

    survivors = [
        item for item in result["candidates"]
        if item["status"] == "active"
    ]
    eliminated = [
        item for item in result["candidates"]
        if item["status"] == "eliminated"
    ]

    assert len(survivors) == 1
    assert len(eliminated) == 1
    assert len(survivors[0]["stages"]) == 2
    assert len(eliminated[0]["stages"]) == 1
    assert survivors[0]["epochs_run"] == 2
    assert result["total_training_seconds"] > 0


def test_progressive_search_rejects_empty_candidates():
    from dlens.agents._progressive_search import run_progressive_search

    with pytest.raises(ValueError, match="At least one candidate"):
        run_progressive_search(
            candidates=[],
            train_dataset=None,
            val_dataset=None,
            trainer=None,
            inferencer=None,
            config=None,
        )


@pytest.mark.parametrize(
    ("initial_epochs", "additional_epochs"),
    [(0, 1), (1, 0), (-1, 2)],
)
def test_progressive_search_rejects_invalid_epoch_budgets(
    initial_epochs, additional_epochs
):
    from dlens.agents._progressive_search import run_progressive_search

    with pytest.raises(ValueError, match="Epoch budgets"):
        run_progressive_search(
            candidates=[object()],
            train_dataset=None,
            val_dataset=None,
            trainer=None,
            inferencer=None,
            config=None,
            initial_epochs=initial_epochs,
            additional_epochs=additional_epochs,
        )


def test_progressive_search_rejects_duplicate_candidate_names():
    from dlens.agents._progressive_search import run_progressive_search
    from dlens.schemas._model_design import (
        ArchFamily,
        ArchitectureSpec,
    )

    candidate = ArchitectureSpec(
        name="duplicate",
        family=ArchFamily.CNN,
        input_shape=(16, 16),
        channels=1,
        num_classes=2,
        depths=[1, 1],
        widths=[8, 16],
    )

    with pytest.raises(ValueError, match="Candidate names must be unique"):
        run_progressive_search(
            candidates=[candidate, candidate],
            train_dataset=None,
            val_dataset=None,
            trainer=None,
            inferencer=None,
            config=None,
        )


@pytest.mark.parametrize("keep_fraction", [0.0, -0.5, 1.1, math.nan])
def test_progressive_search_rejects_invalid_keep_fraction(keep_fraction):
    from dlens.agents._progressive_search import run_progressive_search

    with pytest.raises(ValueError, match="keep_fraction"):
        run_progressive_search(
            candidates=[object()],
            train_dataset=None,
            val_dataset=None,
            trainer=None,
            inferencer=None,
            config=None,
            keep_fraction=keep_fraction,
        )
