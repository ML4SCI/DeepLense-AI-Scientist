# tools/_model_design.py
"""The ``recommend_architecture`` tool for the Model Design Agent.

A deterministic baseline recommender — the Model Design Agent is a thin wrapper
around this tool (the framework convention: agents are wrappers around tools). The
recommender maps dataset characteristics to a sensible ResNet/CNN baseline plus
training hyperparameters; the agent surfaces and lightly justifies the result.
"""

from __future__ import annotations

from pydantic_ai import Agent

from dlens.schemas._model_design import (
    ArchFamily,
    ArchitectureSpec,
    DatasetCharacteristics,
    TaskType,
    TrainingConfig,
)


def recommend_baseline(characteristics: DatasetCharacteristics) -> dict:
    """Deterministically recommend a baseline architecture + training config.

    Small images (<=64 px) -> resnet18; larger -> resnet34. Loss follows the task.
    Returns a dict with ``architecture`` and ``training_config`` sub-objects.
    """
    h, w = characteristics.image_shape
    small = max(h, w) <= 64
    name = "resnet18" if small else "resnet34"

    architecture = ArchitectureSpec(
        name=name,
        family=ArchFamily.RESNET,
        input_shape=characteristics.image_shape,
        channels=characteristics.channels,
        num_classes=characteristics.num_classes,
        physics_informed=False,
        rationale=(
            f"{name} is a strong, well-understood baseline for "
            f"{characteristics.image_shape} {characteristics.channels}-channel images. "
            "For strong-lensing substructure, an equivariant backbone (e.g. E(2)-CNN) "
            "is a natural physics-informed upgrade in a later iteration."
        ),
    )
    is_clf = characteristics.task == TaskType.CLASSIFICATION
    training_config = TrainingConfig(
        loss="cross_entropy" if is_clf else "mse",
        optimizer="adamw",
        learning_rate=3e-4,
        batch_size=64 if small else 32,
        epochs=30,
        weight_decay=1e-4,
        lr_scheduler="cosine",
    )
    return {
        "architecture": architecture.model_dump(mode="json"),
        "training_config": training_config.model_dump(mode="json"),
    }


def register_model_design_tool(agent: Agent) -> None:
    """Register the deterministic ``recommend_architecture`` tool onto ``agent``."""

    @agent.tool_plain
    def recommend_architecture(characteristics: DatasetCharacteristics) -> dict:
        """Recommend a baseline architecture and training configuration.

        Call this with the dataset characteristics (task, image shape, channels,
        number of classes). Returns ``architecture`` and ``training_config`` objects
        you should report back, optionally with brief refinements/justification.
        """
        return recommend_baseline(characteristics)
