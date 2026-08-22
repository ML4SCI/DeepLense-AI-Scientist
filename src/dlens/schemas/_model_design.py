# schemas/_model_design.py
"""Domain schemas for the model-design stage (agent #2 of WORKFLOW_DESIGN.md).

These map onto the ``architecture`` / ``training_config`` slots of the
``ExperimentRun``. ``DatasetCharacteristics`` is the model-design input, derivable
from one or more ``SimOutput`` batches produced by the Simulation Agent.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from dlens.schemas._simulation import SimOutput


class TaskType(str, Enum):
    """Downstream learning task."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ArchFamily(str, Enum):
    """Coarse architecture family."""

    CNN = "cnn"
    RESNET = "resnet"
    VIT = "vit"
    EQUIVARIANT = "equivariant"
    MLPMIXER = "mlpmixer"
    HYBRID = "hybrid"


class DatasetCharacteristics(BaseModel):
    """What the model-design agent needs to know about the data."""

    task: TaskType = Field(default=TaskType.CLASSIFICATION, description="Learning task.")
    image_shape: tuple[int, int] = Field(description="(H, W) of each image.")
    channels: int = Field(default=1, ge=1, description="Number of image channels.")
    num_classes: Optional[int] = Field(
        default=None, description="Number of classes (classification)."
    )
    num_samples: Optional[int] = Field(
        default=None, description="Total number of samples available."
    )
    notes: str = Field(default="", description="Free-form notes (e.g. class names, SNR).")


# Families the torch backend can actually construct and train. Broadened
# 2026-08-13: the search could previously only propose convnets, which makes an
# "unbiased search" claim over that space uninformative about whether convnets
# are the right answer. Every family listed here is covered by an offline build
# AND train test (tests/test_arch_families.py).
BUILDABLE_FAMILIES = {
    ArchFamily.RESNET,
    ArchFamily.CNN,
    ArchFamily.VIT,
    ArchFamily.EQUIVARIANT,
    ArchFamily.MLPMIXER,
    ArchFamily.HYBRID,
}


class ArchitectureSpec(BaseModel):
    """A proposed neural-network architecture (the ``architecture`` slot).

    ``depths``/``widths`` define the structure per stage; when omitted, the backend
    falls back to the named preset (e.g. ``resnet18``/``resnet34``). Only
    ``BUILDABLE_FAMILIES`` can be instantiated — candidate generators must stay
    inside that space.
    """

    name: str = Field(description="Concrete architecture label, e.g. 'resnet18' or 'cnn_s3w64'.")
    family: ArchFamily = Field(description="Architecture family.")
    input_shape: tuple[int, int] = Field(description="(H, W) the model expects.")
    channels: int = Field(default=1, description="Input channels.")
    num_classes: Optional[int] = Field(default=None, description="Output classes, if classification.")
    depths: Optional[list[int]] = Field(
        default=None,
        description=(
            "Per-stage structure; 2-4 stages, each 1-6. Meaning depends on family: "
            "resnet = residual blocks per stage; cnn = 3x3 convs per stage; "
            "equivariant = group-conv blocks per stage; vit/mlpmixer = the SUM is "
            "the number of transformer/mixer blocks; hybrid = conv stage depths "
            "with the LAST entry the number of attention blocks."
        ),
    )
    widths: Optional[list[int]] = Field(
        default=None,
        description=(
            "Per-stage size (8-1024), same length as depths. resnet/cnn/equivariant "
            "= channels per stage; vit/mlpmixer = the LAST entry is the embedding "
            "dimension; hybrid = conv channels with the LAST entry the attention "
            "dimension."
        ),
    )
    physics_informed: bool = Field(
        default=False, description="Whether physics priors (e.g. equivariance) are used."
    )
    rationale: str = Field(default="", description="Why this architecture fits the data/task.")

    @model_validator(mode="after")
    def _check_structure(self) -> "ArchitectureSpec":
        if (self.depths is None) != (self.widths is None):
            raise ValueError("depths and widths must be given together")
        if self.depths is not None:
            if not (2 <= len(self.depths) <= 4) or len(self.depths) != len(self.widths):
                raise ValueError("need 2-4 stages with matching depths/widths lengths")
            if any(not (1 <= d <= 6) for d in self.depths):
                raise ValueError("each stage depth must be 1-6")
            if any(not (8 <= w <= 1024) for w in self.widths):
                raise ValueError("each stage width must be 8-1024")
        return self

    def is_buildable(self) -> bool:
        """True if the torch backend can construct this spec."""
        return self.family in BUILDABLE_FAMILIES


class TrainingConfig(BaseModel):
    """Training hyperparameters (the ``training_config`` slot)."""

    loss: str = Field(description="Loss function, e.g. 'cross_entropy' or 'mse'.")
    optimizer: str = Field(default="adamw", description="Optimizer.")
    learning_rate: float = Field(default=3e-4, gt=0, description="Learning rate.")
    batch_size: int = Field(default=64, ge=1, description="Batch size.")
    epochs: int = Field(default=30, ge=1, description="Number of epochs.")
    weight_decay: float = Field(default=1e-4, ge=0, description="Weight decay.")
    lr_scheduler: str = Field(default="cosine", description="LR schedule.")
    # Regularization knobs (available to the experiment planner):
    dropout: float = Field(default=0.0, ge=0, lt=1, description="Dropout before the classifier head.")
    augment: bool = Field(default=False, description="Random flip/90-degree-rotation augmentation.")
    early_stop_patience: int = Field(
        default=0, ge=0,
        description="If >0, hold out 10% of train and stop after this many epochs without "
        "improvement (best weights restored).",
    )


def characteristics_from_sim_outputs(
    outputs: list[SimOutput],
    *,
    task: TaskType = TaskType.CLASSIFICATION,
) -> DatasetCharacteristics:
    """Bridge agent #1 -> agent #2: derive dataset characteristics from sim batches.

    Each ``SimOutput`` is one substructure class; the distinct substructure types
    across batches give the class count, and image shape/sample totals aggregate.
    """
    if not outputs:
        raise ValueError("Need at least one SimOutput to derive characteristics.")
    classes = sorted({o.config.substructure_type.value for o in outputs})
    return DatasetCharacteristics(
        task=task,
        image_shape=outputs[0].image_shape,
        channels=1,
        num_classes=len(classes) if task == TaskType.CLASSIFICATION else None,
        num_samples=sum(o.num_generated for o in outputs),
        notes=f"classes={classes}; backends={sorted({o.backend for o in outputs})}",
    )
