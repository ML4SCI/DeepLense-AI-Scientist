# schemas/_downstream.py
"""Domain schemas for the downstream stages: train -> infer -> analysis.

These are the typed contracts that link the three agents (each stage's output is
the next stage's input) and that populate the train/infer/analysis slots of an
``ExperimentRun`` (see ``schemas/_experiment.py``).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DatasetRef(BaseModel):
    """A reference to an on-disk classification dataset.

    Files are ``<class>_*.npy`` arrays under ``root`` (one label per class name),
    matching the layout produced by the Simulation Agent.
    """

    root: str = Field(description="Directory containing <class>_*.npy image arrays.")
    class_names: list[str] = Field(description="Class names, in label order.")
    image_shape: tuple[int, int] = Field(description="(H, W) of each image.")
    num_samples: Optional[int] = Field(default=None, description="Total samples, if known.")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class TrainResult(BaseModel):
    """Output of the Train stage: a trained model and its training metrics."""

    run_id: str = Field(description="Unique id for this training run.")
    weights_path: str = Field(description="Path to the saved model weights.")
    backend: str = Field(description="Training backend used (centroid | mock).")
    num_classes: int = Field(description="Number of classes the model was trained on.")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Training metrics (e.g. train_accuracy)."
    )
    epochs_run: int = Field(default=1, description="Number of epochs/passes run.")


class InferResult(BaseModel):
    """Output of the Inference stage: predictions on a dataset."""

    run_id: str = Field(description="Unique id for this inference run.")
    num_samples: int = Field(description="Number of samples scored.")
    num_classes: int = Field(description="Number of classes.")
    predictions: list[int] = Field(description="Predicted class index per sample.")
    probabilities: Optional[list[list[float]]] = Field(
        default=None, description="Per-class probabilities per sample, if available."
    )
    true_labels: Optional[list[int]] = Field(
        default=None, description="Ground-truth labels, if the data was labeled."
    )
    accuracy: Optional[float] = Field(default=None, description="Accuracy, if labels present.")
    backend: str = Field(default="unknown", description="Inference backend used.")


class AnalysisResult(BaseModel):
    """Output of the Analysis stage: evaluation summary for the planner to consume."""

    accuracy: float = Field(description="Overall accuracy.")
    macro_auc: Optional[float] = Field(default=None, description="Macro one-vs-rest ROC AUC.")
    confusion_matrix: list[list[int]] = Field(description="Rows = true, cols = predicted.")
    per_class: dict[str, dict[str, float]] = Field(
        description="class_name -> {precision, recall, f1, support}."
    )
    summary: str = Field(description="Short human-readable summary of performance.")
