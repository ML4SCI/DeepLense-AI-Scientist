"""Dataset contract for paired image super resolution."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SRDatasetRef(BaseModel):
    """One split containing matching ``LR/<id>.npy`` and ``HR/<id>.npy`` files.

    Shapes exclude the channel dimension. Images on disk may be ``(H, W)`` for
    a single channel or ``(C, H, W)``; the loader always returns ``(C, H, W)``.
    """

    root: str = Field(description="Split directory containing LR/ and HR/.")
    lr_shape: tuple[int, int] = Field(description="Low resolution (height, width).")
    hr_shape: tuple[int, int] = Field(description="High resolution (height, width).")
    scale_factor: int = Field(ge=2, description="Integer enlargement in each axis.")
    channels: int = Field(default=1, ge=1)
    num_samples: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def check_shapes(self) -> SRDatasetRef:
        if any(dim < 1 for dim in (*self.lr_shape, *self.hr_shape)):
            raise ValueError("LR and HR dimensions must be positive")
        expected = tuple(dim * self.scale_factor for dim in self.lr_shape)
        if self.hr_shape != expected:
            raise ValueError(
                f"HR shape {self.hr_shape} must be {expected} for scale factor "
                f"{self.scale_factor}"
            )
        return self
