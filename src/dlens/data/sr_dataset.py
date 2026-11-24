"""Lazy, paired NumPy image loader for supervised super resolution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dlens.schemas._super_resolution import SRDatasetRef

PairTransform = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def _as_chw(
    array: np.ndarray, channels: int, shape: tuple[int, int] | None, path: Path
) -> np.ndarray:
    """Check one real-valued image and return a writable float32 CHW array."""
    if array.ndim == 2 and channels == 1:
        array = array[None, :, :]
    if shape is not None and array.shape != (channels, *shape):
        raise ValueError(f"{path}: expected shape {(channels, *shape)}, got {array.shape}")
    if array.ndim != 3 or array.shape[0] != channels:
        raise ValueError(f"{path}: expected {channels} channels in CHW order, got {array.shape}")
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"{path}: expected a real numeric array, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: image contains NaN or infinity")
    # Make a writable copy: np.load with mmap_mode='r' returns read-only memory.
    # Preserve the input values, including small LR excursions outside [0, 1].
    return np.array(array, dtype=np.float32, copy=True)


def random_pair_transform(
    lr: torch.Tensor, hr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply identical flips and a 90-degree rotation to an aligned pair."""
    if bool(torch.randint(0, 2, ()).item()):
        lr, hr = lr.flip(-1), hr.flip(-1)
    if bool(torch.randint(0, 2, ()).item()):
        lr, hr = lr.flip(-2), hr.flip(-2)
    turns = int(torch.randint(0, 4, ()).item())
    return lr.rot90(turns, (-2, -1)), hr.rot90(turns, (-2, -1))


class SRPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Return ``(LR, HR, sample_id)`` without holding the dataset in memory.

    Filenames must match exactly between LR/ and HR/. With an ``SRDatasetRef``,
    file headers are checked against the declared shapes at construction. With
    a split path, each pair is checked on read for single-channel 2x images.
    Pixel values are checked on read in both cases. Use augmentation for training.
    """

    def __init__(
        self,
        ref: SRDatasetRef | Path | str,
        *,
        augment: bool = False,
        pair_transform: PairTransform | None = None,
    ) -> None:
        self.ref = ref if isinstance(ref, SRDatasetRef) else None
        self.augment = augment
        self.pair_transform = pair_transform
        root = Path(self.ref.root if self.ref is not None else ref)
        lr_dir, hr_dir = root / "LR", root / "HR"
        if not lr_dir.is_dir() or not hr_dir.is_dir():
            raise FileNotFoundError(f"Expected LR/ and HR/ directories under {root}")

        lr_files = {p.name: p for p in lr_dir.glob("*.npy")}
        hr_files = {p.name: p for p in hr_dir.glob("*.npy")}
        missing_hr = sorted(lr_files.keys() - hr_files.keys())
        missing_lr = sorted(hr_files.keys() - lr_files.keys())
        if missing_hr or missing_lr:
            raise ValueError(
                f"Unpaired files under {root}: missing HR {missing_hr[:5]}, "
                f"missing LR {missing_lr[:5]}"
            )
        if not lr_files:
            raise ValueError(f"No .npy pairs found under {root}")
        self._pairs = [(lr_files[name], hr_files[name], Path(name).stem)
                       for name in sorted(lr_files)]
        if isinstance(ref, SRDatasetRef):
            if ref.num_samples is not None and len(self._pairs) != ref.num_samples:
                raise ValueError(f"{root}: expected {ref.num_samples} pairs, found {len(self._pairs)}")
            for lr_path, hr_path, _ in self._pairs:
                self._check_header(lr_path, ref.lr_shape, ref.channels)
                self._check_header(hr_path, ref.hr_shape, ref.channels)

    @staticmethod
    def _check_header(path: Path, shape: tuple[int, int], channels: int) -> None:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = (channels, *shape)
        if array.shape != expected and not (channels == 1 and array.shape == shape):
            raise ValueError(f"{path}: expected shape {expected}, got {array.shape}")
        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
            raise ValueError(f"{path}: expected a real numeric array, got {array.dtype}")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        lr_path, hr_path, sample_id = self._pairs[index]
        ref = self.ref
        channels = ref.channels if ref is not None else 1
        lr = torch.from_numpy(_as_chw(
            np.load(lr_path, mmap_mode="r", allow_pickle=False),
            channels, ref.lr_shape if ref is not None else None, lr_path,
        ))
        hr_shape = ref.hr_shape if ref is not None else (2 * lr.shape[-2], 2 * lr.shape[-1])
        hr = torch.from_numpy(_as_chw(
            np.load(hr_path, mmap_mode="r", allow_pickle=False),
            channels, hr_shape, hr_path,
        ))
        if self.augment:
            lr, hr = random_pair_transform(lr, hr)
        if self.pair_transform is not None:
            lr, hr = self.pair_transform(lr, hr)
        return lr, hr, sample_id
