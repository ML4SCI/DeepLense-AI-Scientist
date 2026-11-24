"""Checks for paired super-resolution data loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from dlens.data.sr_dataset import SRPairDataset
from dlens.schemas._super_resolution import SRDatasetRef


class SRDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "LR").mkdir()
        (self.root / "HR").mkdir()
        self.ref = SRDatasetRef(
            root=str(self.root), lr_shape=(2, 2), hr_shape=(4, 4),
            scale_factor=2, channels=1, num_samples=1,
        )

    def write_pair(self, lr: np.ndarray, name: str = "sample1") -> None:
        hr = np.repeat(np.repeat(lr, 2, axis=-2), 2, axis=-1)
        np.save(self.root / "LR" / f"{name}.npy", lr)
        np.save(self.root / "HR" / f"{name}.npy", hr)

    def test_loads_matching_pair_without_clipping(self) -> None:
        lr = np.array([[[-0.1, 0.2], [0.8, 1.1]]], dtype=np.float64)
        self.write_pair(lr)
        ds = SRPairDataset(self.ref)
        low, high, sample_id = ds[0]
        self.assertEqual(sample_id, "sample1")
        self.assertEqual((len(ds), low.shape, high.shape), (1, (1, 2, 2), (1, 4, 4)))
        self.assertEqual(low.dtype, torch.float32)
        self.assertAlmostEqual(float(low.min()), -0.1)
        self.assertAlmostEqual(float(low.max()), 1.1)
        np.testing.assert_allclose(high.numpy()[0, ::2, ::2], low.numpy()[0])

    def test_batches_pairs_and_ids(self) -> None:
        self.write_pair(np.ones((1, 2, 2)), "sample1")
        self.write_pair(np.zeros((1, 2, 2)), "sample2")
        ref = self.ref.model_copy(update={"num_samples": 2})
        low, high, ids = next(iter(DataLoader(SRPairDataset(ref), batch_size=2)))
        self.assertEqual(low.shape, (2, 1, 2, 2))
        self.assertEqual(high.shape, (2, 1, 4, 4))
        self.assertEqual(ids, ("sample1", "sample2"))

    def test_rejects_missing_partner(self) -> None:
        np.save(self.root / "LR" / "sample1.npy", np.zeros((1, 2, 2)))
        with self.assertRaisesRegex(ValueError, "missing HR"):
            SRPairDataset(self.ref)

    def test_split_path_accepts_different_pair_sizes(self) -> None:
        self.write_pair(np.ones((2, 3)), "a")
        self.write_pair(np.zeros((1, 4, 5)), "b")
        dataset = SRPairDataset(self.root)
        self.assertEqual(
            [(tuple(lr.shape), tuple(hr.shape), sample_id) for lr, hr, sample_id in dataset],
            [((1, 2, 3), (1, 4, 6), "a"), ((1, 4, 5), (1, 8, 10), "b")],
        )

    def test_split_path_rejects_non_x2_pairs_when_read(self) -> None:
        self.write_pair(np.zeros((2, 3)))
        np.save(self.root / "HR" / "sample1.npy", np.zeros((4, 5)))
        dataset = SRPairDataset(self.root)
        with self.assertRaisesRegex(ValueError, "expected shape"):
            dataset[0]

    def test_rejects_bad_shape_and_scale(self) -> None:
        self.write_pair(np.zeros((1, 2, 2)))
        np.save(self.root / "HR" / "sample1.npy", np.zeros((1, 5, 5)))
        with self.assertRaisesRegex(ValueError, "expected shape"):
            SRPairDataset(self.ref)
        with self.assertRaises(ValidationError):
            SRDatasetRef(
                root=str(self.root), lr_shape=(2, 2), hr_shape=(5, 5), scale_factor=2
            )

    def test_rejects_nonfinite_pixels_when_read(self) -> None:
        lr = np.zeros((1, 2, 2))
        lr[0, 0, 0] = np.nan
        self.write_pair(lr)
        ds = SRPairDataset(self.ref)
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            ds[0]

    def test_random_transform_keeps_pair_aligned(self) -> None:
        self.write_pair(np.array([[[1, 2], [3, 4]]], dtype=np.float32))
        ds = SRPairDataset(self.ref, augment=True)
        for _ in range(20):
            low, high, _ = ds[0]
            down = high.reshape(1, 2, 2, 2, 2).mean(dim=(2, 4))
            torch.testing.assert_close(low, down)


if __name__ == "__main__":
    unittest.main()
