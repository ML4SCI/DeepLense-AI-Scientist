"""Numerical checks for super-resolution image metrics."""

from __future__ import annotations

import math
import unittest

import torch

from dlens.sr_metrics import evaluate_batches, psnr_per_image, ssim_per_image, summarize_scores


class SRMetricsTests(unittest.TestCase):
    def test_psnr_known_errors(self) -> None:
        target = torch.zeros((2, 1, 16, 16))
        prediction = torch.stack((torch.ones((1, 16, 16)),
                                  torch.full((1, 16, 16), 0.5)))
        scores = psnr_per_image(prediction, target)
        self.assertAlmostEqual(float(scores[0]), 0.0, places=6)
        self.assertAlmostEqual(float(scores[1]), 20 * math.log10(2), places=6)

    def test_identical_images(self) -> None:
        image = torch.rand((2, 1, 16, 16), generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.isinf(psnr_per_image(image, image)).all())
        torch.testing.assert_close(ssim_per_image(image, image), torch.ones(2), atol=1e-5, rtol=0)

    def test_exact_matches_have_finite_validation_scores(self) -> None:
        lr = torch.zeros(2, 1, 8, 8)
        hr = torch.zeros(2, 1, 16, 16)
        batches = evaluate_batches([(lr, hr, ["first", "second"])], torch.device("cpu"),
                                   max_pairs=1, include_mse=True)
        [(_, _, _, rows)] = batches
        self.assertEqual(rows, [{"sample_id": "first", "psnr_db": 150.0, "ssim": 1.0, "mse": 0.0}])
        self.assertEqual(summarize_scores(rows)["psnr_db"], {"mean": 150.0, "std": 0.0})

    def test_ssim_decreases_when_structure_changes(self) -> None:
        image = torch.zeros((1, 1, 16, 16))
        image[:, :, 4:12, 4:12] = 1
        shifted = torch.roll(image, shifts=3, dims=-1)
        score = ssim_per_image(shifted, image)
        self.assertLess(float(score[0]), 1.0)

    def test_rejects_mismatch_and_small_images(self) -> None:
        image = torch.zeros((1, 1, 16, 16))
        with self.assertRaisesRegex(ValueError, "same NCHW shape"):
            psnr_per_image(image, image[:, :, :8, :8])
        with self.assertRaisesRegex(ValueError, "at least as large"):
            ssim_per_image(image[:, :, :8, :8], image[:, :, :8, :8])


if __name__ == "__main__":
    unittest.main()
