"""External baseline must describe the same validation images and metrics."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dlens.sr_baseline import METRIC_POLICY, load_baseline, load_iterative_baseline


class IterativeBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.rows = [
            {"sample_id": "a", "bicubic_psnr_db": 30.0, "bicubic_ssim": 0.8,
             "model_psnr_db": 31.0, "model_ssim": 0.81},
            {"sample_id": "b", "bicubic_psnr_db": 32.0, "bicubic_ssim": 0.82,
             "model_psnr_db": 33.0, "model_ssim": 0.83},
        ]
        with (self.folder / "validation_per_image.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.summary = {"model": "Iterative Auto Encoder", "metric_policy": dict(METRIC_POLICY),
                        "normalization": "none", "validation_pairs": 2,
                        "model_validation": {"psnr_db": 32.0, "ssim": 0.82}}
        self._write_summary()

    def _write_summary(self) -> None:
        (self.folder / "baseline_summary.json").write_text(json.dumps(self.summary))

    def test_uses_imported_model_means_after_matching_current_bicubic(self) -> None:
        current = [{"sample_id": row["sample_id"], "psnr_db": row["bicubic_psnr_db"],
                    "ssim": row["bicubic_ssim"]} for row in self.rows]
        means, source = load_iterative_baseline(self.folder, current)
        self.assertEqual(means["psnr_db"], 32.0)
        self.assertAlmostEqual(means["ssim"], 0.82)
        self.assertEqual(source["validation_pairs_used"], 2)
        subset, _ = load_iterative_baseline(self.folder, current[:1])
        self.assertEqual(subset["psnr_db"], 31.0)
        self.assertAlmostEqual(subset["ssim"], 0.81)

    def test_rejects_different_images_even_when_ids_match(self) -> None:
        current = [{"sample_id": "a", "psnr_db": 29.0, "ssim": 0.8}]
        with self.assertRaisesRegex(ValueError, "bicubic psnr_db mismatch"):
            load_iterative_baseline(self.folder, current)

    def test_rejects_different_ids_or_metric_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation IDs"):
            load_iterative_baseline(self.folder, [{"sample_id": "other"}])
        self.summary["metric_policy"]["ssim_window"] = 7
        self._write_summary()
        with self.assertRaisesRegex(ValueError, "metric settings"):
            load_iterative_baseline(self.folder, [{"sample_id": "a", "psnr_db": 30.0,
                                                  "ssim": 0.8}])

    def test_rejects_other_baseline_models(self) -> None:
        self.summary["model"] = "Unsupported Baseline"
        self._write_summary()
        current = [{"sample_id": row["sample_id"], "psnr_db": row["bicubic_psnr_db"],
                    "ssim": row["bicubic_ssim"]} for row in self.rows]
        with self.assertRaisesRegex(ValueError, "unexpected baseline model"):
            load_baseline(self.folder, current)


if __name__ == "__main__":
    unittest.main()
