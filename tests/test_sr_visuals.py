"""Validation comparisons use fixed examples and completed checkpoints."""

import csv
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from dlens.sr_models import SRArchitecture, build_sr_model
from dlens.sr_visuals import fixed_sample_ids, render_comparison
from scripts.report_sr_scientist import render


class _TinyDataset:
    def __init__(self):
        self._pairs = [(None, None, "sample_1"), (None, None, "sample_2")]

    def __getitem__(self, index):
        lr = torch.full((1, 15, 15), 0.4 + index * 0.1)
        hr = torch.full((1, 30, 30), 0.4 + index * 0.1)
        return lr, hr, self._pairs[index][2]


def _write_scores(path: Path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("sample_id", "psnr_db", "ssim"))
        writer.writeheader()
        writer.writerow({"sample_id": "sample_1", "psnr_db": 30, "ssim": 0.8})
        writer.writerow({"sample_id": "sample_2", "psnr_db": 31, "ssim": 0.9})


class VisualTests(unittest.TestCase):
    def test_fixed_ids_and_rendered_comparison(self):
        self.assertEqual(fixed_sample_ids([str(i) for i in range(7)]),
                         ["0", "2", "4", "6"])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec = SRArchitecture("cnn", width=8, depth=1, image_size=15)
            checkpoint = root / "model.pt"
            torch.save({"model": build_sr_model(spec).state_dict()}, checkpoint)
            scores = root / "scores.csv"
            _write_scores(scores)
            run = {"id": "round_01_screen_01", "architecture": asdict(spec),
                   "checkpoint": str(checkpoint), "per_image_csv": str(scores)}
            output = root / "visualizations" / "screening.png"
            render_comparison(_TinyDataset(), ["sample_1", "sample_2"],
                              [run], scores, output, torch.device("cpu"), "Screening")
            with Image.open(output) as image:
                self.assertEqual(image.size, (4 * 184, 104 + 2 * 210))
            self.assertFalse((output.parent / "screening.png.tmp").exists())

    def test_report_links_saved_plots(self):
        # Render needs setup fields but no completed runs for this report excerpt.
        record = {"config": {"train_split": "train", "search_epochs": 5,
                             "use_llm": False},
                  "baseline_pairs": 2, "test_split_used": False,
                  "selection_rule": "both metrics", "baseline": {"psnr_db": 30, "ssim": .8},
                  "runs": [], "visualizations": {
                      "sample_ids": ["sample_1"],
                      "rounds": {"1": {"screening": "visualizations/round_01_screening.png"}},
                      "final": "visualizations/round_winners.png"}}
        report = render(record)
        self.assertIn("![Round 1 screening](visualizations/round_01_screening.png)", report)
        self.assertIn("![Final round winners](visualizations/round_winners.png)", report)


if __name__ == "__main__":
    unittest.main()
