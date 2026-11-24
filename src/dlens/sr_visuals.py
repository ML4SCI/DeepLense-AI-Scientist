"""Deterministic validation-image comparisons for completed SR checkpoints."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dlens.sr_metrics import bicubic
from dlens.sr_models import SRArchitecture, build_sr_model


def fixed_sample_ids(available_ids: list[str], count: int = 4) -> list[str]:
    """Spread examples across the fixed validation ordering without choosing by score."""
    if not available_ids:
        raise ValueError("cannot visualize an empty validation set")
    count = min(count, len(available_ids))
    if count == 1:
        return [available_ids[0]]
    return [available_ids[round(i * (len(available_ids) - 1) / (count - 1))]
            for i in range(count)]


def _scores(path: Path) -> dict[str, tuple[float, float]]:
    with path.open(newline="") as file:
        return {row["sample_id"]: (float(row["psnr_db"]), float(row["ssim"]))
                for row in csv.DictReader(file)}


def display_tile(image: torch.Tensor | np.ndarray, size: int, *,
                 sqrt_stretch: bool = False) -> Image.Image:
    array = image.detach().float().cpu().squeeze().numpy() if isinstance(image, torch.Tensor) else image
    scaled = np.clip(array, 0, 1)
    if sqrt_stretch:
        scaled = np.sqrt(scaled)
    pixels = (scaled * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="L").resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def display_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _short_run_id(run_id: str) -> str:
    screen = re.fullmatch(r"round_(\d+)_screen_(\d+)", run_id)
    if screen:
        return f"r{int(screen[1])} screen {int(screen[2])}"
    later = re.fullmatch(
        r"round_(\d+)_(promote|tune)_round_\d+_screen_(\d+)_(base|half|\d+)",
        run_id)
    if later:
        stage = "promo" if later[2] == "promote" else "tune"
        return f"r{int(later[1])} {stage} s{int(later[3])} {later[4]}"
    return run_id[-20:]


def render_comparison(
    dataset, sample_ids: list[str], runs: list[dict], bicubic_csv: Path,
    output: Path, device: torch.device, title: str,
) -> None:
    """Save LR, bicubic, every supplied model, and HR in one labeled PNG."""
    if not runs:
        raise ValueError("a comparison requires at least one trained model")
    positions = {sample_id: index for index, (_, _, sample_id)
                 in enumerate(dataset._pairs)}
    samples = [dataset[positions[sample_id]] for sample_id in sample_ids]
    model_scores = [_scores(Path(run["per_image_csv"])) for run in runs]
    bicubic_scores = _scores(bicubic_csv)
    predictions = []
    for run in runs:
        # Load one checkpoint at a time so five-model comparisons fit Kaggle RAM.
        spec = SRArchitecture(**run["architecture"])
        model = build_sr_model(spec).to(device)
        saved = torch.load(run["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        model.eval()
        with torch.inference_mode():
            predictions.append([model(lr[None].to(device))[0].cpu()
                                for lr, _, _ in samples])
        del model, saved

    labels = ["LR input", "Bicubic"] + [
        (f"{run['architecture']['family'].upper()} "
         f"w{run['architecture']['width']}/d{run['architecture']['depth']}\n"
         f"{_short_run_id(run['id'])}")
        for run in runs
    ] + ["HR reference"]
    rows = []
    for index, (lr, hr, sample_id) in enumerate(samples):
        panels = [lr, bicubic(lr[None], hr[None])[0], *(images[index] for images in predictions), hr]
        scores = [bicubic_scores[sample_id], *(model[sample_id] for model in model_scores)]
        captions = [sample_id[:23], *(f"{psnr:.3f} dB  ·  {ssim:.5f}" for psnr, ssim in scores),
                    "Ground truth"]
        rows.append((panels, captions))
    render_grid(output, title, "Same fixed validation images. Display values clipped to [0, 1].",
                labels, rows)


def render_grid(
    output: Path, title: str, subtitle: str, labels: list[str],
    rows: list[tuple[list[torch.Tensor], list[str]]], *, tile: int = 150,
    column_width: int = 184, row_height: int = 210, sqrt_stretch: bool = False,
) -> None:
    top = 104
    canvas = Image.new("RGB", (len(labels) * column_width, top + len(rows) * row_height),
                       "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = display_font(18), display_font(12)
    draw.text((12, 10), title, fill="black", font=title_font)
    draw.text((12, 37), subtitle, fill="#444444", font=label_font)
    for column, label in enumerate(labels):
        draw.multiline_text((column * column_width + 7, 61), label,
                            fill="black", font=label_font, spacing=1)
    for index, (panels, captions) in enumerate(rows):
        y = top + index * row_height
        for column, (panel, caption) in enumerate(zip(panels, captions, strict=True)):
            x = column * column_width + 7
            canvas.paste(display_tile(panel, tile, sqrt_stretch=sqrt_stretch), (x, y))
            draw.text((x, y + tile + 5), caption, fill="black", font=label_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    canvas.save(temporary, format=Image.registered_extensions().get(output.suffix.lower(), "PNG"))
    temporary.replace(output)
