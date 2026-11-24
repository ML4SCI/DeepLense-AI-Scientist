"""Per-image metrics for single- or multi-channel super resolution outputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.nn import functional as F

METRIC_POLICY = {"data_range": 1.0, "clamp": False, "ssim_window": 11,
                 "ssim_sigma": 1.5, "psnr_cap_db": 150.0}
METRICS = ("psnr_db", "ssim")


def matches_metric_policy(policy: object) -> bool:
    legacy = {key: value for key, value in METRIC_POLICY.items() if key != "psnr_cap_db"}
    return policy == METRIC_POLICY or policy == legacy


def _check_images(prediction: torch.Tensor, target: torch.Tensor, data_range: float) -> None:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have the same NCHW shape")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must contain only finite values")


def psnr_per_image(
    prediction: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0
) -> torch.Tensor:
    """Return PSNR in dB for every NCHW image; perfect matches return +inf."""
    _check_images(prediction, target, data_range)
    mse = _mse_per_image(prediction, target)
    return 10.0 * torch.log10((data_range * data_range) / mse)


def _mse_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.double() - target.double()).square().mean(dim=(1, 2, 3))


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return Gaussian-window SSIM for every NCHW image.

    Uses the common population-covariance form with K1=0.01 and K2=0.03.
    The 11x11 window is applied without padding, then averaged over valid
    spatial positions and channels.
    """
    _check_images(prediction, target, data_range)
    if window_size < 3 or window_size % 2 == 0 or sigma <= 0:
        raise ValueError("window_size must be odd and >=3, and sigma must be positive")
    if min(prediction.shape[-2:]) < window_size:
        raise ValueError("images must be at least as large as the SSIM window")

    # Smooth five independent fields in one grouped convolution. Using two
    # one-dimensional passes is equivalent to a separable 2-D Gaussian window.
    channels = prediction.shape[1]
    axis = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype)
    axis = axis - window_size // 2
    gaussian = torch.exp(-(axis.square()) / (2 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    fields = torch.cat(
        (prediction, target, prediction.square(), target.square(), prediction * target),
        dim=1,
    )
    groups = 5 * channels
    horizontal = gaussian.reshape(1, 1, 1, window_size).repeat(groups, 1, 1, 1)
    vertical = gaussian.reshape(1, 1, window_size, 1).repeat(groups, 1, 1, 1)
    smooth = F.conv2d(fields, horizontal, groups=groups)
    smooth = F.conv2d(smooth, vertical, groups=groups)
    mu_x, mu_y, ex2, ey2, exy = smooth.split(channels, dim=1)
    var_x = ex2 - mu_x.square()
    var_y = ey2 - mu_y.square()
    cov_xy = exy - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    return (numerator / denominator).mean(dim=(1, 2, 3))


def bicubic(lr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    return F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)


def score_images(prediction: torch.Tensor, target: torch.Tensor, ids: Sequence[str],
                 *, include_mse: bool = False) -> list[dict]:
    """Score unclipped pixels and cap reported PSNR for finite aggregation."""
    psnr = psnr_per_image(prediction, target, data_range=METRIC_POLICY["data_range"])
    values = {
        "psnr_db": psnr.clamp(max=METRIC_POLICY["psnr_cap_db"]),
        "ssim": ssim_per_image(prediction, target, data_range=METRIC_POLICY["data_range"],
                               window_size=METRIC_POLICY["ssim_window"],
                               sigma=METRIC_POLICY["ssim_sigma"]),
    }
    if include_mse:
        values["mse"] = _mse_per_image(prediction, target)
    columns = {key: scores.tolist() for key, scores in values.items()}
    return [{"sample_id": sample_id, **{key: scores[i] for key, scores in columns.items()}}
            for i, sample_id in enumerate(ids)]


def evaluate_batches(
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, Sequence[str]]],
    device: torch.device,
    predict: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None,
    *, max_pairs: int | None = None, include_mse: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]]:
    """Yield the inputs, predictions, and scores for each validation batch."""
    offset = 0
    for lr, hr, ids in loader:
        if max_pairs is not None:
            take = max_pairs - offset
            lr, hr, ids = lr[:take], hr[:take], ids[:take]
        with torch.inference_mode():
            lr, hr = lr.to(device), hr.to(device)
            prediction = bicubic(lr, hr) if predict is None else predict(lr, hr, offset)
            rows = score_images(prediction, hr, ids, include_mse=include_mse)
        yield lr, hr, prediction, rows
        offset += len(rows)
        if max_pairs is not None and offset >= max_pairs:
            break


def summarize_scores(rows: list[dict], metrics: Sequence[str] = METRICS,
                     *, extended: bool = False) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("validation requires at least one image score")
    stats = {}
    for key in metrics:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"validation requires finite {key} scores")
        stats[key] = {"mean": float(values.mean()), "std": float(values.std())}
        if extended:
            stats[key].update(median=float(np.median(values)), min=float(values.min()),
                              max=float(values.max()))
    return stats
