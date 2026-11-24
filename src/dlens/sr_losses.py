"""Named, reproducible objectives for super-resolution experiments."""

from __future__ import annotations

import torch
from torch.nn import functional as F

RECONSTRUCTION_LOSSES = (
    "l1", "mse", "charbonnier", "huber", "l1_ssim", "l1_edge",
    "l1_frequency",
)
LEGACY_LOSSES = ("l1_mse",)
DIFFUSION_LOSSES = ("noise_mse", "noise_l1", "noise_huber")
ADVERSARIAL_LOSSES = ("l1_adversarial",)
ALL_LOSSES = RECONSTRUCTION_LOSSES + LEGACY_LOSSES + DIFFUSION_LOSSES + ADVERSARIAL_LOSSES
COMPOSITE_WEIGHTS = {"l1_ssim": 0.1, "l1_edge": 0.05,
                     "l1_frequency": 0.05, "l1_adversarial": 0.01}


def default_loss(family: str) -> str:
    return "noise_mse" if family == "sr3" else "l1_adversarial" if family == "realesrgan" else "l1_mse"


def allowed_losses(family: str) -> tuple[str, ...]:
    if family == "sr3":
        return DIFFUSION_LOSSES
    if family == "realesrgan":
        return ADVERSARIAL_LOSSES
    return RECONSTRUCTION_LOSSES + LEGACY_LOSSES


def validate_loss(family: str, loss_id: str) -> None:
    if loss_id not in allowed_losses(family):
        raise ValueError(f"loss {loss_id!r} is incompatible with {family}; choose from {allowed_losses(family)}")


def _ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable local SSIM for images in [0, 1]."""
    window = min(11, prediction.shape[-2], prediction.shape[-1])
    if window % 2 == 0:
        window -= 1
    mu_x = F.avg_pool2d(prediction, window, stride=1)
    mu_y = F.avg_pool2d(target, window, stride=1)
    sigma_x = F.avg_pool2d(prediction * prediction, window, stride=1) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window, stride=1) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window, stride=1) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def _edge(image: torch.Tensor) -> torch.Tensor:
    kernels = image.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)))
    kernels = torch.stack((kernels, kernels.T)).unsqueeze(1)
    return F.conv2d(image, kernels.repeat(image.shape[1], 1, 1, 1),
                    padding=1, groups=image.shape[1])


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, loss_id: str,
                        l1_weight: float = 1.0) -> torch.Tensor:
    if loss_id == "l1_mse":
        return l1_weight * F.l1_loss(prediction, target) + (1 - l1_weight) * F.mse_loss(prediction, target)
    if loss_id == "l1":
        return F.l1_loss(prediction, target)
    if loss_id == "mse":
        return F.mse_loss(prediction, target)
    if loss_id == "charbonnier":
        return torch.sqrt((prediction - target).square() + 1e-6).mean()
    if loss_id == "huber":
        return F.smooth_l1_loss(prediction, target, beta=0.1)
    if loss_id == "l1_ssim":
        return F.l1_loss(prediction, target) + COMPOSITE_WEIGHTS[loss_id] * (1 - _ssim(prediction, target))
    if loss_id == "l1_edge":
        return F.l1_loss(prediction, target) + COMPOSITE_WEIGHTS[loss_id] * F.l1_loss(_edge(prediction), _edge(target))
    if loss_id == "l1_frequency":
        frequency_error = (torch.fft.rfft2(prediction, norm="ortho")
                           - torch.fft.rfft2(target, norm="ortho")).abs().mean()
        return F.l1_loss(prediction, target) + COMPOSITE_WEIGHTS[loss_id] * frequency_error
    if loss_id == "l1_adversarial":
        return F.l1_loss(prediction, target)
    raise ValueError(f"unknown reconstruction loss: {loss_id}")
