"""Iterative residual prediction for image super-resolution.

A timestep-conditioned U-Net predicts the difference between the target
high-resolution image and the upscaled low-resolution input. At inference,
the model starts from random noise and iteratively refines the residual,
then adds the final prediction to the upscaled input.

Adapted from the iterative-auto-encoder-kaggle.ipynb notebook in the source
project's baselines/Atal_gupta_Iterative_auto_encoder directory.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=time.device) *
                         (-math.log(10000) / (half - 1)))
        angles = time.float()[:, None] * freq[None]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, up: bool = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(2 * in_ch if up else in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.transform = (nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) if up
                          else nn.Conv2d(out_ch, out_ch, 4, 2, 1))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(F.relu(self.conv1(x)))
        hidden = hidden + F.relu(self.time_mlp(time))[:, :, None, None]
        hidden = self.norm2(F.relu(self.conv2(hidden)))
        return self.transform(hidden)


class SimpleUnet(nn.Module):
    def __init__(self, base_width: int = 32):
        super().__init__()
        widths = [base_width * (2 ** i) for i in range(5)]
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(),
                                      nn.Linear(32, 32), nn.ReLU())
        self.conv0 = nn.Conv2d(2, widths[0], 3, padding=1)
        self.downs = nn.ModuleList(Block(widths[i], widths[i + 1], 32)
                                   for i in range(4))
        self.ups = nn.ModuleList(Block(widths[i], widths[i - 1], 32, up=True)
                                 for i in range(4, 0, -1))
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % 16, 0, (-height) % 16))
        time = self.time_mlp(timestep)
        x = self.conv0(x)
        skips = []
        for down in self.downs:
            x = down(x, time)
            skips.append(x)
        for up in self.ups:
            x = up(torch.cat((x, skips.pop()), dim=1), time)
        return self.output(x)[..., :height, :width]


def alpha_schedule(timesteps: int, device: torch.device) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


def diffuse(clean_residual: torch.Tensor, time: torch.Tensor,
            alpha_bar: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    alpha = alpha_bar[time].view(-1, 1, 1, 1)
    return alpha.sqrt() * clean_residual + (1.0 - alpha).sqrt() * noise


@torch.inference_mode()
def predict_hr(model: SimpleUnet, lr_up: torch.Tensor, *, timesteps: int = 1000,
               steps: int = 8, seed: int = 42) -> torch.Tensor:
    generator = torch.Generator(device=lr_up.device).manual_seed(seed)
    noisy = torch.randn(lr_up.shape, device=lr_up.device, generator=generator)
    alpha_bar = alpha_schedule(timesteps, lr_up.device)
    times = torch.linspace(timesteps - 1, 0, steps, device=lr_up.device).long()
    for index, time_value in enumerate(times):
        time = torch.full((len(lr_up),), int(time_value), device=lr_up.device,
                          dtype=torch.long)
        clean = model(torch.cat((lr_up, noisy), dim=1), time)
        if index + 1 == len(times):
            return lr_up + clean
        alpha = alpha_bar[time_value]
        next_alpha = alpha_bar[times[index + 1]]
        estimated_noise = (noisy - alpha.sqrt() * clean) / (1.0 - alpha).sqrt()
        noisy = next_alpha.sqrt() * clean + (1.0 - next_alpha).sqrt() * estimated_noise
    raise RuntimeError("inference requires at least one step")


def predict_pair(model: SimpleUnet, lr: torch.Tensor, hr: torch.Tensor, *,
                 timesteps: int, steps: int, seed: int) -> torch.Tensor:
    lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bilinear", align_corners=False)
    return predict_hr(model, lr_up, timesteps=timesteps, steps=steps, seed=seed)
