"""Small conditional diffusion SR model with reproducible DDIM validation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DenoiseBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.time = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.conv1(x) + self.time(embedding)[:, :, None, None])
        return x + self.conv2(hidden)


class SR3Model(nn.Module):
    """Conditional noise predictor; forward performs deterministic 8-step DDIM."""

    def __init__(self, width: int, depth: int, noise_steps: int = 32):
        super().__init__()
        self.noise_steps = noise_steps
        self.stem = nn.Conv2d(2, width, 3, padding=1)
        self.down = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.low_blocks = nn.ModuleList(DenoiseBlock(width) for _ in range(depth))
        self.up = nn.Conv2d(width, width, 3, padding=1)
        self.high_blocks = nn.ModuleList(DenoiseBlock(width) for _ in range(depth))
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                      nn.Linear(width, width))
        steps = torch.arange(noise_steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos((steps / noise_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = (alpha_bar / alpha_bar[0]).clamp(min=1e-5).float()
        self.register_buffer("alpha_bar", alpha_bar)

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        width = self.stem.out_channels
        half = width // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device)
                                / max(half - 1, 1))
        phases = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((phases.sin(), phases.cos()), 1)
        if embedding.shape[1] < width:
            embedding = F.pad(embedding, (0, width - embedding.shape[1]))
        return self.time_mlp(embedding)

    def predict_noise(self, noisy: torch.Tensor, condition: torch.Tensor,
                      timesteps: torch.Tensor) -> torch.Tensor:
        embedding = self._time_embedding(timesteps)
        high = self.stem(torch.cat((noisy, condition), 1))
        low = self.down(high)
        for block in self.low_blocks:
            low = block(low, embedding)
        high = high + self.up(F.interpolate(low, size=high.shape[-2:], mode="nearest"))
        for block in self.high_blocks:
            high = block(high, embedding)
        return self.head(F.silu(high))

    def training_loss(self, lr: torch.Tensor, hr: torch.Tensor,
                      loss_id: str = "noise_mse") -> torch.Tensor:
        condition = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic",
                                  align_corners=False)
        timesteps = torch.randint(1, self.noise_steps + 1, (hr.shape[0],), device=hr.device)
        alpha = self.alpha_bar[timesteps][:, None, None, None]
        noise = torch.randn_like(hr)
        noisy = alpha.sqrt() * hr + (1 - alpha).sqrt() * noise
        prediction = self.predict_noise(noisy, condition, timesteps)
        if loss_id == "noise_mse":
            return F.mse_loss(prediction, noise)
        if loss_id == "noise_l1":
            return F.l1_loss(prediction, noise)
        if loss_id == "noise_huber":
            return F.smooth_l1_loss(prediction, noise, beta=0.1)
        raise ValueError(f"unknown diffusion loss: {loss_id}")

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        condition = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        generator = torch.Generator(device=lr.device).manual_seed(42)
        # Use the same initial noise field for every image so a sample's output
        # is independent of validation batch size and position.
        image = torch.randn((1, *condition.shape[1:]), device=lr.device, dtype=lr.dtype,
                            generator=generator).expand_as(condition)
        schedule = torch.linspace(self.noise_steps, 1, 8, device=lr.device).long()
        for index, step in enumerate(schedule):
            t = torch.full((lr.shape[0],), int(step), device=lr.device, dtype=torch.long)
            alpha = self.alpha_bar[step]
            predicted_noise = self.predict_noise(image, condition, t)
            estimated_clean = (image - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            previous = schedule[index + 1] if index + 1 < len(schedule) else 0
            previous_alpha = self.alpha_bar[previous]
            image = previous_alpha.sqrt() * estimated_clean + (1 - previous_alpha).sqrt() * predicted_noise
        return image
