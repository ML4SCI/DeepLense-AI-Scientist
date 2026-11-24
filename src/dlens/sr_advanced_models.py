"""Compact, trainable x2 variants of established SR architecture designs.

The width and depth knobs deliberately make these smaller than published models.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(width, max(4, width // 4), 1), nn.ReLU(),
                                 nn.Conv2d(max(4, width // 4), width, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class EDSRBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.net(x)


class RCAB(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, padding=1))
        self.attention = ChannelAttention(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attention(self.conv(x))


class DistillationBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        small = max(4, width // 4)
        self.distill = nn.Conv2d(width, small, 1)
        self.conv = nn.Conv2d(width, width, 3, padding=1)
        self.fuse = nn.Conv2d(width + small, width, 1)
        self.attention = ChannelAttention(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distilled = F.relu(self.distill(x))
        remaining = F.relu(self.conv(x))
        return x + self.attention(self.fuse(torch.cat((distilled, remaining), 1)))


class SeparableBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 1), nn.ReLU(),
                                 nn.Conv2d(width, width, 3, padding=1, groups=width),
                                 nn.Conv2d(width, width, 1), ChannelAttention(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DenseBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        growth = max(4, width // 2)
        self.layers = nn.ModuleList(nn.Conv2d(width + i * growth, growth, 3, padding=1)
                                    for i in range(3))
        self.fuse = nn.Conv2d(width + 3 * growth, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(F.leaky_relu(layer(torch.cat(features, 1)), 0.2))
        return x + 0.2 * self.fuse(torch.cat(features, 1))


class RRDB(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.blocks = nn.Sequential(*(DenseBlock(width) for _ in range(3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.blocks(x)


class ChannelTransformerBlock(nn.Module):
    """Channel covariance attention for alternating DAT-style blocks."""

    def __init__(self, width: int):
        super().__init__()
        self.qkv = nn.Conv2d(width, width * 3, 1)
        self.project = nn.Conv2d(width, width, 1)
        self.feedforward = nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.GELU(),
                                         nn.Conv2d(width * 2, width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, 3, channels, height * width).unbind(1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        weights = torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1)
        attended = torch.matmul(weights, v).reshape(batch, channels, height, width)
        x = x + self.project(attended)
        return x + self.feedforward(x)


class WindowAttentionBlock(nn.Module):
    def __init__(self, width: int, kind: str, shifted: bool):
        super().__init__()
        self.kind, self.shifted, self.window = kind, shifted, 5
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 4 if width % 4 == 0 else 1,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Linear(width * 2, width))
        self.channel = ChannelAttention(width) if kind == "hat" else None
        self.cross = nn.Conv2d(width, width, 3, padding=1) if kind == "hat" else None
        self.channel_transform = ChannelTransformerBlock(width) if kind == "dat" and shifted else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_transform is not None:
            return self.channel_transform(x)
        batch, channels, height, width = x.shape
        residual = x
        w = self.window
        shift = w // 2 if self.shifted else 0
        if shift:
            x = torch.roll(x, (-shift, -shift), (-2, -1))
        x = F.pad(x, (0, (-width) % w, 0, (-height) % w))
        hp, wp = x.shape[-2:]
        tokens = x.reshape(batch, channels, hp // w, w, wp // w, w)
        tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(-1, w * w, channels)
        normalized = self.norm1(tokens)
        attended = self.attn(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        y = tokens.reshape(batch, hp // w, wp // w, w, w, channels)
        y = y.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, hp, wp)
        y = y[..., :height, :width]
        if shift:
            y = torch.roll(y, (shift, shift), (-2, -1))
        if self.channel is not None:
            y = y + self.channel(residual)
        if self.cross is not None:
            y = y + self.cross(y)
        return residual + y


class AdvancedSR(nn.Module):
    """Low-resolution feature trunk and learned sub-pixel x2 reconstruction."""

    def __init__(self, family: str, width: int, depth: int):
        super().__init__()
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        if family == "edsr":
            blocks = [EDSRBlock(width) for _ in range(depth)]
        elif family == "rcan":
            blocks = [RCAB(width) for _ in range(depth)]
        elif family == "imdn":
            blocks = [DistillationBlock(width) for _ in range(depth)]
        elif family == "bsrn":
            blocks = [SeparableBlock(width) for _ in range(depth)]
        elif family == "realesrgan":
            blocks = [RRDB(width) for _ in range(depth)]
        else:
            blocks = [WindowAttentionBlock(width, family, i % 2 == 1)
                      for i in range(depth)]
        self.blocks = nn.Sequential(*blocks)
        self.fuse = nn.Conv2d(width, width, 3, padding=1)
        self.upsample = nn.Sequential(nn.Conv2d(width, 4, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        features = self.stem(lr)
        return base + self.upsample(features + self.fuse(self.blocks(features)))


class PatchDiscriminator(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(width, width, 4, stride=2, padding=1),
                                 nn.LeakyReLU(0.2), nn.Conv2d(width, width * 2, 3, padding=1),
                                 nn.LeakyReLU(0.2), nn.Conv2d(width * 2, 1, 3, padding=1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)
