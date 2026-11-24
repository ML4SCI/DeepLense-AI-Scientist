"""Buildable x2 image-to-image models across the AI Scientist families.

These are SR backbones, not the classification models with a new label: each
forward pass returns an image at twice the input height and width.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = ("cnn", "srcnn", "resnet", "vit", "equivariant", "mlpmixer", "hybrid",
            "edsr", "rcan", "swinir", "hat", "dat", "imdn", "bsrn",
            "realesrgan", "sr3")


def canonical_sr_family(family: str) -> str:
    return "srcnn" if family == "scnn" else family


@dataclass(frozen=True)
class SRArchitecture:
    family: str
    width: int = 32
    depth: int = 2
    patch: int = 5
    image_size: int = 75

    def __post_init__(self) -> None:
        family = canonical_sr_family(self.family)
        if family not in FAMILIES:
            raise ValueError(f"unsupported SR family: {self.family}")
        object.__setattr__(self, "family", family)
        if self.width < 8 or self.depth < 1 or self.patch < 1 or self.image_size < 1:
            raise ValueError("width >= 8, depth >= 1, patch >= 1, and image_size >= 1 are required")

    @property
    def name(self) -> str:
        return f"{self.family}_w{self.width}_d{self.depth}_p{self.patch}"


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvFeatures(nn.Module):
    def __init__(self, width: int, depth: int, residual: bool):
        super().__init__()
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.blocks = nn.Sequential(*(
            ResidualBlock(width) if residual else
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
            for _ in range(depth)
        ))
        self.head = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(F.relu(self.stem(x))))


class SRCNN(nn.Module):
    """SRCNN-style refinement at HR size: 9x9 features, 1x1 mapping, 5x5 output.

    ``depth`` controls the number of 1x1 mapping layers. Unlike the generic
    CNN family, every convolution operates on the bicubic-upsampled image.
    """

    def __init__(self, width: int, depth: int):
        super().__init__()
        hidden = max(8, width // 2)
        layers: list[nn.Module] = [nn.Conv2d(1, width, 9, padding=4), nn.ReLU()]
        for index in range(depth):
            layers.extend((nn.Conv2d(width if index == 0 else hidden, hidden, 1),
                           nn.ReLU()))
        layers.append(nn.Conv2d(hidden, 1, 5, padding=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, width: int, tokens: int):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.token_mlp = nn.Sequential(nn.Linear(tokens, max(8, tokens // 2)), nn.GELU(),
                                       nn.Linear(max(8, tokens // 2), tokens))
        self.channel_norm = nn.LayerNorm(width)
        self.channel_mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                         nn.Linear(width * 2, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))




class TokenFeatures(nn.Module):
    def __init__(self, spec: SRArchitecture):
        super().__init__()
        self.patch = spec.patch
        self.embed = nn.Conv2d(1, spec.width, spec.patch, stride=spec.patch)
        tokens = ((spec.image_size + spec.patch - 1) // spec.patch) ** 2
        if spec.family in ("vit", "hybrid"):
            self.position = nn.Parameter(torch.zeros(1, tokens, spec.width))
            nn.init.trunc_normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=spec.width, nhead=4 if spec.width % 4 == 0 else 1,
                dim_feedforward=spec.width * 2, dropout=0, batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(layer, num_layers=spec.depth,
                                                enable_nested_tensor=False)
        else:
            self.blocks = nn.ModuleList(MixerBlock(spec.width, tokens)
                                        for _ in range(spec.depth))
        self.decode = nn.ConvTranspose2d(spec.width, 1, spec.patch, stride=spec.patch)
        self.stem = ConvFeatures(spec.width, 1, residual=True) if spec.family == "hybrid" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if self.stem is not None:
            x = x + self.stem(x)
        p = self.patch
        x = F.pad(x, (0, (-width) % p, 0, (-height) % p))
        grid = self.embed(x)
        batch, channels, gh, gw = grid.shape
        tokens = grid.flatten(2).transpose(1, 2)
        if isinstance(self.blocks, nn.TransformerEncoder):
            tokens = self.blocks(tokens + self.position)
        else:
            for block in self.blocks:
                tokens = tokens + block(tokens)
        grid = tokens.transpose(1, 2).reshape(batch, channels, gh, gw)
        return self.decode(grid)[..., :height, :width]


class SRModel(nn.Module):
    def __init__(self, spec: SRArchitecture):
        super().__init__()
        self.spec = spec
        if spec.family == "srcnn":
            self.features = SRCNN(spec.width, spec.depth)
        elif spec.family in ("cnn", "resnet", "equivariant"):
            self.features = ConvFeatures(spec.width, spec.depth, spec.family != "cnn")
        else:
            self.features = TokenFeatures(spec)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        if self.spec.family == "srcnn":
            return base + self.features(base)
        if self.spec.family == "equivariant":
            # C4 group average shares predictions over four right-angle rotations.
            residual = sum(torch.rot90(self.features(torch.rot90(lr, k, (-2, -1))),
                                          -k, (-2, -1)) for k in range(4)) / 4
        else:
            residual = self.features(lr)
        return base + F.interpolate(residual, size=base.shape[-2:],
                                    mode="bilinear", align_corners=False)


def build_sr_model(spec: SRArchitecture) -> nn.Module:
    if spec.family == "sr3":
        from dlens.sr3_model import SR3Model

        return SR3Model(spec.width, spec.depth)
    if spec.family in {"edsr", "rcan", "swinir", "hat", "dat", "imdn", "bsrn",
                       "realesrgan"}:
        from dlens.sr_advanced_models import AdvancedSR

        return AdvancedSR(spec.family, spec.width, spec.depth)
    return SRModel(spec)
