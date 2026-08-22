# tools/_arch_families.py
"""Model builders for the architecture families beyond cnn/resnet.

Added 2026-08-13 at the mentor's request. The search's biggest stated limitation
was that it could only propose what the backend implements: two convolutional
families. An "unbiased search" over a space containing only convnets cannot say
anything about whether convnets are the right answer.

Every family here must ACTUALLY BUILD AND TRAIN on the target data
(150x150 single-channel, 3 classes) across roughly 0.02M-30M parameters. Nothing
is exposed to the generator that the backend cannot instantiate.

How ``depths``/``widths`` map per family (the generator prompt states the same,
without saying which family is preferable):

* ``vit``        — ``sum(depths)`` transformer blocks at embed dim ``widths[-1]``.
* ``mlpmixer``   — ``sum(depths)`` mixer blocks at hidden dim ``widths[-1]``.
* ``hybrid``     — convolutional stem over ``depths[:-1]``/``widths[:-1]``, then
                   ``depths[-1]`` transformer blocks at dim ``widths[-1]``.
* ``equivariant``— C4 (90-degree rotation) group-equivariant convnet;
                   ``depths`` = group-conv blocks per stage, ``widths`` = base
                   channels per stage (each carries 4 orientation channels).

The equivariant family is genuinely equivariant, not a convnet with a label:
weights are shared across the four 90-degree rotations of the kernel and the
orientation axis is cycled accordingly, with orientation-shared normalisation and
an orientation pool before the classifier. That makes the features invariant to
90-degree rotations of the input, which is a real inductive bias for lensed arcs
at arbitrary position angle. Verified numerically in
tests/test_arch_families.py rather than asserted: two earlier drafts of this
module passed a build test while being ordinary convnets with a misleading label
(one laid the output channels out as (rotation, channel) while the orientation
pool read them as (channel, rotation); one used stride-2 convolutions, which do
not commute with rotation).

Exactness has one documented boundary. Every 2x2 pool must see an even spatial
size, so the input is padded up to a multiple of 2^stages. For an EVEN input size
that shortfall is always even and the padding is symmetric, so invariance is
exact to float precision — our data is 150x150. For an ODD input size the
shortfall is always odd, no symmetric padding exists, and invariance is only
approximate. That case is tested and bounded rather than hidden.
"""

from __future__ import annotations

import math


def _require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "The torch backend requires PyTorch: uv pip install torch"
        ) from exc


def _heads_for(dim: int) -> int:
    """Largest head count in {8,4,2,1} that divides ``dim``."""
    for h in (8, 4, 2):
        if dim % h == 0:
            return h
    return 1


def _patch_for(size: int) -> int:
    """Pick a patch size that divides the image, preferring ~10-16 px patches.

    150 is divisible by 15 and 10; 64 by 16 and 8. Falling back to 1 would blow
    up the token count, so we pad instead (see ``_PatchEmbed``).
    """
    for p in (16, 15, 14, 12, 10, 8, 6, 5, 4):
        if size % p == 0:
            return p
    return 15


def build_family(arch, dropout: float = 0.0):
    """Build one of the non-cnn/resnet families. Returns an ``nn.Module``."""
    _require_torch()
    import torch
    from torch import nn
    import torch.nn.functional as F

    from dlens.schemas._model_design import ArchFamily

    num_classes = arch.num_classes or 2
    in_ch = arch.channels or 1
    h, w = arch.input_shape
    depths = tuple(arch.depths) if arch.depths else (2, 2)
    widths = tuple(arch.widths) if arch.widths else (64, 128)

    # ---------------------------------------------------------------- shared --
    class PatchEmbed(nn.Module):
        """Conv patchifier; pads the input up to a whole number of patches."""

        def __init__(self, cin, dim, patch):
            super().__init__()
            self.patch = patch
            self.proj = nn.Conv2d(cin, dim, kernel_size=patch, stride=patch)

        def forward(self, x):
            p = self.patch
            ph, pw = (-x.shape[-2]) % p, (-x.shape[-1]) % p
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph))
            x = self.proj(x)                      # [B, dim, H/p, W/p]
            return x.flatten(2).transpose(1, 2)   # [B, N, dim]

    class Block(nn.Module):
        """Pre-norm transformer block."""

        def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
            super().__init__()
            self.n1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
            self.n2 = nn.LayerNorm(dim)
            hidden = max(dim, int(dim * mlp_ratio))
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim)
            )

        def forward(self, x):
            y = self.n1(x)
            x = x + self.attn(y, y, y, need_weights=False)[0]
            return x + self.mlp(self.n2(x))

    class MixerBlock(nn.Module):
        """MLP-Mixer block: token mixing then channel mixing."""

        def __init__(self, dim, tokens, drop=0.0):
            super().__init__()
            th = max(8, tokens // 2)
            ch = max(dim, dim * 2)
            self.n1 = nn.LayerNorm(dim)
            self.token = nn.Sequential(
                nn.Linear(tokens, th), nn.GELU(), nn.Dropout(drop), nn.Linear(th, tokens)
            )
            self.n2 = nn.LayerNorm(dim)
            self.chan = nn.Sequential(
                nn.Linear(dim, ch), nn.GELU(), nn.Dropout(drop), nn.Linear(ch, dim)
            )

        def forward(self, x):                        # [B, N, D]
            y = self.n1(x).transpose(1, 2)           # [B, D, N]
            x = x + self.token(y).transpose(1, 2)
            return x + self.chan(self.n2(x))

    # ------------------------------------------------------------------ vit --
    if arch.family == ArchFamily.VIT:
        dim = int(widths[-1])
        blocks = max(1, sum(depths))
        patch = _patch_for(h)
        heads = _heads_for(dim)
        tokens = math.ceil(h / patch) * math.ceil(w / patch)

        class ViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = PatchEmbed(in_ch, dim, patch)
                self.cls = nn.Parameter(torch.zeros(1, 1, dim))
                self.pos = nn.Parameter(torch.zeros(1, tokens + 1, dim))
                nn.init.trunc_normal_(self.pos, std=0.02)
                nn.init.trunc_normal_(self.cls, std=0.02)
                self.blocks = nn.ModuleList(
                    [Block(dim, heads, drop=dropout) for _ in range(blocks)]
                )
                self.norm = nn.LayerNorm(dim)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(dim, num_classes)

            def forward(self, x):
                x = self.embed(x)
                x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
                x = x + self.pos[:, : x.shape[1]]
                for b in self.blocks:
                    x = b(x)
                return self.head(self.drop(self.norm(x)[:, 0]))

        return ViT()

    # ------------------------------------------------------------- mlpmixer --
    if arch.family == ArchFamily.MLPMIXER:
        dim = int(widths[-1])
        blocks = max(1, sum(depths))
        patch = _patch_for(h)
        tokens = math.ceil(h / patch) * math.ceil(w / patch)

        class Mixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = PatchEmbed(in_ch, dim, patch)
                self.blocks = nn.ModuleList(
                    [MixerBlock(dim, tokens, drop=dropout) for _ in range(blocks)]
                )
                self.norm = nn.LayerNorm(dim)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(dim, num_classes)

            def forward(self, x):
                x = self.embed(x)
                for b in self.blocks:
                    x = b(x)
                return self.head(self.drop(self.norm(x).mean(dim=1)))

        return Mixer()

    # ---------------------------------------------------------------- hybrid --
    if arch.family == ArchFamily.HYBRID:
        # Convolutional stem over all but the last stage, then attention blocks.
        conv_d = depths[:-1] if len(depths) > 1 else (2,)
        conv_w = widths[:-1] if len(widths) > 1 else (max(16, widths[-1] // 2),)
        dim = int(widths[-1])
        blocks = max(1, depths[-1])
        heads = _heads_for(dim)

        class Hybrid(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                cin = in_ch
                for d, wd in zip(conv_d, conv_w):
                    for _ in range(d):
                        layers += [
                            nn.Conv2d(cin, int(wd), 3, 1, 1, bias=False),
                            nn.BatchNorm2d(int(wd)),
                            nn.ReLU(inplace=True),
                        ]
                        cin = int(wd)
                    layers.append(nn.MaxPool2d(2))
                self.stem = nn.Sequential(*layers)
                self.proj = nn.Conv2d(cin, dim, 1)
                self.grid = 8                            # fixed 64-token grid
                self.pos = nn.Parameter(torch.zeros(1, 64, dim))
                nn.init.trunc_normal_(self.pos, std=0.02)
                self.blocks = nn.ModuleList(
                    [Block(dim, heads, drop=dropout) for _ in range(blocks)]
                )
                self.norm = nn.LayerNorm(dim)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(dim, num_classes)

            def _to_grid(self, x):
                """Pool to a fixed g x g token grid, divisibly.

                AdaptiveAvgPool2d is not implemented on MPS for non-divisible
                sizes, and the stem produces odd maps (150 -> 75 -> 37), so pad up
                to a multiple of g and use a plain average pool instead.
                """
                g = self.grid
                hh, ww = x.shape[-2], x.shape[-1]
                th, tw = -(-hh // g) * g, -(-ww // g) * g
                if (th, tw) != (hh, ww):
                    x = F.pad(x, (0, tw - ww, 0, th - hh))
                return F.avg_pool2d(x, kernel_size=(th // g, tw // g))

            def forward(self, x):
                x = self._to_grid(self.proj(self.stem(x)))
                x = x.flatten(2).transpose(1, 2) + self.pos
                for b in self.blocks:
                    x = b(x)
                return self.head(self.drop(self.norm(x).mean(dim=1)))

        return Hybrid()

    # ----------------------------------------------------------- equivariant --
    if arch.family == ArchFamily.EQUIVARIANT:

        class C4Conv(nn.Module):
            """C4 group-equivariant convolution (weights shared over 4 rotations).

            Lifting layer (``lifting=True``) maps a plain image to a feature map
            with an explicit orientation axis; later layers additionally cycle the
            input-orientation axis so equivariance composes.
            """

            def __init__(self, cin, cout, k=3, stride=1, lifting=False):
                super().__init__()
                self.cin, self.cout, self.k, self.stride = cin, cout, k, stride
                self.lifting = lifting
                shape = (cout, cin, k, k) if lifting else (cout, cin, 4, k, k)
                self.weight = nn.Parameter(torch.empty(*shape))
                nn.init.kaiming_normal_(
                    self.weight.view(cout, -1, k, k), mode="fan_out", nonlinearity="relu"
                )
                self.bias = nn.Parameter(torch.zeros(cout))

            def forward(self, x):
                k, s = self.k, self.stride
                # Channel layout is (channel, orientation): output index cout*4+r.
                # Stacking on a NEW axis 1 and reshaping preserves that order;
                # torch.cat(dim=0) would give (orientation, channel) and silently
                # break the orientation pooling downstream.
                if self.lifting:
                    ws = [torch.rot90(self.weight, r, dims=(2, 3)) for r in range(4)]
                    w4 = torch.stack(ws, dim=1)                    # [Cout, 4, Cin, k, k]
                    weight = w4.reshape(self.cout * 4, self.cin, k, k)
                else:
                    ws = []
                    for r in range(4):
                        wr = torch.rot90(self.weight, r, dims=(3, 4))   # rotate spatially
                        wr = torch.roll(wr, shifts=r, dims=2)           # cycle orientation
                        ws.append(wr)                                   # [Cout, Cin, 4, k, k]
                    w4 = torch.stack(ws, dim=1)                    # [Cout, 4, Cin, 4, k, k]
                    weight = w4.reshape(self.cout * 4, self.cin * 4, k, k)
                bias = self.bias.repeat_interleave(4)              # matches (cout, r)
                return F.conv2d(x, weight, bias, stride=s, padding=k // 2)

        class OrientBN(nn.Module):
            """BatchNorm sharing statistics across the four orientations."""

            def __init__(self, cout):
                super().__init__()
                self.cout = cout
                self.bn = nn.BatchNorm3d(cout)

            def forward(self, x):
                b, c4, hh, ww = x.shape
                x = x.view(b, self.cout, 4, hh, ww)
                return self.bn(x).view(b, c4, hh, ww)

        class Equivariant(nn.Module):
            def __init__(self):
                super().__init__()
                blocks = []
                cin, lifting = in_ch, True
                for s, (d, wd) in enumerate(zip(depths, widths)):
                    wd = int(wd)
                    for _ in range(d):
                        # stride is always 1: strided sampling does not commute
                        # with rotation, so it destroys equivariance outright.
                        # Downsampling is done by 2x2 max pooling instead, which
                        # commutes exactly when the spatial size is even.
                        blocks += [
                            C4Conv(cin, wd, 3, 1, lifting=lifting),
                            OrientBN(wd),
                            nn.ReLU(inplace=True),
                        ]
                        cin, lifting = wd, False
                    blocks.append(nn.MaxPool2d(2))
                self.body = nn.Sequential(*blocks)
                self.cout = cin
                # Every 2x2 pool must see an EVEN spatial size, or it discards a
                # strip that is not rotation-symmetric and equivariance is lost.
                # Padding the input symmetrically up to a multiple of 2^stages
                # keeps every stage even; symmetric padding of a square commutes
                # with 90-degree rotation, so exactness is preserved.
                self.multiple = 2 ** len(depths)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(cin, num_classes)

            def _pad_symmetric(self, x):
                m = self.multiple
                out = []
                for size in (x.shape[-2], x.shape[-1]):
                    need = (-size) % m
                    out.append((need // 2, need - need // 2))
                (top, bot), (left, right) = out
                if any((top, bot, left, right)):
                    x = F.pad(x, (left, right, top, bot))
                return x

            def forward(self, x):
                x = self.body(self._pad_symmetric(x))
                b, c4, hh, ww = x.shape
                # Pool over orientation -> invariance to 90-degree rotations.
                x = x.view(b, self.cout, 4, hh, ww).amax(dim=2)
                x = F.adaptive_avg_pool2d(x, 1).flatten(1)
                return self.head(self.drop(x))

        return Equivariant()

    raise ValueError(f"_arch_families cannot build family {arch.family.value!r}")
