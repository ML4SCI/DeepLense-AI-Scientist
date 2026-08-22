"""Every buildable family must actually instantiate, run forward, and train.

The rule this file enforces: nothing is exposed to the candidate generator that
the backend cannot build. A family that the LLM can name but the backend cannot
instantiate would silently shrink the search space and invalidate the
"unbiased search" claim, so each family is checked here rather than assumed.

Offline: no LLM, no network, CPU only, tiny tensors.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dlens.schemas._model_design import (  # noqa: E402
    BUILDABLE_FAMILIES,
    ArchFamily,
    ArchitectureSpec,
)
from dlens.tools._torch_backends import build_model  # noqa: E402

H = W = 150
CLASSES = 3

# One representative config per family, sized to stay fast on CPU.
CONFIGS = {
    ArchFamily.CNN:         ([2, 2], [16, 32]),
    ArchFamily.RESNET:      ([2, 2], [16, 32]),
    ArchFamily.VIT:         ([1, 1], [32, 64]),
    ArchFamily.MLPMIXER:    ([1, 1], [32, 64]),
    ArchFamily.HYBRID:      ([1, 1], [16, 64]),
    ArchFamily.EQUIVARIANT: ([1, 1], [8, 16]),
}


def _spec(family: ArchFamily, depths, widths, hw: int = H) -> ArchitectureSpec:
    return ArchitectureSpec(
        name=f"{family.value}_test", family=family, input_shape=(hw, hw),
        channels=1, num_classes=CLASSES, depths=depths, widths=widths,
        physics_informed=(family is ArchFamily.EQUIVARIANT),
    )


def test_every_buildable_family_has_a_config():
    """If a family is declared buildable it must be covered here."""
    assert set(CONFIGS) == set(BUILDABLE_FAMILIES)


@pytest.mark.parametrize("family", sorted(CONFIGS, key=lambda f: f.value))
def test_family_builds_and_forwards(family: ArchFamily):
    depths, widths = CONFIGS[family]
    model = build_model(_spec(family, depths, widths))
    out = model(torch.randn(2, 1, H, W))
    assert out.shape == (2, CLASSES)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("family", sorted(CONFIGS, key=lambda f: f.value))
def test_family_trains(family: ArchFamily):
    """A few optimizer steps on a separable toy task must reduce the loss.

    Building is not enough: a family that cannot be optimized would poison the
    search with false negatives that look like architectural findings.
    """
    depths, widths = CONFIGS[family]
    torch.manual_seed(0)
    model = build_model(_spec(family, depths, widths))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossf = torch.nn.CrossEntropyLoss()

    # Class k puts a bright square in a different quadrant: trivially separable.
    x = torch.randn(6, 1, H, W) * 0.1
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    for i, c in enumerate(y):
        r, col = [(0, 0), (0, 75), (75, 0)][int(c)]
        x[i, 0, r:r + 60, col:col + 60] += 3.0

    model.train()
    first = None
    for step in range(25):
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    assert loss.item() < first, f"{family.value}: loss did not decrease ({first} -> {loss.item()})"


@pytest.mark.parametrize("family", sorted(CONFIGS, key=lambda f: f.value))
def test_family_parameter_range(family: ArchFamily):
    """Each family must reach both ends of the range the generator is told about."""
    small = build_model(_spec(family, [1, 1], [8, 16]))
    large = build_model(_spec(family, [4, 4, 4], [64, 128, 256]))
    n_small = sum(p.numel() for p in small.parameters()) / 1e6
    n_large = sum(p.numel() for p in large.parameters()) / 1e6
    assert n_small < 0.5, f"{family.value}: smallest config is {n_small:.3f}M, not small"
    assert n_large > 1.0, f"{family.value}: largest config is {n_large:.3f}M, not large"
    assert n_large > n_small


def test_equivariant_is_actually_rotation_invariant():
    """The equivariant family must earn its name.

    C4 group convolution shares weights across the four 90-degree rotations and
    cycles the orientation axis, and the network pools over orientation before
    the classifier, so the output must be invariant to rot90 of the input. This
    is checked numerically because two earlier implementations passed a build
    test while being ordinary convnets with a misleading label: one had the
    output channels laid out as (rotation, channel) while the orientation pool
    read them as (channel, rotation), and one used stride-2 convolutions, which
    do not commute with rotation.
    """
    torch.manual_seed(0)
    model = build_model(_spec(ArchFamily.EQUIVARIANT, [2, 2], [8, 16])).eval()
    x = torch.randn(3, 1, H, W)
    with torch.no_grad():
        base = model(x)
        for r in (1, 2, 3):
            rotated = model(torch.rot90(x, r, dims=(2, 3)))
            assert torch.allclose(rotated, base, atol=1e-4), (
                f"rot90 x{r}: max deviation {(rotated - base).abs().max().item():.3e}"
            )


@pytest.mark.parametrize("hw", [150, 128, 64, 96])
def test_equivariant_exact_for_even_inputs(hw: int):
    """Exact invariance holds for EVEN input sizes, which is what we train on.

    Each 2x2 pool must see an even spatial size or it discards a strip that is
    not rotation-symmetric. The model pads the input up to a multiple of
    2^stages; when the input size is even that padding is symmetric (an even
    size and an even multiple always leave an even shortfall), so exactness is
    preserved. Our data is 150x150.
    """
    torch.manual_seed(0)
    model = build_model(_spec(ArchFamily.EQUIVARIANT, [1, 1], [8, 8], hw=hw)).eval()
    x = torch.randn(2, 1, hw, hw)
    with torch.no_grad():
        base = model(x)
        for r in (1, 2, 3):
            rotated = model(torch.rot90(x, r, dims=(2, 3)))
            assert torch.allclose(rotated, base, atol=1e-4), f"not invariant at {hw}px, rot{r}"


@pytest.mark.parametrize("hw", [129, 99])
def test_equivariant_approximate_for_odd_inputs(hw: int):
    """ODD input sizes cannot be exactly invariant, and we do not pretend otherwise.

    With an odd size and an even pooling multiple the shortfall is always odd, so
    no symmetric zero-padding exists — a half-pixel asymmetry is unavoidable. The
    residual stays small, but this test exists so the limitation is recorded
    rather than discovered later. The training data is 150x150 (even), so this
    path is not exercised in practice.
    """
    torch.manual_seed(0)
    model = build_model(_spec(ArchFamily.EQUIVARIANT, [1, 1], [8, 8], hw=hw)).eval()
    x = torch.randn(2, 1, hw, hw)
    with torch.no_grad():
        base = model(x)
        devs = [(model(torch.rot90(x, r, dims=(2, 3))) - base).abs().max().item() for r in (1, 2, 3)]
    assert max(devs) > 1e-6, "unexpectedly exact; the parity argument may be wrong"
    assert max(devs) < 1.0, f"residual larger than expected at {hw}px: {max(devs):.3e}"
