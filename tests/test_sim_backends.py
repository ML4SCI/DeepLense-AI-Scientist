"""Mock backend, runner, output layout, and backend selection (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dlens.schemas import SimConfig, SimModelConfig, SubstructureType
from dlens.tools._sim_backends import MockBackend, deeplense_available, get_backend
from dlens.tools._sim_runner import execute_simulation


def _cfg(**kw):
    base = dict(substructure_type=SubstructureType.CDM, num_images=3)
    base.update(kw)
    return SimConfig(**base)


def test_mock_shapes_and_dtypes_match_model_config():
    mb = MockBackend(seed=0)
    imgs_i = mb.generate(_cfg(model_config_name=SimModelConfig.MODEL_I))
    imgs_ii = mb.generate(_cfg(model_config_name=SimModelConfig.MODEL_II))
    imgs_iii = mb.generate(_cfg(model_config_name=SimModelConfig.MODEL_III))
    assert all(im.shape == (150, 150) for im in imgs_i)
    assert all(im.shape == (64, 64) for im in imgs_ii)
    # Model_III (HST) is 64x64 float like Model_II — verified against a real
    # DeepLens run with lenstronomy's HST band config (simple_sim_2 -> 64x64).
    assert all(im.shape == (64, 64) for im in imgs_iii)
    assert np.issubdtype(imgs_i[0].dtype, np.integer)
    assert np.issubdtype(imgs_ii[0].dtype, np.floating)
    assert np.issubdtype(imgs_iii[0].dtype, np.floating)


def test_model_iii_accepted_end_to_end_by_schema_and_mock():
    cfg = SimConfig(
        substructure_type=SubstructureType.VORTEX,
        model_config_name=SimModelConfig.MODEL_III,
        axion_mass=1e-23,
        num_images=2,
    )
    assert cfg.model_config_name.value == "Model_III"
    imgs = MockBackend(seed=0).generate(cfg)
    assert len(imgs) == 2 and imgs[0].shape == (64, 64)


def test_mock_is_deterministic():
    a = MockBackend(seed=42).generate(_cfg(num_images=2))
    b = MockBackend(seed=42).generate(_cfg(num_images=2))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_get_backend_auto_falls_back_to_mock():
    assert deeplense_available() is False
    assert get_backend("auto").name == "mock"
    assert get_backend("mock").name == "mock"
    with pytest.raises(RuntimeError, match="not importable"):
        get_backend("deeplense")
    with pytest.raises(ValueError):
        get_backend("nonsense")


def test_runner_writes_expected_layout(tmp_path: Path):
    cfg = _cfg(
        substructure_type=SubstructureType.VORTEX,
        axion_mass=1e-23,
        z_source=1.5,
        model_config_name=SimModelConfig.MODEL_I,
        num_images=4,
    )
    out = execute_simulation(cfg, backend=MockBackend(seed=1), output_root=tmp_path, make_preview=False)

    assert out.num_generated == 4
    assert out.image_shape == (150, 150)
    assert out.backend == "mock"
    assert out.filenames == [f"vortex_{i:04d}.npy" for i in range(4)]

    run_dir = Path(out.output_dir)
    assert run_dir.parent == tmp_path
    for name in out.filenames:
        assert (run_dir / name).exists()

    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["run_id"] == out.run_id
    assert meta["image_shape"] == [150, 150]
    assert meta["config"]["substructure_type"] == "vortex"
    assert meta["config"]["axion_mass"] == 1e-23
