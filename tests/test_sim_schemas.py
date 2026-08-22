"""Simulation domain-schema validation and serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlens.schemas import SimConfig, SimModelConfig, SimOutput, SubstructureType


def test_defaults_are_sensible():
    cfg = SimConfig(substructure_type=SubstructureType.NO_SUBSTRUCTURE)
    assert cfg.model_config_name == SimModelConfig.MODEL_I
    assert cfg.num_images == 5
    assert cfg.z_halo == 0.5 and cfg.z_source == 1.0
    assert cfg.cosmology.H0 == 70.0


def test_vortex_requires_axion_mass_even_when_omitted():
    with pytest.raises(ValidationError, match="axion_mass is required"):
        SimConfig(substructure_type=SubstructureType.VORTEX, num_images=2)
    SimConfig(substructure_type=SubstructureType.VORTEX, axion_mass=1e-23, z_source=1.5)


def test_source_must_be_behind_lens_even_with_defaulted_zsource():
    with pytest.raises(ValidationError, match="greater than"):
        SimConfig(substructure_type=SubstructureType.CDM, z_halo=2.0)


def test_num_images_bounds():
    with pytest.raises(ValidationError):
        SimConfig(substructure_type=SubstructureType.CDM, num_images=0)
    with pytest.raises(ValidationError):
        SimConfig(substructure_type=SubstructureType.CDM, num_images=101)


def test_output_serialization_and_image_paths():
    cfg = SimConfig(substructure_type=SubstructureType.CDM)
    out = SimOutput(
        run_id="abcd1234",
        config=cfg,
        num_generated=2,
        image_shape=(64, 64),
        pixel_value_range=(0.0, 3.5),
        timestamp="2026-06-06T00:00:00+00:00",
        output_dir="simulations/abcd1234",
        filenames=["cdm_0000.npy", "cdm_0001.npy"],
        backend="mock",
    )
    dumped = out.model_dump(mode="json")
    assert dumped["config"]["model_config_name"] == "Model_I"
    assert dumped["image_shape"] == [64, 64]
    assert out.image_paths == [
        "simulations/abcd1234/cdm_0000.npy",
        "simulations/abcd1234/cdm_0001.npy",
    ]
