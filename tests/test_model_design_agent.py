"""Model Design Agent (#2) + deterministic recommender, fully offline."""

from __future__ import annotations

import asyncio

from dlens.agents._model_design import ModelDesignAgent, ModelDesignReport
from dlens.agents._scripted_model_design import make_scripted_model_design_model
from dlens.schemas._model_design import (
    ArchFamily,
    DatasetCharacteristics,
    TaskType,
    characteristics_from_sim_outputs,
)
from dlens.schemas._simulation import SimConfig, SimOutput, SubstructureType
from dlens.tools._model_design import recommend_baseline


def _sim_output(sub: SubstructureType, n: int = 10) -> SimOutput:
    return SimOutput(
        run_id=f"r_{sub.value}",
        config=SimConfig(substructure_type=sub, axion_mass=1e-23 if sub == SubstructureType.VORTEX else None,
                         z_source=1.5),
        num_generated=n,
        image_shape=(64, 64),
        pixel_value_range=(0.0, 4.0),
        timestamp="2026-06-06T00:00:00+00:00",
        output_dir=f"sim/{sub.value}",
        filenames=[f"{sub.value}_{i:04d}.npy" for i in range(n)],
        backend="mock",
    )


def test_recommend_baseline_small_images_classification():
    ch = DatasetCharacteristics(task=TaskType.CLASSIFICATION, image_shape=(64, 64), num_classes=3)
    rec = recommend_baseline(ch)
    assert rec["architecture"]["name"] == "resnet18"
    assert rec["architecture"]["family"] == ArchFamily.RESNET.value
    assert rec["architecture"]["num_classes"] == 3
    assert rec["training_config"]["loss"] == "cross_entropy"


def test_recommend_baseline_large_images_regression():
    ch = DatasetCharacteristics(task=TaskType.REGRESSION, image_shape=(150, 150), channels=1)
    rec = recommend_baseline(ch)
    assert rec["architecture"]["name"] == "resnet34"
    assert rec["training_config"]["loss"] == "mse"


def test_characteristics_bridge_from_sim_outputs():
    outs = [_sim_output(SubstructureType.NO_SUBSTRUCTURE),
            _sim_output(SubstructureType.CDM),
            _sim_output(SubstructureType.VORTEX)]
    ch = characteristics_from_sim_outputs(outs)
    assert ch.image_shape == (64, 64)
    assert ch.num_classes == 3
    assert ch.num_samples == 30


def test_agent_designs_model_offline():
    async def run():
        ch = DatasetCharacteristics(task=TaskType.CLASSIFICATION, image_shape=(64, 64),
                                    num_classes=3, num_samples=30)
        model = make_scripted_model_design_model(ch.model_dump(mode="json"))
        agent = ModelDesignAgent(model=model)
        result = await agent.design(ch)
        assert isinstance(result.output, ModelDesignReport)
        assert result.output.architecture.name == "resnet18"
        assert result.output.architecture.num_classes == 3
        assert result.output.training_config.loss == "cross_entropy"

    asyncio.run(run())
