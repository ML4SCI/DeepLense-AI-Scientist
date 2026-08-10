# schemas/_simulation.py
"""Domain schemas for the gravitational-lensing simulation stage.

These map onto the ``sim_config`` / ``sim_output`` slots of the ``ExperimentRun``
described in docs/WORKFLOW_DESIGN.md: ``SimConfig`` is the validated specification
the Simulation Agent populates; ``SimOutput`` is the generation metadata returned
to downstream agents.

Pure domain models — they depend only on Pydantic (no ``agents`` import), so the
schemas package stays free of import cycles.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SubstructureType(str, Enum):
    """Types of dark matter substructure available in DeepLenseSim."""

    NO_SUBSTRUCTURE = "no_sub"
    CDM = "cdm"
    VORTEX = "vortex"


class SimModelConfig(str, Enum):
    """DeepLenseSim model configurations (named ``SimModelConfig`` to avoid
    confusion with the downstream neural-network *model design* stage).

    Model_I:   150x150 px, 0.05 arcsec/px, Gaussian PSF (``simple_sim``)
    Model_II:  64x64 px, Euclid-realistic instrument (``simple_sim_2``)
    Model_III: 64x64 px, HST-realistic instrument (``simple_sim_2``)

    Model_IV is deliberately NOT supported: upstream generates it with raw
    lenstronomy ``SimAPI`` + real galaxy images (Galaxy10 DECals, external
    ~2.7 GB dataset) rather than the ``DeepLens`` wrapper this stage uses.
    """

    MODEL_I = "Model_I"
    MODEL_II = "Model_II"
    MODEL_III = "Model_III"


class CosmologyParams(BaseModel):
    """Cosmological parameters for the simulation."""

    H0: float = Field(default=70.0, description="Hubble constant in km/s/Mpc")
    Om0: float = Field(default=0.3, description="Matter density parameter")
    Ob0: float = Field(default=0.05, description="Baryon density parameter")


class SimConfig(BaseModel):
    """Complete specification for a DeepLenseSim run (the ``sim_config``).

    All fields have sensible defaults so partial specifications work; the
    validator enforces the physics constraints (source behind the lens; axion
    mass present for vortices).
    """

    # ``model_config_name`` starts with ``model_`` — opt out of Pydantic's
    # protected namespace so the field name is kept.
    model_config = ConfigDict(protected_namespaces=())

    substructure_type: SubstructureType = Field(
        description="Type of dark matter substructure to simulate",
    )
    model_config_name: SimModelConfig = Field(
        default=SimModelConfig.MODEL_I,
        description="Which DeepLenseSim configuration to use (Model_I, Model_II, or Model_III)",
    )
    num_images: int = Field(
        default=5, ge=1, le=100, description="Number of images to generate (1-100)"
    )
    halo_mass: float = Field(default=1e12, description="Main halo mass in solar masses")
    z_halo: float = Field(
        default=0.5, gt=0, lt=5, description="Redshift of the dark matter halo (lens)"
    )
    z_source: float = Field(
        default=1.0, gt=0, lt=10, description="Redshift of the source galaxy"
    )
    axion_mass: Optional[float] = Field(
        default=None,
        description="Axion mass in eV (required for vortex substructure, typically 1e-24 to 1e-22)",
    )
    vortex_mass: float = Field(
        default=3e10, description="Vortex mass in solar masses (for vortex substructure)"
    )
    cosmology: CosmologyParams = Field(
        default_factory=CosmologyParams, description="Cosmological parameters"
    )

    @model_validator(mode="after")
    def _check_physics(self) -> "SimConfig":
        # Cross-field constraints at model level so they fire even when a
        # constrained field is left at its default.
        if self.z_source <= self.z_halo:
            raise ValueError(
                f"Source redshift ({self.z_source}) must be greater than "
                f"halo redshift ({self.z_halo})"
            )
        if self.substructure_type == SubstructureType.VORTEX and self.axion_mass is None:
            raise ValueError("axion_mass is required when substructure_type is vortex")
        return self


class SimOutput(BaseModel):
    """Generation metadata returned with each simulation batch (the ``sim_output``).

    Serialized to ``metadata.json`` alongside the generated arrays.
    """

    run_id: str = Field(description="Unique identifier for this simulation run")
    config: SimConfig = Field(description="The configuration used")
    num_generated: int = Field(description="Number of images successfully generated")
    image_shape: tuple[int, int] = Field(description="Shape of each generated image")
    pixel_value_range: tuple[float, float] = Field(
        description="(min, max) pixel values across all images"
    )
    timestamp: str = Field(description="When the simulation was executed (ISO 8601)")
    output_dir: str = Field(description="Directory where images are saved")
    filenames: list[str] = Field(description="List of saved .npy filenames")
    backend: str = Field(
        default="unknown",
        description="Which simulation backend produced the images (deeplense | mock)",
    )

    @property
    def image_paths(self) -> list[str]:
        """Full (relative) paths to the saved .npy images."""
        return [f"{self.output_dir}/{name}" for name in self.filenames]
