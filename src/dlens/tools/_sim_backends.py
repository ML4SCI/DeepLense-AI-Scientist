# tools/_sim_backends.py
"""Simulation backends for the Data Simulation Agent.

Two interchangeable implementations of one small protocol:

* ``DeepLensBackend`` — wraps the real ``deeplense.lens.DeepLens`` class from
  DeepLenseSim (https://github.com/mwt5345/DeepLenseSim). Heavy, version-pinned
  scientific deps (lenstronomy==1.9.2, pyHalo, colossus); imported lazily so the
  framework works without them.
* ``MockBackend`` — deterministic synthetic arrays of the correct shape per model
  config. Lets the agent + pipeline run with no GPU, no LLM, and no DeepLenseSim
  install (offline-first, in keeping with the framework's local-first principle).

``get_backend()`` selects one ("auto" prefers the real backend, falling back to
the mock when DeepLenseSim isn't importable).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from dlens.schemas import SimConfig, SimModelConfig, SubstructureType

# Pixel grid size per model configuration (matches simple_sim / simple_sim_2).
_SHAPE_BY_MODEL: dict[SimModelConfig, int] = {
    SimModelConfig.MODEL_I: 150,
    SimModelConfig.MODEL_II: 64,
    SimModelConfig.MODEL_III: 64,
}


@runtime_checkable
class SimBackend(Protocol):
    """A simulation backend turns a validated config into a list of image arrays."""

    name: str

    def generate(self, config: SimConfig) -> list[np.ndarray]:
        """Return ``config.num_images`` 2-D image arrays."""
        ...


def deeplense_available() -> bool:
    """True if the real ``deeplense`` package can be imported."""
    import importlib.util

    return importlib.util.find_spec("deeplense") is not None


class DeepLensBackend:
    """Wraps DeepLenseSim's ``DeepLens`` class (verified call sequence per image):

    * ``DeepLens(H0=, Om0=, Ob0=, z_halo=, z_gal=, [axion_mass=])``
    * ``make_single_halo(halo_mass)``
    * substructure: ``make_no_sub()`` / ``make_old_cdm()`` / ``make_vortex(vortex_mass)``
    * Model_I:   ``make_source_light()``     then ``simple_sim()``      -> 150x150
    * Model_II:  ``set_instrument('Euclid')``, ``make_source_light_mag()``, ``simple_sim_2()`` -> 64x64
    * Model_III: HST band config (see below), ``make_source_light_mag()``, ``simple_sim_2()`` -> 64x64
    * read ``lens.image_real``

    Model_III note: upstream's ``set_instrument('hst')`` is a silent no-op in the
    published DeepLenseSim (only 'euclid' is implemented), so we set
    ``kwargs_single_band`` directly from lenstronomy's HST ObservationConfig —
    which is what the Model_III README describes ("HST observation
    characteristics as done by default in lenstronomy").
    """

    name = "deeplense"

    def generate(self, config: SimConfig) -> list[np.ndarray]:
        from deeplense.lens import DeepLens  # lazy: heavy scientific deps

        images: list[np.ndarray] = []
        for _ in range(config.num_images):
            kwargs = {
                "H0": config.cosmology.H0,
                "Om0": config.cosmology.Om0,
                "Ob0": config.cosmology.Ob0,
                "z_halo": config.z_halo,
                "z_gal": config.z_source,
            }
            if config.axion_mass is not None:
                kwargs["axion_mass"] = config.axion_mass

            lens = DeepLens(**kwargs)
            lens.make_single_halo(config.halo_mass)

            if config.substructure_type == SubstructureType.NO_SUBSTRUCTURE:
                lens.make_no_sub()
            elif config.substructure_type == SubstructureType.CDM:
                lens.make_old_cdm()
            elif config.substructure_type == SubstructureType.VORTEX:
                lens.make_vortex(config.vortex_mass)

            if config.model_config_name == SimModelConfig.MODEL_I:
                lens.make_source_light()
                lens.simple_sim()
            elif config.model_config_name == SimModelConfig.MODEL_II:
                lens.set_instrument("Euclid")
                lens.make_source_light_mag()
                lens.simple_sim_2()
            elif config.model_config_name == SimModelConfig.MODEL_III:
                # set_instrument('hst') no-ops upstream; configure the band
                # directly (F160W + Gaussian PSF, mirroring the euclid branch).
                from lenstronomy.SimulationAPI.ObservationConfig.HST import HST

                lens.kwargs_single_band = HST(
                    band="WFC3_F160W", psf_type="GAUSSIAN"
                ).kwargs_single_band()
                lens.make_source_light_mag()
                lens.simple_sim_2()

            images.append(np.asarray(lens.image_real))
        return images


class MockBackend:
    """Synthetic, deterministic stand-in for DeepLenseSim.

    Produces an Einstein-ring-like field whose shape/dtype mimic the real backend
    (Model_I -> 150x150 int counts via Poisson; Model_II/Model_III -> 64x64
    float), with substructure-dependent perturbations so the three classes look
    different.
    """

    name = "mock"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def generate(self, config: SimConfig) -> list[np.ndarray]:
        n = _SHAPE_BY_MODEL[config.model_config_name]
        rng = np.random.default_rng(self._seed)
        return [self._one(config, n, rng) for _ in range(config.num_images)]

    def _one(self, config: SimConfig, n: int, rng: np.random.Generator) -> np.ndarray:
        yy, xx = np.mgrid[0:n, 0:n].astype(float)
        jitter_x, jitter_y = rng.normal(0, n * 0.01, size=2)
        cx = (n - 1) / 2.0 + jitter_x
        cy = (n - 1) / 2.0 + jitter_y
        r = np.hypot(xx - cx, yy - cy)

        r0 = n * 0.30
        ring = np.exp(-((r - r0) ** 2) / (2 * (n * 0.04) ** 2))
        core = np.exp(-(r**2) / (2 * (n * 0.06) ** 2))
        field = ring + 0.4 * core

        if config.substructure_type == SubstructureType.CDM:
            for _ in range(8):
                px, py = rng.uniform(0, n, size=2)
                field += 0.15 * np.exp(
                    -(((xx - px) ** 2 + (yy - py) ** 2)) / (2 * (n * 0.02) ** 2)
                )
        elif config.substructure_type == SubstructureType.VORTEX:
            ang = rng.uniform(0, np.pi)
            line = np.abs((xx - cx) * np.sin(ang) - (yy - cy) * np.cos(ang))
            field += 0.25 * np.exp(-(line**2) / (2 * (n * 0.02) ** 2)) * (r < r0 * 1.3)

        field = np.clip(field, 0, None)

        if config.model_config_name == SimModelConfig.MODEL_I:
            return rng.poisson(field * 200.0).astype(np.int64)
        return (field * 4.0 + rng.normal(0, 1e-3, size=field.shape)).clip(min=0).astype(np.float64)


def get_backend(name: str = "auto", *, seed: int = 0) -> SimBackend:
    """Return a backend by name.

    * ``"deeplense"`` — the real backend (raises if DeepLenseSim isn't importable).
    * ``"mock"`` — the synthetic backend.
    * ``"auto"`` — the real backend if available, else the mock.
    """
    name = (name or "auto").lower()
    if name == "mock":
        return MockBackend(seed=seed)
    if name == "deeplense":
        if not deeplense_available():
            raise RuntimeError(
                "DeepLensBackend requested but 'deeplense' is not importable. "
                "Install the real backend (see README: pyHalo + lenstronomy==1.9.2) "
                "or use the mock backend."
            )
        return DeepLensBackend()
    if name == "auto":
        return DeepLensBackend() if deeplense_available() else MockBackend(seed=seed)
    raise ValueError(f"Unknown backend: {name!r} (expected auto | deeplense | mock)")
