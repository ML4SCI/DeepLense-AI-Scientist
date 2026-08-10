# data/sim_prompts.py
"""Synthetic natural-language validation prompts for the code-generation agent.

Derived from the DeepLenseSim repo (github.com/mwt5345/DeepLenseSim) — Models I/II/III
x {no_sub, cdm, axion}. Each prompt describes, in plain English, what the corresponding
DeepLenseSim script simulates, so a correct code-gen output should reproduce an
equivalent image. Used as an initial validation set until Michael provides ground-truth
prompts. (Model_IV's scripts are empty in the repo and are omitted.)
"""

from __future__ import annotations

_INSTRUMENT = {
    "Model_I": "a simple Gaussian PSF (no realistic instrument), about 150x150 pixels",
    "Model_II": "the Euclid instrument configuration (about 64x64 pixels, magnitude-based source)",
    "Model_III": "the HST instrument configuration (magnitude-based source)",
}
_SUBSTRUCTURE = {
    "no_sub": "no dark-matter substructure",
    "cdm": "cold dark matter subhalos",
    "axion": "axion (vortex) substructure with a vortex mass of 3e10 solar masses",
}

SYNTHETIC_PROMPTS: list[dict[str, str]] = [
    {
        "name": f"{model}_{sub}",
        "model": model,
        "substructure": sub,
        "description": (
            "Simulate a strong gravitational-lensing image with a single 1e12 "
            f"solar-mass main halo, {sub_desc}, and a Sersic source galaxy, using "
            f"{inst}. Produce the final lensed image."
        ),
    }
    for model, inst in _INSTRUMENT.items()
    for sub, sub_desc in _SUBSTRUCTURE.items()
]
