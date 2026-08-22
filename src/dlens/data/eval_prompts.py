# data/eval_prompts.py
"""Expanded evaluation suite for the code-generation agent (paper 2).

28 prompts = the 9 canonical DeepLenseSim-derived prompts + systematic parameter
variations (halo mass, redshifts, axion/vortex mass, instruments) + custom
grid/PSF requests (raw-lenstronomy path, explicit sizes -> enforced) + phrasing
styles (terse / verbose / bulleted), all within the capability envelope of the
four canonical model recipes (DeepLens wrapper ctor/args + raw lenstronomy).

Fields: name, category (canonical | param_variation | custom_grid | phrasing),
description, expected_shape (enforced only when the prompt pins an exact size).
"""

from __future__ import annotations

from dlens.data.sim_prompts import SYNTHETIC_PROMPTS

EXPANDED_PROMPTS: list[dict] = [
    # --- canonical 9 (sizes are approximate in text -> not enforced) ---
    *[
        {"name": p["name"], "category": "canonical", "description": p["description"],
         "expected_shape": None}
        for p in SYNTHETIC_PROMPTS
    ],
    # --- parameter variations (wrapper capability envelope) ---
    {"name": "var_halo_low", "category": "param_variation", "expected_shape": None,
     "description": "Simulate a strong gravitational-lensing image with a single 5e11 "
     "solar-mass main halo, no dark-matter substructure, and a Sersic source galaxy, "
     "using a simple Gaussian PSF. Produce the final lensed image."},
    {"name": "var_halo_high", "category": "param_variation", "expected_shape": None,
     "description": "Simulate a strong lensing image for a massive 3e12 solar-mass "
     "halo with cold dark matter subhalos and a Sersic source, simple Gaussian PSF."},
    {"name": "var_redshifts_1", "category": "param_variation", "expected_shape": None,
     "description": "Strong lensing simulation: lens halo of 1e12 solar masses at "
     "redshift 0.3, source galaxy at redshift 1.8, no substructure, Sersic source, "
     "Gaussian PSF. Output the lensed image."},
    {"name": "var_redshifts_2", "category": "param_variation", "expected_shape": None,
     "description": "Simulate a lens at z=0.8 with a source at z=2.5, halo mass 1e12 "
     "solar masses, CDM subhalo substructure, Sersic source light, Gaussian PSF."},
    {"name": "var_axion_specific", "category": "param_variation", "expected_shape": None,
     "description": "Generate a strong lensing image with axion (vortex) dark-matter "
     "substructure for an axion mass of 5e-23 eV, main halo 1e12 solar masses, "
     "vortex mass 3e10 solar masses, Sersic source, simple Gaussian PSF."},
    {"name": "var_vortex_light", "category": "param_variation", "expected_shape": None,
     "description": "Simulate an axion vortex lensing image with a LIGHTER vortex of "
     "1e10 solar masses (axion mass 1e-23 eV), 1e12 solar-mass main halo, Sersic "
     "source, Gaussian PSF."},
    {"name": "var_euclid_halo", "category": "param_variation", "expected_shape": None,
     "description": "Using the Euclid instrument configuration, simulate a lensing "
     "image with a 2e12 solar-mass halo and no substructure (magnitude-based Sersic "
     "source)."},
    {"name": "var_hst_cdm_z", "category": "param_variation", "expected_shape": None,
     "description": "HST instrument configuration: strong lens with CDM subhalos, "
     "halo mass 1e12 solar masses, lens at z=0.6, source at z=1.4, magnitude-based "
     "Sersic source."},
    {"name": "var_euclid_axion_mass", "category": "param_variation", "expected_shape": None,
     "description": "Euclid-like lensing image with vortex substructure: axion mass "
     "2e-24 eV, vortex mass 3e10 solar masses, main halo 1e12 solar masses."},
    {"name": "var_nosub_deep_source", "category": "param_variation", "expected_shape": None,
     "description": "Smooth lens (no substructure), halo 8e11 solar masses at z=0.5, "
     "distant source at z=3.0, Sersic light profile, simple Gaussian PSF."},
    {"name": "var_cdm_massive_close", "category": "param_variation", "expected_shape": None,
     "description": "CDM-substructure lensing image: 5e12 solar-mass halo at low "
     "redshift z=0.2 with source at z=1.0, Sersic source, Gaussian PSF."},
    {"name": "var_hst_nosub", "category": "param_variation", "expected_shape": None,
     "description": "No-substructure control image with the HST instrument "
     "configuration, default halo mass 1e12 solar masses, magnitude-based source."},
    # --- custom grid / PSF (raw lenstronomy; exact sizes -> ENFORCED) ---
    {"name": "grid_128_psf02", "category": "custom_grid", "expected_shape": (128, 128),
     "description": "Using raw lenstronomy (not the DeepLense wrapper), simulate a "
     "single strong-lensing image on a grid of exactly 128x128 pixels with pixel "
     "scale 0.05 arcsec and a Gaussian PSF of FWHM 0.2 arcsec. SIE lens with "
     "Einstein radius 1.2 arcsec, Sersic source. Save the final image."},
    {"name": "grid_96_psf01", "category": "custom_grid", "expected_shape": (96, 96),
     "description": "Raw lenstronomy simulation: exactly 96x96 pixels, 0.08 arcsec "
     "per pixel, Gaussian PSF FWHM 0.1 arcsec, SIE lens (theta_E 1.5 arcsec) plus "
     "external shear, elliptical Sersic source."},
    {"name": "grid_200_psf03", "category": "custom_grid", "expected_shape": (200, 200),
     "description": "Simulate with raw lenstronomy on exactly 200x200 pixels (0.04 "
     "arcsec/pixel), Gaussian PSF FWHM 0.3 arcsec, SIE lens with Einstein radius "
     "1.8 arcsec and a compact Sersic source."},
    {"name": "grid_64_raw", "category": "custom_grid", "expected_shape": (64, 64),
     "description": "A small raw-lenstronomy render: exactly 64x64 pixels at 0.1 "
     "arcsec/pixel, Gaussian PSF FWHM 0.15 arcsec, SIE lens, Sersic source."},
    # --- phrasing styles (same physics as canonical Model_I cdm) ---
    {"name": "style_terse", "category": "phrasing", "expected_shape": None,
     "description": "Lens sim. Halo 1e12 Msun. CDM subhalos. Sersic source. "
     "Gaussian PSF. ~150x150 px. Output image."},
    {"name": "style_verbose", "category": "phrasing", "expected_shape": None,
     "description": "I am preparing training data for a dark-matter classification "
     "study and I would like you to write a script that produces one simulated "
     "strong gravitational lensing observation. The lensing galaxy should be "
     "modeled as a single dark-matter halo of about one trillion (1e12) solar "
     "masses, and I want cold-dark-matter subhalo substructure included in the "
     "mass distribution. The background source should be a galaxy with a Sersic "
     "light profile, and the image should be convolved with a simple Gaussian "
     "point-spread function, roughly 150 by 150 pixels. Please save the resulting "
     "image at the end."},
    {"name": "style_bullets", "category": "phrasing", "expected_shape": None,
     "description": "Simulation request:\n- type: strong lensing image\n- main halo: "
     "1e12 solar masses\n- substructure: CDM subhalos\n- source: Sersic profile\n"
     "- PSF: Gaussian, simple\n- size: about 150x150 pixels\n- output: save final "
     "image array"},
]
