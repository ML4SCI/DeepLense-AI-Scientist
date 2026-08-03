# prompts/_codegen.py
"""System prompt for the V2 simulation code-generation agent.

Includes a version-grounded lenstronomy API cheat-sheet: the LLM knows lenstronomy
in general but blends argument names across versions (e.g. old camelCase ``numPix``
vs current snake_case ``num_pix``; ``sigma`` vs ``fwhm`` on PSF), which produces
plausible-but-wrong calls. Grounding the exact signatures of the pinned version
fixes that. Regenerate the sheet for the sandbox image's pinned version with
``scripts/gen_lenstronomy_cheatsheet.py`` (run it inside the image).
"""

from __future__ import annotations

# Verified by introspection against lenstronomy 1.9.2 (the sandbox image's pin —
# DeepLenseSim's version). NOTE: argument casing changes across versions (1.9.2 uses
# camelCase numPix/deltaPix; 1.14+ uses num_pix/delta_pix) — regenerate with the
# script above whenever the pin changes.
LENSTRONOMY_API_CHEATSHEET = """\
LENSTRONOMY API CHEAT-SHEET (verified against the sandbox's installed version, lenstronomy 1.9.2 — use these EXACT signatures):
- Imports:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    import lenstronomy.Util.simulation_util as sim_util
    import lenstronomy.Util.param_util as param_util
- Grid/data (this version uses camelCase numPix/deltaPix):
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exposure_time=None, background_rms=None)
    data_class = ImageData(**kwargs_data)
- PSF (use fwhm; there is NO 'sigma' argument):
    psf_class = PSF(psf_type='GAUSSIAN', fwhm=0.15, pixel_size=deltaPix)
- Models (first arg is the list; kwargs are LISTS of dicts, one per profile):
    lens_model_class = LensModel(lens_model_list=['SIE', 'SHEAR'])
    kwargs_lens = [{'theta_E': ..., 'e1': ..., 'e2': ..., 'center_x': 0, 'center_y': 0},
                   {'gamma1': ..., 'gamma2': ...}]
    source_model_class = LightModel(light_model_list=['SERSIC_ELLIPSE'])
    kwargs_source = [{'amp': ..., 'R_sersic': ..., 'n_sersic': ..., 'e1': ..., 'e2': ...,
                      'center_x': ..., 'center_y': ...}]
- Ellipticity/shear helpers:
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    gamma1, gamma2 = param_util.shear_polar2cartesian(phi, gamma)
- Render:
    image_model = ImageModel(data_class, psf_class, lens_model_class=lens_model_class,
                             source_model_class=source_model_class,
                             kwargs_numerics={'supersampling_factor': 1})
    image = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)\
"""

CODEGEN_SYSTEM_PROMPT = """\
You write a single, self-contained Python script that simulates a strong
gravitational-lensing image described in natural language, using lenstronomy.

HARD REQUIREMENTS:
- Output ONLY the code in the `code` field (no markdown fences, no prose outside it).
- The script must be runnable as-is and must SAVE the final 2-D image as a NumPy array
  to the path given by the environment variable `DLENS_OUTPUT`, e.g.:
      import os, numpy as np
      np.save(os.environ["DLENS_OUTPUT"], image)
- No plotting, no network access, no file writes other than DLENS_OUTPUT.
- Prefer explicit, standard lenstronomy usage.

TWO VALID APPROACHES:
1. For setups close to the standard DeepLense models, you MAY use the convenience
   wrapper available in the sandbox:
      from deeplense.lens import DeepLens
      lens = DeepLens(); lens.make_single_halo(1e12); lens.make_no_sub()
      lens.make_source_light(); lens.simple_sim(); image = lens.image_real
   (substructure: make_no_sub / make_old_cdm / make_vortex(3e10);
    instrument: set_instrument('Euclid'|'hst') + make_source_light_mag() + simple_sim_2()).
2. For NEW requirements, write raw lenstronomy directly: define the lens mass model
   (`LensModel`), the source light (`LightModel`), the imaging data + PSF, build an
   `ImageModel` (or `SimulationAPI`), render the image, and save it.

Keep it minimal and correct. In `reasoning`, briefly note the lens model, source,
instrument, and substructure you chose and why.

""" + LENSTRONOMY_API_CHEATSHEET
