"""Regenerate the lenstronomy API cheat-sheet from the INSTALLED version.

The code-gen prompt embeds exact call signatures so the LLM doesn't blend argument
names across lenstronomy versions. Whenever the sandbox image pins a different
lenstronomy (e.g. 1.9.2 for DeepLenseSim), run this INSIDE that environment and
paste the output into LENSTRONOMY_API_CHEATSHEET in src/dlens/prompts/_codegen.py:

    docker run --rm dlens-lenstronomy:latest python - < scripts/gen_lenstronomy_cheatsheet.py
"""

from __future__ import annotations

import inspect


def _sig(fn) -> str:
    try:
        return str(inspect.signature(fn))
    except (ValueError, TypeError):
        return "(<signature unavailable>)"


def main() -> None:
    import lenstronomy
    import lenstronomy.Util.param_util as param_util
    import lenstronomy.Util.simulation_util as sim_util
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel

    print(f"# lenstronomy {lenstronomy.__version__} — verified signatures\n")
    for name, fn in [
        ("sim_util.data_configure_simple", sim_util.data_configure_simple),
        ("PSF", PSF.__init__),
        ("ImageData", ImageData.__init__),
        ("LensModel", LensModel.__init__),
        ("LightModel", LightModel.__init__),
        ("ImageModel", ImageModel.__init__),
        ("ImageModel.image", ImageModel.image),
        ("param_util.phi_q2_ellipticity", param_util.phi_q2_ellipticity),
        ("param_util.shear_polar2cartesian", param_util.shear_polar2cartesian),
    ]:
        sig = _sig(fn).replace("(self, ", "(").replace("(self)", "()")
        print(f"{name}{sig}")


if __name__ == "__main__":
    main()
