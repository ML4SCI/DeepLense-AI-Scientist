"""Generate real DeepLense Model_I images (runs INSIDE the dlens-lenstronomy container).

Mirrors DeepLenseSim/Model_I/sim_{no_sub,cdm,axion}.py exactly:
  no_sub: DeepLens(); make_single_halo(1e12); make_no_sub();  make_source_light(); simple_sim()
  cdm   : ...          make_single_halo(1e12); make_old_cdm(); make_source_light(); simple_sim()
  axion : ...          make_single_halo(1e12); make_vortex(3e10); make_source_light(); simple_sim()

Usage (in container): python gen_model1.py <class> <count> <outdir>
"""

import sys
import time

import numpy as np
from deeplense.lens import DeepLens

CLASS_SETUP = {
    "no_sub": lambda lens: lens.make_no_sub(),
    "cdm": lambda lens: lens.make_old_cdm(),
    "axion": lambda lens: lens.make_vortex(3e10),
}


def make_lens(cls: str) -> DeepLens:
    # sim_axion.py samples a per-image axion mass: 10**U(-24, -22), passed to the ctor.
    if cls == "axion":
        return DeepLens(axion_mass=10 ** np.random.uniform(-24, -22))
    return DeepLens()


def main() -> None:
    cls, count, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    setup = CLASS_SETUP[cls]
    t0 = time.time()
    done = 0
    attempts = 0
    while done < count and attempts < count * 3:
        attempts += 1
        try:
            lens = make_lens(cls)
            lens.make_single_halo(1e12)
            setup(lens)
            lens.make_source_light()
            lens.simple_sim()
            img = np.asarray(lens.image_real, dtype=np.float32)
            if img.ndim != 2 or not np.isfinite(img).all():
                continue
            np.save(f"{outdir}/{cls}_{done:05d}.npy", img)
            done += 1
            if done % 50 == 0 or done == count:
                rate = done / (time.time() - t0)
                print(f"[{cls}] {done}/{count}  ({rate:.2f} img/s)", flush=True)
        except Exception as exc:  # rare bad realizations: skip, keep going
            print(f"[{cls}] skip ({type(exc).__name__}: {exc})", flush=True)
    print(f"[{cls}] DONE {done}/{count} in {time.time()-t0:.0f}s ({attempts} attempts)", flush=True)


if __name__ == "__main__":
    main()
