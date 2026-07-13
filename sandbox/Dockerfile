# Real-CPython sandbox image for executing agent-generated lenstronomy code (V2).
#
# Agreed 2026-07-11: self-hosted Docker image with lenstronomy pre-installed is the
# execution/validation backend (Pydantic's Monty can't import native libs). The agent
# writes generated code to /work/prog.py and the sandbox runs it isolated; the script
# saves its image to $DLENS_OUTPUT (/work/output.npy), which is read back for validation.
#
# NOTE: lenstronomy==1.9.2 is old and dependency-heavy. The pins below are VERIFIED
# working on Python 3.10 (numpy<2, scipy<1.14, astropy<6 -> scipy 1.13.x, astropy
# 5.3.x): newer astropy removes `isiterable`, which lenstronomy 1.9.2 imports, and
# newer scipy drops APIs it uses. Keep these pins in sync if the lenstronomy pin moves.

FROM python:3.10-slim

# Patch base-image CVEs where possible (the container also runs non-root with
# --network=none, so exposure is minimal).
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Core scientific stack + lenstronomy (era-matched pins; see NOTE above).
RUN pip install --no-cache-dir \
        "numpy<2" "scipy<1.14" "astropy<6" colossus "lenstronomy==1.9.2"

# Substructure realizations (pyHalo) + the DeepLenseSim convenience wrapper (`deeplense`).
RUN pip install --no-cache-dir "git+https://github.com/dangilman/pyHalo.git"
RUN pip install --no-cache-dir "git+https://github.com/mwt5345/DeepLenseSim.git"

# Untrusted generated code runs as a non-root user.
RUN useradd -m -u 10001 sandbox
USER sandbox
WORKDIR /work

# Sanity check: the stack imports.
CMD ["python", "-c", "import lenstronomy, numpy, scipy, astropy; print('lenstronomy sandbox OK')"]
