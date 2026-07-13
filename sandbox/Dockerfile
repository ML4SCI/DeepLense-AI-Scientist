# Real-CPython sandbox image for executing agent-generated lenstronomy code (V2).
#
# Agreed 2026-07-11: self-hosted Docker image with lenstronomy pre-installed is the
# execution/validation backend (Pydantic's Monty can't import native libs). The agent
# writes generated code to /work/prog.py and the sandbox runs it isolated; the script
# saves its image to $DLENS_OUTPUT (/work/output.npy), which is read back for validation.
#
# NOTE: lenstronomy==1.9.2 is old and dependency-heavy; Python 3.10 + these pins are a
# known-workable starting point, but the exact numpy/scipy/astropy pins may still need
# tuning (this is the install pain we discussed). Treat versions as a starting point.

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Core scientific stack + lenstronomy (pin as needed).
RUN pip install --no-cache-dir \
        "numpy<2" scipy astropy colossus "lenstronomy==1.9.2"

# Substructure realizations (pyHalo) + the DeepLenseSim convenience wrapper (`deeplense`).
RUN pip install --no-cache-dir "git+https://github.com/dangilman/pyHalo.git"
RUN pip install --no-cache-dir "git+https://github.com/mwt5345/DeepLenseSim.git"

# Untrusted generated code runs as a non-root user.
RUN useradd -m -u 10001 sandbox
USER sandbox
WORKDIR /work

# Sanity check: the stack imports.
CMD ["python", "-c", "import lenstronomy, numpy, scipy, astropy; print('lenstronomy sandbox OK')"]
