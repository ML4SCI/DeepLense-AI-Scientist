# Simulation sandbox (V2 code-generation agent)

Isolated real-CPython environment with **lenstronomy** pre-installed, used to run and
validate the code produced by `SimulationCodegenAgent`. Chosen (2026-07-11) because
Pydantic's Monty / code mode cannot import native libraries (numpy/scipy/lenstronomy).

## Build
```bash
docker build -t dlens-lenstronomy:latest sandbox/
```

## How the agent uses it
`DockerSandbox` (in `src/dlens/tools/_sandbox.py`) writes the generated script to a temp
dir, then runs it isolated:
```bash
docker run --rm --network=none --memory=2g --cpus=1 --pids-limit=256 \
  -v "$JOB_DIR":/work -w /work -e DLENS_OUTPUT=/work/output.npy \
  dlens-lenstronomy:latest timeout 120 python /work/prog.py
```
The generated script saves its final image to `$DLENS_OUTPUT`; the agent reads it back
and validates it (runs cleanly, produces a finite, non-trivial 2-D array).

## Notes
- `LocalSandbox` exists only for **offline tests with trusted code** — it runs on the
  host and is **not** an isolation boundary. Never point it at untrusted LLM output.
- lenstronomy's pinned, dependency-heavy stack is exactly why we isolate it in a
  container rather than the host; the version pins in the Dockerfile are a starting
  point and may need tuning.
- Reference approach: https://medium.com/@kacperwlodarczyk/ai-agents-in-a-sandbox-docker-execution-ci-cd-integration-with-pydantic-deep-acc6e5647148
