# Code-gen model comparison: gpt-5.2 vs gpt-5.6-luna

**Date: 2026-07-20** · Harness: `scripts/model_comparison_eval.py` · Raw data: `docs/model_comparison_results.json`

I ran both models through the simulation agent's real workflow — generate lenstronomy
code from a natural-language prompt → execute in the Docker sandbox (lenstronomy 1.9.2)
→ validate the output image — on all 9 synthetic prompts (Models I–III × no_sub / cdm /
axion), max 3 attempts each, identical prompts and settings. Real API calls, real
container execution, nothing mocked.

## Results

| Metric | gpt-5.2 | gpt-5.6-luna |
|---|---|---|
| Final pass (validated image) | **9/9** | 8/9 |
| First-attempt pass | 2/9 | **3/9** |
| Mean attempts | 1.78 | 1.78 |
| Structured output (schema) ok | 16/16 | 16/16 |
| Generated code parses (ast) | 16/16 | 16/16 |
| Mean generation latency | 13.7 s | **9.4 s** |
| Mean wall time per prompt (incl. sandbox) | 26.0 s | **18.1 s** |
| API | chat completions | **requires `/v1/responses`** |

Per-prompt: gpt-5.2 passed everything (Model_I prompts on attempt 1, the rest on
attempt 2). Luna passed 8 (its three Model_I passes were all attempt 1) but failed
`Model_III_axion` on all 3 attempts.

## Failure modes

- **Luna's one hard failure:** it imported `CDM` from `pyHalo.preset_models`, which
  doesn't exist in our pinned pyHalo, and never recovered across 3 retries.
  **Terminology corrected 2026-08-12: this is version blending, not a hallucinated
  API.** The import is *valid* for 2022-era pyHalo — DeepLenseSim's own `lens.py`
  uses exactly that import — so nothing was invented; the symbol existed, in a
  different release. The failure occurred against the image state before the pyHalo
  era-pin commit `c88954c`. Calling it hallucination misdescribes the failure mode
  and understates the case for version-pinning, which is the whole point of the
  grounding work; the corrected framing above is the one used throughout. Notably
  this is *outside* the lenstronomy cheat-sheet's coverage
  (we ground lenstronomy signatures, not pyHalo/deeplense) — same class of
  version-grounding gap we fixed for lenstronomy.
- **Integration gotcha (documented for the framework):** gpt-5.6-* rejects function
  tools on `/v1/chat/completions` with reasoning enabled (HTTP 400: "use /v1/responses
  or set reasoning_effort to 'none'"). Our `OpenAIModel` wrapper (chat completions)
  therefore can't drive it for structured output; the eval uses
  `OpenAIResponsesModel` (`responses:` prefix in the harness). Disabling reasoning
  instead would handicap the model, so I didn't benchmark that path.
- Both models' retry failures before eventual success were ordinary sandbox/validation
  errors that the error-feedback loop corrected — the loop is doing its job.

## Recommendation

**Keep gpt-5.2 as the default** for the code-gen agent: it's the only model with a
perfect pass rate, and it runs on the chat-completions path our framework wrapper
already uses. **Luna is worth revisiting** — it's ~30% faster end-to-end and slightly
better at first-attempt success, but it needs (a) a Responses-API model wrapper in the
framework and (b) cheat-sheet/RAG coverage extended to pyHalo before its one failure
mode is closed. Re-evaluate once ground-truth prompts from Michael land and the RAG
context is in place.
