# Two-Stage Simulation Path: Parameter Extraction + Validation

Michael and Lucca proposed this design in the 2026-08-01 meeting: instead of going
straight from natural language to lenstronomy code, go **NL -> physical parameter
set -> deterministic validation -> code generation**. The reasoning is that
validating generated physics *code* is a trust problem, but validating the
*parameters* against known physical distributions is tractable and deterministic.
Lucca's own pipeline samples system parameters from published distributions and
targets SNR >= ~25; this stage is built to plug his numbers in when they arrive.

## Design

Three pieces, composed — the existing direct NL->code path is untouched and remains
the default (my 168-run grounding ablation depends on its exact behavior):

1. **`ParamExtractionAgent`** (`agents/_param_extraction.py`, gpt-5.2): NL ->
   `LensParameterSet` (`schemas/_lens_params.py`). The schema mirrors lenstronomy
   1.9.2's *actual* input dictionaries — model lists paired positionally with lists
   of kwargs dicts, camelCase `numPix`/`deltaPix` data block, `fwhm`-based PSF —
   all introspected from the pinned install, same method as the codegen cheat-sheet.
2. **`validate_parameters`** (`tools/_param_validator.py`): a plain function, no
   LLM. Every range is configurable (`ValidationRanges`) and every failure returns
   a specific, actionable message.
3. **`TwoStageSimulationAgent`** (`agents/_two_stage_codegen.py`): runs extraction,
   validates, retries extraction with the failure messages (bounded, default 3,
   same style as the codegen retry loop), then hands the validated set to the
   **unchanged** `SimulationCodegenAgent` as binding constraints. Codegen never
   runs on unvalidated parameters.

Select it explicitly: `TwoStageSimulationAgent` in code, or
`scripts/simulation_codegen_demo.py --two-stage`.

## Validation ranges and provenance

| Check | Default | Source |
|---|---|---|
| z_lens < z_source, z ranges | (0,5) lens, (0,10) source | DeepLenseSim recipe (0.5/1.0) + V1 SimConfig |
| Einstein radius | 0.1–10 arcsec | heuristic — needs Lucca's ER-by-redshift distribution |
| \|e1\|, \|e2\| | <= 0.5 | lenstronomy e1/e2 convention (q > ~1/3); heuristic — needs Lucca |
| n_sersic | 0.36–8 | lenstronomy numerical floor ~0.36; upper heuristic — needs Lucca |
| R_sersic, exposure_time, background_rms | > 0 | definition |
| numPix / deltaPix | 16–1024 / 0.01–0.5" | DeepLenseSim recipes (150@0.05", 64px) + 1.9.2 ObservationConfigs (HST 0.08", Euclid 0.101") |
| PSF | GAUSSIAN needs fwhm (>0); pixel_size == deltaPix | lenstronomy 1.9.2 (no `sigma` argument) |
| halo mass | 1e10–1e14 M_sun | DeepLenseSim recipe (1e12 canonical) |
| axion mass | 1e-24–1e-22 eV | DeepLenseSim recipe |
| vortex mass | 1e9–1e12 M_sun | DeepLenseSim recipe (3e10 canonical) |
| profile names + kwargs keys | exact 1.9.2 `param_names` | introspected (SIE, SIS, EPL, SHEAR, NFW, SERSIC, SERSIC_ELLIPSE, GAUSSIAN) |
| integrated SNR | >= 25 | Lucca's target; estimator is my proxy (below) |

**SNR estimator.** Aperture-integrated: total Sersic flux ~ `amp * 2π R_sersic²`
against Poisson + background noise in a 2·R_sersic aperture, magnification ignored
(conservative). I first shipped a *peak-pixel* version and live extraction exposed
it as an order of magnitude too strict — gpt-5.2's textbook amp values scored
SNR 1–4 and retries couldn't realistically satisfy it. The integrated proxy is the
right shape but still a placeholder for Lucca's actual definition. Checks that
can't be computed (noiseless configs) are omitted, not failed.

## Live results (gpt-5.2, 6 of my 28 eval prompts)

All six validated on the **first attempt** after the SNR recalibration; prompt
specifics were honored (z=0.3/1.8 when asked; axion 5e-23 eV; 128 px grid with
0.2" PSF; HST-like 64 px @ 0.08", 5400 s for Model_III). One full end-to-end run
(extract -> validate -> codegen -> Docker sandbox) passed, first attempt, 64x64
image. The direct path's offline behavior is unchanged (33 tests pass, 8 of them
the original codegen suite).

## Slurm generation (`tools/_slurm.py`)

For Michael's job-submission pain: `write_job()` emits the generated program plus
a ready-to-submit sbatch script — partition/time/cpus/mem/array size/module
loads/env activation all in `SlurmJobConfig`, nothing site-specific hardcoded.
Array mode gives each task its own `DLENS_OUTPUT` file. **It never submits**
(no submit function exists); `check_sbatch()` dry-runs via `sbatch --test-only`
where available, else a structural check. I have no cluster access, so the real
`--test-only` path is untested until someone runs it on the cluster.

## Pending from Michael / Lucca

- **Michael's 10–20 ground-truth prompts -> lenstronomy input dictionaries.** The
  seam is `LensParameterSet` + `params_to_codegen_notes()`: his dictionaries load
  as fixtures and become both extraction-accuracy tests and codegen inputs. Until
  then I'm building against my 28-prompt suite.
- **Lucca's published distributions** (Einstein radius by redshift, halo mass,
  ellipticity, Sersic index) to replace every range marked "heuristic", and his
  **exact SNR definition** to replace my estimator.
- Cluster specifics (partition names, modules) for a real `SlurmJobConfig` preset,
  and one `sbatch --test-only` run to confirm the dry-run path.
