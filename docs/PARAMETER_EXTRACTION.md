# Two-Stage Simulation Path: Parameter Extraction + Validation

Michael and Lucca proposed this design in the 2026-08-01 meeting: instead of going
straight from natural language to lenstronomy code, go **NL -> physical parameter
set -> deterministic validation -> code generation -> verification**. The reasoning
is that validating generated physics *code* is a trust problem, but validating the
*parameters* against known physical distributions is tractable and deterministic.

## Design

Four pieces, composed — the existing direct NL->code path is untouched and remains
the default (my 168-run grounding ablation depends on its exact behavior):

1. **`ParamExtractionAgent`** (`agents/_param_extraction.py`, gpt-5.2): NL ->
   `LensParameterSet` (`schemas/_lens_params.py`). The schema mirrors lenstronomy
   1.9.2's *actual* input dictionaries — model lists paired positionally with lists
   of kwargs dicts, camelCase `numPix`/`deltaPix` data block, `fwhm`-based PSF —
   all introspected from the pinned install, same method as the codegen cheat-sheet.
2. **`validate_parameters`** (`tools/_param_validator.py`): a plain function, no
   LLM. Every range is configurable (`ValidationRanges`) and every failure returns
   a specific, actionable message. Ranges are now empirically derived — see the
   provenance table below.
3. **`TwoStageSimulationAgent`** (`agents/_two_stage_codegen.py`): runs extraction,
   validates, retries extraction with the failure messages (bounded), then hands
   the validated set to the **unchanged** `SimulationCodegenAgent` as binding
   constraints. Codegen never runs on unvalidated parameters.
4. **`compare_code_to_params`** (`tools/_code_param_check.py`) — closes the loop:
   the validated parameters used to bind only via prompt. After codegen passes the
   sandbox, this parses the lenstronomy input dicts back OUT of the generated
   script (AST, not regex) and diffs them field-by-field against the validated
   set: matched / diverged / missing / **unresolved**. A value the script computes
   at runtime is reported unresolved and does NOT pass — never guessed. Divergence
   triggers a fresh codegen pass carrying the exact per-field feedback (bounded,
   default 2 passes). Resolution follows literals, names, constant-folded
   arithmetic, and dict/list subscripts of literals (`kwargs_data["exposure_time"]`
   — an idiom gpt-5.2 actually produces; the first live run failed on exactly this
   and the resolver was extended, still zero-guessing).

Select it explicitly: `TwoStageSimulationAgent` in code, or
`scripts/simulation_codegen_demo.py --two-stage`.

## SNR calibration against real data

I measured SNR on real Model_I images from the unmodified recipe
(`~/Personal/GSoC/deeplense_data`, 150x150 Poisson-count images):

| Definition | Real images (n) | median | IQR | p5 / p95 |
|---|---|---|---|---|
| aperture-integrated | 2,400 (train+test, all 3 classes) | **304.9** | 269.9–353.8 | 230.9 / 449.5 |
| peak-pixel | 900 | **18.1** | 16.3–20.1 | 14.8 / 22.0 |

The story, honestly: my v1 estimator was peak-pixel without any lensing term —
it under-predicted by ~6x and failed every live extraction. I "fixed" it by
switching to aperture-integrated — self-consistent, but real images score ~305
under that definition, nowhere near the nominal "SNR ~25" from the meeting. The
peak-pixel measurement lands at ~18, which IS the same scale as ~25 — so
peak-pixel is evidently the definition family the recipe means, and what was
actually missing was the lensing amplification.

The estimator is now **peak-pixel x an empirical lensing boost of 5.9**,
calibrated so canonical recipe parameters (amp=20, deltaPix=0.05,
background_rms=1e-2, lens.py L196/L229-233) reproduce the measured median. Check:
the recipe's own exposure draw `10**U(3, 3.5)` (lens.py L230) then predicts SNR
14.8 / 18.1 / 21.2 at t = 1000 / 1778 / 3162 s — against measured p5/median/p95 of
14.8 / 18.1 / 22.0. The acceptance band is [14, 60]: floor = measured p5 (which is
also exactly the recipe's lowest-exposure prediction), ceiling = a ~3x-p95
runaway-amp guard (heuristic).

**Flag for Lucca:** real DeepLenseSim Model_I images measure ~18 peak SNR — a
strict ">= 25" threshold would reject *every real image the recipe produces*.
Either his SNR definition differs in detail or his own sims are brighter; needs
his exact definition eventually.

## Validation ranges and provenance

Sources: `recipe <file>:<line>` = DeepLenseSim's actual code; `measured` = the
real-image statistics above; `1.9.2` = introspected lenstronomy;
`Collett 2015` = verified population statistics (corroboration only);
`heuristic` = still a guess.

| Check | Default | Provenance |
|---|---|---|
| z_lens < z_source | — | physics; recipe fixes z_halo=0.5, z_gal=1.0 (lens.py:37) |
| z guards | (0,5) / (0,10) | **heuristic** (recipe fixes single values; OM10 medians not verifiable enough to cite) |
| Einstein radius | 0.12–13.0" | derived: recipe mass_to_radius (lens.py:69-101) over the halo rails: 1e10->0.128", 1e12->1.281" (canonical), 1e14->12.8". Corroborated by Collett 2015: Euclid-discoverable theta_E ~ 0.66+/-0.40" |
| \|e1\|, \|e2\| | <= 0.5 | **heuristic**; recipe fixes lens (0.1, 0) (lens.py:113), source (-0.1, 0.1) (lens.py:196) |
| n_sersic | 0.36–8 | floor: lenstronomy numerics; upper **heuristic**; recipe fixes n=1 (lens.py:196) |
| numPix | 16–1024 | guard around recipe 150 (lens.py:231) and 64 (lens.py:295) |
| deltaPix | 0.01–0.5" | spans recipe 0.05 (lens.py:232) + 1.9.2 configs: HST 0.08, Euclid 0.101, LSST 0.2, DES 0.263 (margins **heuristic**; no Roman config exists in 1.9.2) |
| PSF | fwhm>0, pixel_size==deltaPix | 1.9.2 API; recipe fwhm=0.087 (lens.py:233) |
| halo mass | 1e10–1e14 M_sun | **heuristic rails** around the FIXED recipe 1e12 (Model_I/sim_no_sub.py:13) |
| axion mass | 1e-24–1e-22 eV | derived EXACTLY: recipe draw 10**U(-24,-22) (Model_I/sim_axion.py:12) |
| vortex mass | 1e9–1e12 M_sun | **heuristic rails** around the FIXED recipe 3e10 (Model_I/sim_axion.py:17) |
| profile names/keys | exact 1.9.2 param_names | introspected |
| peak SNR | 14–60 (boost 5.9) | **measured** (900 real images); ceiling heuristic |

Recipe facts recorded for completeness (not yet in `LensParameterSet`): CDM
subhalos are 25 (Poisson) point masses drawn from 1e6–1e10 M_sun with beta=-0.9
(lens.py:52) placed at r ~ U(0.25, 2.0)" (lens.py:179); the vortex is 100 point
masses along a de-Broglie-length line (lens.py:131); source centers are drawn
U(-0.35, 0.35)" (lens.py:193).

## Live results (gpt-5.2)

- **Extraction + validation, 6 of my 28 eval prompts: 6/6 first-attempt pass**
  with the derived ranges; SNR estimates 18.8–29.0 — right in the real-data band
  (the noiseless custom-grid prompt correctly skips the SNR check).
- **Full two-stage incl. verification** (extract -> validate -> codegen -> Docker
  -> AST diff): PASSED, 19/19 fields verified, 2 codegen passes — the first pass
  was caught by the verifier and the per-field feedback fixed it. In the run
  before the subscript-resolver fix, verification correctly FAILED a
  sandbox-passing script — exactly the gap this stage was built to close.

## Slurm generation (`tools/_slurm.py`)

Unchanged: `write_job()` emits program + sbatch script, all site specifics in
`SlurmJobConfig`, never submits (no submit function exists); `check_sbatch()`
dry-runs via `sbatch --test-only` where available. Still untested on a real
cluster — I have no access.

## Still pending from Michael / Lucca

- **Michael's 10–20 ground-truth prompts -> lenstronomy input dictionaries** (the
  seam is unchanged: `LensParameterSet` fixtures + `params_to_codegen_notes()`).
- **Lucca's exact SNR definition** — my peak-with-boost estimator is calibrated to
  real Model_I images, but the boost (5.9) bakes in the recipe's geometry; and the
  ~18-vs-25 discrepancy above needs his input.
- **Genuinely still heuristic:** redshift guards, ellipticity bound, Sersic upper
  bound, halo/vortex mass rails (recipe fixes single values — only Lucca's
  population distributions can turn these into real ranges), deltaPix margins,
  and the SNR ceiling.
