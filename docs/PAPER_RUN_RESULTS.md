# Definitive paper run — gpt-5.2 end to end, held-out test evaluated once

**Date: 2026-07-27** · Runner: `scripts/run_paper_experiment.py` · Raw log: `docs/paper_run.json`
**This run supersedes the earlier mixed-model results** (gpt-5.2 planner-only run of
Jul 20; Luna combined run of Jul 25) for all headline paper claims.

> **CORRECTION 2026-08-12 — the resnet34 baseline is 21.28 M parameters, not 11.17 M.**
> This document originally recorded 11.17 M for the `resnet34_ref` arm. That is the
> parameter count of the **resnet18** preset, not resnet34. Every derived figure below
> is corrected accordingly (the parameter reduction becomes ~8.4x, not ~4.4x);
> accuracy, AUC and runtime numbers are unaffected.
>
> Evidence: `_BLOCKS_BY_NAME` in `src/dlens/tools/_torch_backends.py` maps
> `resnet34 -> (3,4,6,3)` with widths `(64,128,256,512)`, which for 1 input channel and
> 3 classes builds **21.280 M** parameters; the same table maps `resnet18 -> (2,2,2,2)`,
> which builds **11.172 M** — the number that had been recorded. Confirmed physically:
> every resnet34 checkpoint in the multi-seed pilot is **85,243,851 bytes** ≈ 21.31 M
> float32 parameters, while the `cnn_medium_s4` checkpoints are 10,187,621 bytes
> ≈ 2.55 M, matching that arm's stated 2.539 M. The model trained was therefore a
> genuine ResNet-34; only the recorded count was wrong.
>
> Every other architecture parameter count cited in this repository was re-derived the
> same way and is correct.

## Protocol

- **LLM: gpt-5.2 for every role** (candidate generation, judge, planner) — asserted
  at run start (`effective LLM: OpenAIChatModel model_name=gpt-5.2`). No Luna anywhere.
- **Held-out test set, used exactly once:** 1,800 fresh images (600/class) generated
  with the same DeepLenseSim Model_I recipe under explicit new seeds (777/778/779),
  verified **zero MD5 overlap** with all 9,000 train/val images. The test
  `DatasetRef` is constructed only inside `final_test_evaluation()`, which asserts
  both tuning arms are complete.
- **Selection on val only:** each arm's reported checkpoint is its best-val
  iteration; test metrics reported once, after all selection.
- Training seeded (`torch.manual_seed(0)`); torch 2.13.0 on MPS; 1,800-image train
  subset / 1,800 val, 150×150 px.

## Dataset

| Split | Images | Provenance |
|---|---|---|
| train (subset) | 1,800 (600/class) | Jul 20 generation, unseeded process entropy |
| val | 1,800 | same generation, seeded 80/20 split |
| **test** | **1,800 (600/class)** | **fresh Jul 27 generation, seeds 777/778/779; 0 hash overlap** |

## Phase A — tree architecture search (gpt-5.2)

10 candidates generated (0 unbuildable), 0.024M–31.5M params. Judge top-4 with the
explicit reasoning that ~1–5M params is the likely sweet spot for 1,800 samples:

| Candidate (top-4) | Params | Val acc (10 ep) | Gap | AUC |
|---|---|---|---|---|
| **cnn_medium_s4** (depths [2,3,3,4], widths [32,64,128,256]) | 2.54M | **0.7733** | +0.066 | 0.9154 |
| resnet_small_s4 | 2.80M | 0.7189 | +0.244 | 0.8774 |
| cnn_small_deep_s4 | 1.91M | 0.7022 | +0.028 | 0.8663 |
| resnet_wide_shallow_s3 | 1.23M | 0.5211 | +0.371 | 0.7118 |

Refinement round (3 novel variants of the winner): 0.618 / 0.513 / 0.620 — none
beat it. **Winner: cnn_medium_s4.** Note the contrast with the Luna run: with
10-epoch fidelity and gpt-5.2's judging, the search now prefers a mid-capacity model
(2.54M) rather than a tiny one — and it pays off below.

## Phase B — two tuning arms, identical protocol (ReAct planner, gpt-5.2)

| Arm | it | Train | Val | Gap | AUC | Decision |
|---|---|---|---|---|---|---|
| search_winner (cnn_medium_s4) | 0 | 0.9994 | 0.8233 | 0.1761 | 0.9434 | overfitting → `augment` |
| | 1 | 0.8872 | **0.8600** | 0.0272 | **0.9717** | guard: stop |
| resnet34_ref (21.28M) | 0 | 0.9839 | 0.7517 | 0.2322 | 0.8922 | overfitting → `augment` |
| | 1 | 0.8233 | **0.8167** | 0.0066 | 0.9283 | guard: stop |

Reproducibility spot-check: the resnet34 arm's iteration-0 numbers are **identical**
to the Jul 20 run (0.9839 / 0.7517 / 0.8922) — seeded training reproduces exactly.
The planner's remedy differed slightly (augment only, vs augment+early-stop on
Jul 20): same diagnosis, LLM decision nondeterminism on the remedy — noted honestly.

## Final: validation vs held-out test (the paper table)

| Arm | Params | Val acc | **Test acc** | Val AUC | **Test AUC** |
|---|---|---|---|---|---|
| **search winner (cnn_medium_s4)** | 2.54M | 0.8600 | **0.8706** | 0.9717 | **0.9700** |
| resnet34 reference | 21.28M | 0.8167 | 0.8100 | 0.9283 | 0.9293 |

Per-class on TEST (search winner): no_sub P/R/F1 0.894/1.000/0.944 · cdm
0.885/0.720/0.794 · axion 0.835/0.892/0.862; confusion (rows=true no_sub,cdm,axion):
[600,0,0] / [62,432,106] / [9,56,535]. resnet34_ref: no_sub 0.938/0.990/0.964 · cdm
0.776/0.612/0.684 · axion 0.716/0.828/0.768; confusion [594,6,0] / [36,367,197] /
[3,100,497]. Both models: errors concentrate in cdm↔axion, consistent with all
previous runs.

**Did val flatter the models? No.** Test tracks val closely for both arms
(search: +0.0106 on test; reference: −0.0067). The zero-overlap held-out set
confirms the val-selected models generalize.

**Headline:** the searched architecture beats the hand-picked resnet34 reference by
**+6.1 points test accuracy (0.8706 vs 0.8100)** and +0.041 test AUC with **~8.4×
fewer parameters**, under an identical same-LLM, same-data, same-protocol pipeline.

## Runtime

| Phase | Time |
|---|---|
| A — tree search (gen 14.8s, judge 7.7s, 7 short trainings) | 441.5 s (7.4 min) |
| B1 — tune search winner (2 iterations) | 418.0 s (7.0 min) |
| B2 — tune resnet34 reference (2 iterations) | 395.6 s (6.6 min) |
| C — single test evaluation (both arms) | 6.1 s |
| **Total** | **1,261.2 s (21.0 min)** |

## Deviations from plan

None. Knobs stayed at run defaults (10 candidates, top-4, 10-epoch candidate
training, tuning cap 4; both arms stopped early via the gap-closed guard at
iteration 1). The search winner differing from the Luna run's pick is expected
(different LLM + judged fidelity) and reported as-is.
