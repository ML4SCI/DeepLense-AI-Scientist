# Multi-seed replication pilot — full-data Experiments A and B

**Date: 2026-08-10** · Runner: `scripts/run_multi_seed.py` ·
Artifacts: `~/Personal/GSoC/deeplense_artifacts/pilot/` (outside the repository;
this document is the in-repo record written from them on 2026-08-12).

Two experiments, both on the **full** dataset (7,200 train / 1,800 val) with the
held-out test set (1,800) touched exactly once per final model. Test-set
discipline is enforced in code: the test `DatasetRef` is only constructed inside
`evaluate_on_test()`, which refuses to run until selection and tuning are marked
complete.

**Invocation.** `--seeds 0,1 --pipeline-runs 1 --num-candidates 10 --top-k 4
--rounds 2 --refine-variants 3 --candidate-epochs 10 --tune-epochs 25
--tune-iterations 4`, `DLENS_LLM=gpt-5.2`, torch 2.13.0 on MPS.

> **Baseline parameter count.** The `resnet34` arm is **21.280 M** parameters. An
> earlier value of 11.17 M in `docs/PAPER_RUN_RESULTS.md` was the resnet18 preset;
> see the correction note there. Physical confirmation: every resnet34 checkpoint
> in this pilot is 85,243,851 bytes ≈ 21.31 M float32 parameters.

---

## Experiment A — is the accuracy gap robust across seeds?

Fixed architectures, two seeds each, both tuned under the identical ReAct planner
protocol, then evaluated once on TEST.

### Per seed

| Arch | Params | Seed | Val acc | Test acc | Val AUC | Test AUC | Iters | Tune (s) |
|---|---|---|---|---|---|---|---|---|
| `cnn_medium_s4` | 2.539 M | 0 | 0.9194 | 0.9156 | 0.9836 | 0.9823 | 2 | 1646.1 |
| `cnn_medium_s4` | 2.539 M | 1 | 0.9106 | 0.9172 | 0.9855 | 0.9844 | 2 | 1605.0 |
| `resnet34` | 21.280 M | 0 | 0.9217 | 0.9083 | 0.9847 | 0.9788 | 2 | 1705.8 |
| `resnet34` | 21.280 M | 1 | 0.9128 | 0.9050 | 0.9815 | 0.9785 | 3 | 1977.3 |

### Means and spreads (n = 2 seeds)

| Arch | Val mean (spread) | Test mean (spread) | Val AUC | Test AUC |
|---|---|---|---|---|
| `cnn_medium_s4` | **0.9150** (0.0089) | **0.9164** (0.0017) | 0.9845 | 0.9833 |
| `resnet34` | **0.9172** (0.0089) | **0.9067** (0.0033) | 0.9831 | 0.9787 |

### Paired differences (`cnn_medium_s4` − `resnet34`, same seed)

| Metric | Seed 0 | Seed 1 | Mean |
|---|---|---|---|
| Validation | −0.0022 | −0.0022 | **−0.0022** |
| Test | +0.0072 | +0.0122 | **+0.0097** |

**Reading.** On test the searched architecture is ahead by ~1.0 point, consistent
in sign across both seeds, and the difference exceeds each arm's own seed spread
(0.0017 and 0.0033). On validation the sign reverses and the difference
(−0.0022) is well inside the seed spread (0.0089), i.e. noise. **With two seeds
no interval can be computed**, so the defensible claim is equivalence at 8.4x
fewer parameters, not superiority. Five or more seeds would be needed to bound
the difference.

### Planner behaviour

All four runs began overfit and the planner applied a regularization intervention
at iteration 0, reading only computed metrics:

| Run | it0 train / val / gap | Intervention |
|---|---|---|
| `cnn_medium_s4` seed 0 | 0.9978 / 0.8928 / 0.1050 | `augment`, `early_stop_patience=5` |
| `cnn_medium_s4` seed 1 | 0.9990 / 0.9006 / 0.0984 | `augment`, `early_stop_patience=5` |
| `resnet34` seed 0 | 0.9854 / 0.8206 / 0.1648 | `augment` |
| `resnet34` seed 1 | 0.9753 / 0.8533 / 0.1220 | `augment`, `early_stop_patience=5` |

Every run terminated on the gap-closed guard, not on a planner decision — see
`docs/AGENT_RELIABILITY.md` §0 for why that distinction matters.

### Seeding is genuinely varied

Verified rather than assumed: the two `cnn_medium_s4` runs differ on 112 lines of
their training logs and from epoch 1 onward (loss 1.1046/acc 0.3433 vs
1.0987/0.3543); the two `resnet34` runs differ on 134 lines. `DataLoader` is
constructed with `num_workers` unset (0), so no worker process exists that could
have re-imported a stale backend.

> **Known metadata defect.** `expA_resnet34_seed1/config.json` records
> `git_commit = 0742b03`. That is wrong: all four Experiment A runs were produced
> by the code at `66604c4`. The recording is per-run via `git rev-parse HEAD`, and
> the repository was checked out to a different branch mid-pilot. Training
> behaviour was unaffected (module caching; no worker processes), and the other
> three configs correctly record `66604c4`.

---

## Experiment B — does the agent independently converge on a small architecture?

One complete pipeline run at full data: generate → judge → shortlist training →
prune → refine → winner → ReAct tuning → TEST.

**Winner: `resnet_s3_deeper_d5-4-4_w32-64-128`, 1.491 M parameters.**
Search 1,284.7 s, tuning 522.9 s. Zero candidates dropped as unbuildable.

### Candidates proposed, with the judge's ranking

| Judge rank | Candidate | Params |
|---|---|---|
| **1** | `resnet_s3_deeper_d4-4-4_w32-64-128` | 1.472 M |
| 2 | `resnet_s2_wide_d3-3_w64-128` | 1.047 M |
| 3 | `resnet_s3_narrow_d2-2-2_w24-48-96` | 0.393 M |
| 4 | `cnn_s3_wide_d2-3-3_w64-128-256` | 1.884 M |
| 5 | `cnn_s4_mid_d2-2-3-3_w32-64-128-256` | 1.911 M |
| 6 | `resnet_s4_balanced_d2-2-3-3_w32-64-128-256` | 4.273 M |
| 7 | `cnn_s3_small_d1-2-2_w16-32-64` | 0.070 M |
| 8 | `resnet_s2_tiny_d2-2_w16-32` | 0.043 M |
| 9 | `cnn_s2_minimal_d2-2_w16-32` | 0.017 M |
| **10** | `resnet_s4_large_d3-4-6-3_w64-128-256-512` | 21.280 M |

Raw ranking (0-based candidate indices): `[5, 7, 2, 6, 3, 4, 1, 0, 9, 8]`.

**Worth noting for the paper:** the candidate the judge ranked **last** —
`resnet_s4_large_d3-4-6-3_w64-128-256-512`, 21.280 M — has exactly the block and
width configuration of the `resnet34` baseline, `(3,4,6,3)` / `(64,128,256,512)`.
The search independently proposed a ResNet-34-equivalent and independently placed
it bottom of ten on overfitting grounds.

### Judge rationale (verbatim extract)

> "Given 7.2k train / 1.8k val and 150x150 grayscale, this is a moderate-size
> dataset where too-tiny models can underfit and very large ones (esp. >10M
> params) are likely to overfit without strong regularization/augmentation. …
> Best bets are mid-capacity (≈0.3–2M params) with enough depth/stages to capture
> multi-scale lensing structure."

It ranked the 21.28 M candidate last as "very likely to overfit and be
inefficient for this dataset size", and the three sub-0.1 M candidates 7th–9th as
"most likely to underfit".

### Shortlist and refinement (10-epoch validation accuracy)

Round 0 — the judge's top four, trained:

| Candidate | Params | Val acc |
|---|---|---|
| `resnet_s3_deeper_d4-4-4_w32-64-128` | 1.472 M | **0.8856** |
| `resnet_s2_wide_d3-3_w64-128` | 1.047 M | 0.8806 |
| `resnet_s3_narrow_d2-2-2_w24-48-96` | 0.393 M | 0.8589 |
| `cnn_s3_wide_d2-3-3_w64-128-256` | 1.884 M | 0.7917 |

Round 1 — three novel variants of the survivor:

| Variant | Params | Val acc |
|---|---|---|
| `resnet_s3_deeper_d5-4-4_w32-64-128` | 1.491 M | **0.8906** |
| `resnet_s3_lighter_d3-4-4_w32-64-128` | 1.454 M | 0.8822 |
| `resnet_s3_wider_d4-4-4_w40-80-160` | 2.299 M | 0.8750 |

Refinement went **deeper rather than wider** for +0.019 M parameters.

### Final tuning and held-out test

| | Value |
|---|---|
| Val accuracy | 0.9167 (AUC 0.9819) |
| Train accuracy / gap | 0.9056 / −0.0111 |
| **Test accuracy** | **0.9083** (AUC 0.9806) |
| Iterations | 2 of 4 |
| Stop reason | `[loop guard] train/val gap -0.011 <= threshold 0.06 — gap closed.` |

Tuning trajectory: iteration 0 train 0.9704 / val 0.8850 / gap 0.0854 →
planner applied `augment` → iteration 1 train 0.9056 / val 0.9167 / gap −0.0111.

Per-class test F1: `no_sub` 0.9826, `cdm` 0.8677, `axion` 0.8740. Confusion
(rows = true no_sub / cdm / axion): `[[594,6,0],[14,538,48],[1,96,503]]` — nearly
all residual error is cdm↔axion, i.e. the physically hard discrimination rather
than detection.

---

## Cross-experiment summary for the paper

| Model | Params | Reduction vs baseline | Val | Test |
|---|---|---|---|---|
| Searched A `cnn_medium_s4` (Exp A, mean of 2 seeds) | 2.539 M | **8.4x** | 0.9150 | 0.9164 |
| Searched B `resnet_s3_deeper_d5-4-4` (Exp B) | 1.491 M | **14.3x** | 0.9167 | 0.9083 |
| Hand-picked `resnet34` (Exp A, mean of 2 seeds) | 21.280 M | — | 0.9172 | 0.9067 |

Note the asymmetry when quoting these together: the Searched A and baseline rows
are means over two seeds; the Searched B row is a single run. Reductions are
computed against 21.280 M.
