# Budget fairness: does a fixed epoch budget compare architecture families?

**Date: 2026-08-17** · Harnesses: `scripts/run_budget_validity.py`,
`scripts/finish_broader_search.py` · Raw:
`docs/broader_search/e4_*.json`, `docs/broader_search/round3.json`,
`docs/broader_search/tuning*.json`

This document did not exist when the papers first cited it; it is written from the
artifacts so the figures have an in-repo home rather than living only in JSON.

The architecture search prunes its shortlist on a short training run (default 10
epochs) and promotes the winner. With the candidate space broadened from two
families to six (`docs/` — see `src/dlens/tools/_torch_backends.py`), the question
is whether that budget measures architecture quality or merely convergence speed.

All runs: 7,200 train / 1,800 validation, 150x150 single channel, 3 classes,
seed 0, batch 32, cross-entropy, identical to `ArchitectureSearch._evaluate`
except for the epoch budget.

## Validation accuracy at three budgets

| Model | Family | Params | @10 ep | @25 ep | @50 ep | gap @50 |
|---|---|---|---|---|---|---|
| `cnn_deeper_s4` | cnn | 1.367 M | **0.9033** | **0.9039** | 0.8956 | +0.1044 |
| `resnet_deep_narrow` | resnet | 3.277 M | 0.8778 | 0.8922 | 0.8633 | +0.1363 |
| `hybrid_conv2_attn6` | hybrid | 4.838 M | 0.4600 | 0.7406 | **0.8456** | +0.0588 |
| `equivariant_c4_mid` | equivariant | 0.287 M | 0.7206 | 0.7894 | **0.8211** | +0.0321 |
| `resnet_micro_4stage` | resnet | 1.228 M | 0.8433 | 0.8294 | 0.8067 | +0.1933 |
| `mlpmixer_small_8b_d256` | mlpmixer | 2.252 M | 0.6683 | 0.6717 | 0.6456 | +0.3544 |
| `vit_small_8b_d192` | vit | 3.623 M | 0.3967 | 0.6567 | 0.6433 | +0.3567 |

The first four rows are round 1's shortlist (what the search trained). The last
three are the non-convolutional candidates the judge ranked 5th, 6th and 7th —
just outside the top-4 cut, so the search never trained them. Without those three
the question cannot be answered at all: round 1's shortlist contains exactly one
non-convolutional model, so any within-shortlist comparison is restricted to what
the filter already promoted.

## Rank by budget

- **@10:** cnn, resnet_deep, resnet_micro, equivariant, mixer, hybrid, vit
- **@25:** cnn, resnet_deep, resnet_micro, equivariant, hybrid, mixer, vit
- **@50:** cnn, resnet_deep, **hybrid**, equivariant, **resnet_micro**, mixer, vit

Kendall tau on the round-1 shortlist: **+1.000** between 10 and 25 epochs,
**+0.667** between 10 and 50. The winner is preserved at every budget. Across all
seven models `hybrid` climbs from 6th to 3rd and `equivariant` overtakes
`resnet_micro_4stage`.

## Mean gain from 10 to 25 epochs, by family group

| Group | n | Mean gain | Ratio vs convolutional |
|---|---|---|---|
| convolutional (cnn + resnet) | 3 | **+0.000385** | 1x |
| attention-based (hybrid + vit) | 2 | **+0.2703** | **702x** |
| all non-convolutional | 4 | +0.1532 | 398x |

At 10 epochs `vit_small_8b_d192` scored **0.3967** against a **0.333** chance
floor for three balanced classes: the budget had not measured that architecture,
it had failed to train it.

## Three convergence regimes coexist in one candidate pool

1. **Peak early, then decline into overfitting.** Convolutional and residual
   models peak at 25 epochs (or at 10, for `resnet_micro_4stage`) and lose accuracy
   by 50, with gaps of +0.104 to +0.193.
2. **Still improving at 50 epochs.** `hybrid_conv2_attn6` (gap +0.0588) and
   `equivariant_c4_mid` (gap +0.0321) are still underfitting at 50. The hybrid's
   deficit against the leading convolutional model narrows monotonically: **44.3
   points at 10 epochs, 16.3 at 25, 5.0 at 50**, with no sign of flattening.
3. **Converge by 25 and overfit severely.** `mlpmixer` and `vit` reach ~0.64--0.67
   and then overfit hard (gaps ~0.355). For these two the verdict is *not*
   budget-dependent: they are genuinely weaker on this task, not under-trained.

Because regimes 1 and 2 move in opposite directions, **no single fixed budget
compares these families fairly.** No non-convolutional family overtakes the
leading convolutional model at any budget tested, but for the hybrid and
equivariant families the comparison is unresolved rather than settled.

## Per-epoch training cost (7,200 images, MPS)

| Model | Family | Params | s/epoch |
|---|---|---|---|
| `resnet_micro_4stage` | resnet | 1.228 M | 4.4 |
| `resnet_deep_narrow` | resnet | 3.277 M | 9.0 |
| `mlpmixer_small_8b_d256` | mlpmixer | 2.252 M | 10.7 |
| `cnn_deeper_s4` | cnn | 1.367 M | 27.6 |
| `vit_small_8b_d192` | vit | 3.623 M | 40.3 |
| `hybrid_conv2_attn6` | hybrid | 4.838 M | 42.1 |
| `equivariant_c4_mid` | equivariant | 0.287 M | 68.9 |
| `equivariant_c4_big` | equivariant | 1.920 M | 316.1 |
| `equivariant_c4_wider` | equivariant | 0.769 M | **1080.0** |

**Range 4.4 to ~1,080 s/epoch, a factor of ~243** — and it is *anti-correlated*
with parameter count: the 0.287 M equivariant model costs 2.5x the 1.367 M CNN,
and the 0.769 M one costs 39x. Cause: exact C4 equivariance forbids strided
convolution, so the network holds full 150x150 resolution throughout while
carrying four orientation channels, and the group convolution rebuilds a
four-fold expanded weight tensor on every forward pass.

Consequence for the search: **88% of rounds 1--2 training time went to two
equivariant models**, and round 3's equivariant candidate was deferred at a
projected ~3 hours for a single 10-epoch evaluation — roughly 36x the three
convolutional candidates in that shortlist combined. Parameter count is a poor
proxy for compute once the space spans families.

## Search outcome and the two-seed result

Round 1 best 0.9033; round 2 produced nothing better; round 3 produced
`cnn_s4_mid_w224` at 0.9139, which the tuning loop raised to 0.9222 (seed 0). A
single-round search terminates at 0.9033 and never sees the winner.

Round 3 is **3 of 4 shortlisted candidates evaluated**. The unevaluated one is
`equivariant_c4_wider`, which that round's judge ranked **first**. No claim that
the search rejected equivariant designs is supportable for round 3: it ran out of
budget.

| | seed 0 | seed 1 | mean | spread |
|---|---|---|---|---|
| `cnn_s4_mid_w224` (1.192 M) val | 0.9222 | 0.9211 | 0.9217 | 0.0011 |
| `cnn_s4_mid_w224` **test** | 0.9278 | 0.9250 | **0.9264** | 0.0028 |
| `resnet34` (21.280 M) test | 0.9083 | 0.9050 | 0.9066 | 0.0033 |
| **paired difference** | +0.0195 | +0.0200 | **+0.0197** | — |

Consistent in sign and magnitude, ~7x the winner's own seed spread, at **17.9x**
fewer parameters.

> **Discarded run.** A first seed-1 attempt reported test 0.9344 but hit a Metal
> command-buffer fault mid-training ("operations may not have completed") while
> sharing the GPU with the 50-epoch sweep. The clean re-run reports 0.9250, so the
> faulted run was 0.0094 optimistic and would have overstated the paired mean by
> half. Retained as `docs/broader_search/tuning_seed1.json`; **do not use.**
