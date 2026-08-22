# Broadened architecture search — six families, iterative beam search

**Date: 2026-08-15/16** · Branch `feat/broader-search` · gpt-5.2 · full data
(7,200 train / 1,800 val, 150x150, 3 classes) · MPS on one laptop.
Raw: `docs/broader_search/` (`run_partial.log`, `e4_budget_validity.json`,
`e4_rejected_10ep.json`, `e4_rejected_25ep.json`).

Two changes, both requested by the mentor: broaden the buildable space beyond
convnets, and make the search genuinely iterative. Then run it once for real.

---

## What changed

**Six buildable families** (was two): `cnn`, `resnet`, `vit`, `mlpmixer`,
`hybrid`, `equivariant`. Each reaches roughly 0.02M–30M+ parameters, so capacity
is independent of family choice. Every family has an offline build **and train**
test (`tests/test_arch_families.py`, 26 tests).

The `equivariant` family is a real C4 group-equivariant convnet — weights shared
across the four 90-degree rotations, orientation axis cycled, orientation pooled
before the classifier — verified numerically to be invariant to `rot90` to float
precision (~5e-7). Two earlier drafts passed a build test while being ordinary
convnets with a misleading label; both were caught by the numeric check, not by
inspection. See the module docstring for the one documented boundary (odd input
sizes cannot be exactly invariant; our data is 150x150).

**Iterative beam search** (was one round plus a shallow refinement of the
winner): R rounds of generate → judge → short-train → prune, with the winner
re-entering *full* generation. From round 2 the generator prompt carries every
measured result so far, so later rounds are informed rather than blind
resampling; structural repeats are dropped.

Bias control is unchanged and extends to the new families: the prompt states what
is buildable and how `depths`/`widths` are interpreted, never which family is
expected to do well. It does not mention that lensed arcs are rotationally
symmetric or that convolutions have been used in this domain. No web-search tool
is enabled anywhere in the pipeline.

---

## The run

N=10, k=4, R=3, `candidate_epochs=10`. **Rounds 1 and 2 completed; round 3 was
stopped partway** — see the runtime note at the end. Eight candidates were fully
evaluated across four families.

| Round | Candidate | Family | Params | Val @10ep | Train time |
|---|---|---|---|---|---|
| 1 | **`cnn_deeper_s4`** | cnn | 1.367 M | **0.9033** | 285 s |
| 2 | `cnn_balanced_s4_midplus` | cnn | 1.026 M | 0.8983 | 242 s |
| 2 | `cnn_deeper_wider_s4` | cnn | 1.414 M | 0.8872 | 385 s |
| 1 | `resnet_deep_narrow` | resnet | 3.277 M | 0.8778 | 95 s |
| 2 | `resnet_medium_balanced` | resnet | 3.184 M | 0.8761 | 91 s |
| 1 | `resnet_micro_4stage` | resnet | 1.228 M | 0.8433 | 48 s |
| 2 | `equivariant_c4_big` | equivariant | 1.920 M | 0.8056 | 3,161 s |
| 1 | `equivariant_c4_mid` | equivariant | 0.287 M | 0.7206 | 695 s |

**Global winner: `cnn_deeper_s4`, 1.367 M parameters, val 0.9033.**

Across all three rounds the generator proposed from **all six families**
(0.07M–21.4M), 0 unbuildable, 1 correctly dropped as a duplicate. Round 2's
winner (0.8983) was *worse* than round 1's and the code correctly kept round 1's
— the beam search does not drift to the most recent result.

### The judge disagreed with the measurements, consistently

The LLM judge ranked an **equivariant model first out of ten in all three
rounds**. Measured, those models finished last (0.7206) and second-last (0.8056)
of the eight evaluated. Its stated reasoning is a capacity-and-inductive-bias
argument; the outcome contradicts it every time.

This is the same shape as the judge result in `docs/AGENT_RELIABILITY.md`
(stable but not valid), on a different search and a broader space.

---

## E4 — is the 10-epoch proxy fair across families?

The search prunes on 10 epochs. Round 1's shortlist was retrained at 25 epochs
under the identical protocol, and — because that shortlist contained **no**
transformer or hybrid model — so were the three non-convolutional candidates the
judge ranked 5th, 6th and 7th and therefore never trained.

| Model | Family | @10 ep | @25 ep | Δ | gap @25 | Shortlisted |
|---|---|---|---|---|---|---|
| `cnn_deeper_s4` | cnn | 0.9033 | 0.9039 | **+0.0006** | +0.0955 | yes |
| `resnet_deep_narrow` | resnet | 0.8778 | 0.8922 | +0.0144 | +0.0592 | yes |
| `resnet_micro_4stage` | resnet | 0.8433 | 0.8294 | −0.0139 | +0.1706 | yes |
| `equivariant_c4_mid` | equivariant | 0.7206 | 0.7894 | +0.0688 | **+0.0052** | yes |
| `hybrid_conv2_attn6` | hybrid | 0.4600 | 0.7406 | **+0.2806** | +0.0777 | no |
| `vit_small_8b_d192` | vit | 0.3967 | 0.6567 | **+0.2600** | +0.3057 | no |
| `mlpmixer_small_8b_d256` | mlpmixer | 0.6683 | 0.6717 | +0.0034 | +0.3283 | no |

- **Mean gain, convolutional: +0.0004** (n=3)
- **Mean gain, non-convolutional: +0.1532** (n=4)

### Rank is preserved; magnitude is not

Kendall tau between the two budgets is **+1.000 on the shortlist** (winner
preserved) and **+0.905 over all seven** — the only swap is `hybrid` overtaking
`mlpmixer`. By the usual proxy-fidelity test, this budget looks sound.

It is not. The same proxy under-measures attention-based models by ~0.28
accuracy while under-measuring convnets by ~0.0006 — a factor of roughly 450.
At 10 epochs `vit_small_8b_d192` scores **0.3967 on a 3-class problem**, barely
above the 0.333 chance floor: the search was not measuring its quality, it was
measuring an untrained network.

**This is the finding.** A low-fidelity proxy can preserve rank almost perfectly
while being catastrophically family-biased in magnitude. Rank correlation — the
standard check, and the one used in `docs/AGENT_RELIABILITY.md` §E4 — would have
certified this proxy as fine.

### What this does and does not license

**Supported:** at every budget we ran, a plain CNN wins, and the shortlist
ranking is stable. The search's choice of `cnn_deeper_s4` stands.

**Not supported:** that transformers, mixers or equivariant designs are worse for
this task. None of them converged. `equivariant_c4_mid` still has a train/val gap
of **+0.0052** at 25 epochs — it is *still underfitting* while the CNN is
overfitting at +0.0955. `hybrid` gained 0.28 and is still climbing. We have not
measured any non-convolutional family's converged performance at any budget we
could afford.

The honest statement is: *these families are far slower to train and lose at
every budget we can afford*, not *these families are worse*.

**Recommendation:** raise `candidate_epochs` before any future search claims to
have surveyed this space fairly. At 10 epochs the search does not choose between
families; it selects convnets by construction.

---

## Runtime, and why round 3 was stopped

The equivariant family dominates cost. Removing stride was necessary for exact
equivariance, so it holds full 150x150 resolution far deeper than a strided
convnet, and group convolution carries 4 orientation channels throughout.

| Model | Params | s/epoch |
|---|---|---|
| `resnet_deep_narrow` | 3.277 M | 9.5 |
| `cnn_deeper_s4` | 1.367 M | 28.5 |
| `equivariant_c4_mid` | 0.287 M | 69.5 |
| `equivariant_c4_big` | 1.920 M | 316 |
| `equivariant_c4_wider` | 0.769 M | ~1,080 |

A 0.769 M equivariant model costs ~40x a 1.8 M CNN. Because the judge shortlisted
an equivariant candidate in every round, every round paid that tax: 88% of round
1–2 training time went to two equivariant models. Round 3's candidate was
projected at ~72 minutes on its own and the run was stopped there in favour of
the E4 measurement, which gates whether the search result is usable at all.

Two consequences worth recording:

1. **The compute-parsimony claim does not survive contact with this space.** "The
   whole pipeline runs in about half an hour on a laptop" was true for a
   convnet-only space. With an equivariant family available it is not.
2. Any future run of this search needs a per-candidate time budget, a cheaper
   equivariant implementation, or both.

## Not measured

Round 3's four candidates, the ReAct tuning phase, and held-out test metrics on
the winner. Tuning plus a single-shot test evaluation on `cnn_deeper_s4` is a
~25-minute job and was deliberately deferred until the budget question was
settled.
