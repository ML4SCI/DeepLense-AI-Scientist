# AI Scientist — tree architecture search + closed-loop tuning (real runs)

**Date: 2026-07-25** · Runner: `scripts/run_ai_scientist.py` · Full log: `docs/ai_scientist_run.json`
**Setup:** DeepLense Model_I (real generated data), 1,800 train / 1,800 val, 150×150 px, 3 classes; LLM = **gpt-5.6-luna** (Responses API); training real (torch/MPS, seeded).

This is the combined design from the Jul 25 discussion: **tree search picks the
architecture, the existing ReAct closed loop tunes it.** Everything below is from
real runs — no mocks.

## Phase A — tree search (main run)

**Generate.** Luna proposed 10 buildable candidates (0 rejected), genuinely diverse:
0.024M → 5.9M params, both `cnn` and `resnet` families, depth-vs-width trade-offs.

**Judge (LLM, pre-training).** Top-4: `cnn_medium_multistage` (0.29M), `resnet18_compact`
(0.70M), `cnn_small_wide` (0.09M), `resnet_deep_narrow` (1.09M) — reasoning favored
moderate capacity for 1,800 samples.

**Run (real, 10 epochs each) + prune (by real val accuracy):**

| candidate | params | val acc | gap | AUC | time |
|---|---|---|---|---|---|
| **cnn_medium_multistage** | 0.29M | **0.598** | +0.116 | 0.774 | 22.5s |
| resnet18_compact | 0.70M | 0.582 | +0.418 | 0.764 | 9.5s |
| resnet_deep_narrow | 1.09M | 0.579 | +0.419 | 0.777 | 11.3s |
| cnn_small_wide | 0.09M | 0.357 | +0.037 | 0.533 | 19.1s |

**Refine.** 3 novel variations of the winner (deduped against already-evaluated
specs) — none beat it. **Winner: `cnn_medium_multistage`** (cnn, depths [2,2,2,2],
widths [16,32,64,128], 0.29M params).

## Phase B — closed-loop tuning of the winner

| it | config | train | val | gap | AUC | planner decision |
|---|---|---|---|---|---|---|
| 0 | winner, no regularization (25 ep) | 0.9961 | 0.6294 | **0.3667** | 0.8068 | diagnose overfitting → `augment: true` |
| 1 | + augmentation | 0.7756 | **0.7489** | **0.0267** | **0.8996** | `[loop guard]` gap closed → stop |

The planner's diagnosis was metrics-only (verbatim: *"The model is strongly overfit:
train_accuracy is 0.9961 while validation accuracy is only 0.6294, a 0.3667 gap…"*).

## Runtime (Michael's demo-planning numbers)

| phase | time |
|---|---|
| candidate generation (Luna) | 7.1 s |
| LLM judging | 4.9 s |
| round-1 training (4 × 10 epochs) | ~62 s |
| refinement generate + training (3) | ~112 s |
| **Phase A total** | **3.1 min** |
| **Phase B (2 tuning iterations @ 25 epochs)** | **2.8 min** |
| **End-to-end total** | **5.9 min** |

Scaling notes: each extra tuning iteration ≈ +1.5–3 min; using the full 7,200-image
train set ≈ ~4× the training times (rough estimate: 20–30 min end-to-end). A
recorded demo run at the current settings comfortably fits in ~6–10 minutes.

## Honest findings (for the paper/discussion)

1. **The combined mechanism works end to end** — generation → judging → real
   ranking → pruning → refinement → closed-loop tuning, fully automated, ~6 min.
2. **Low-fidelity evaluation bias is real.** A first run with 4-epoch candidate
   training produced pure noise (all candidates ≈ chance) and a task prompt that
   mentioned "a large ResNet overfit" pushed the judge toward tiny models — the
   search "won" with an underfit 0.16M model (final val 0.643). Fixes: 10-epoch
   evaluations + a neutral task description. Even at 10 epochs, slow-starting large
   models are disadvantaged — evaluation fidelity vs search cost is the key knob to
   study next (classic NAS trade-off, now with an LLM in the loop).
3. **Best absolute result so far is still the planner-tuned resnet34** (val 0.802 on
   this subset, from the planner-only experiment) vs 0.749 for the search winner —
   i.e., the search adds automation and exploration, not yet accuracy. Options:
   longer candidate budgets, val-accuracy-aware stopping (not just gap), or letting
   the tuner also revisit architecture.
4. Prompts remain bias-free (no preferred-family hints); the search's convergence to
   a moderate plain CNN was its own, metrics-driven choice on this run.
