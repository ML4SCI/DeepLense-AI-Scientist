# Experiment planner — real closed-loop results (overfitting reduction)

**Date: 2026-07-20** · Runner: `scripts/run_experiment_loop.py` · Trajectory JSON: `docs/experiment_loop_trajectory.json`
**Setup:** real DeepLense Model_I data (1,800-image train subset, 600/class; full 1,800-image val), planner = **gpt-5.2** (ReAct strategy), training = torch on MPS, **seeded** so metric changes are attributable to the planner's decisions rather than initialization noise. Max 4 iterations; gap threshold 0.06.

## Trajectory

| it | Config the iteration ran with | Train acc | Val acc | Gap | Val AUC | Planner decision |
|---|---|---|---|---|---|---|
| 0 | resnet34 baseline — no regularization, 25 epochs | 0.9839 | 0.7517 | **0.2322** | 0.8922 | `train` → `{"augment": true, "early_stop_patience": 5}` |
| 1 | + augmentation + early stopping (stopped at best holdout) | 0.7926 | **0.8017** | **−0.0091** | **0.9187** | `[loop guard]` gap −0.009 ≤ 0.06 — **gap closed, stop** |

**Planner's iteration-0 reasoning (verbatim, truncated):**
> "Run 0 shows strong overfitting: train_accuracy=0.9839 vs val accuracy=0.7517 (gap ~0.23) with no augmentation and dropout=0.0. Next smallest intervention to improve generalization is to enable data augmentation; this should reduce memorization and help the weaker 'cdm' class (recall=0.6267)…"

## Summary

The closed loop did what it exists to do: starting from a deliberately overfitting
baseline, the planner **correctly diagnosed overfitting from the observed metrics
alone** (no domain hints in its prompt), chose a standard minimal remedy
(augmentation + early stopping), and one iteration later the train/val gap went from
**0.232 to ≈0** while validation accuracy **improved +5.0 points (0.752 → 0.802)** and
macro AUC rose **0.892 → 0.919**. The loop's gap-closed guard then terminated the run
— total wall time 6.4 minutes on my Mac.

Two supporting observations from the runs leading up to this one:
- **The stop guards work standalone:** with a shorter (12-epoch) baseline whose gap
  happened to land at 0.056, the loop stopped at iteration 0 with "gap closed" without
  spending a planner call.
- **Attribution needs seeding:** an earlier unseeded run showed val swings of ±6 pts
  with an *identical* config across iterations — which is also how I caught (and
  fixed) a bug where the planner's grouped `updated_params` weren't being applied.
  Training is now seeded and the loop prints the effective config per iteration.

## Caveats (prototype, not a final design)

- One remedial iteration on a subset — a demo of the mechanism, not a benchmark;
  4-iteration budget, single seed, and the ReAct strategy is deliberately swappable
  (tree search is the open alternative to explore).
- The 25-epoch unregularized baseline was chosen to *exhibit* the overfitting we saw
  in the full-data run (train 99.7% / val 83.7%); absolute numbers here are on the
  smaller subset and are not comparable to that run.
