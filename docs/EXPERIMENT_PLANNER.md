# Experiment Planner — ReAct prototype (working)

**Last updated: 2026-07-20** · **Status: working prototype on branch `feat/experiment-planner` — for discussion, not final design**

The Experiment Planner is the reasoning core of the AI-Scientist loop (see
[WORKFLOW_DESIGN.md](./WORKFLOW_DESIGN.md), agent #6). The earlier version of this
document proposed two strategies — a ReAct loop vs agentic tree search. **This branch
implements the ReAct option end to end** and runs it for real; tree search remains an
open alternative to explore together (the interface below was kept swappable exactly
for that).

## The loop (implemented)

```
initial (arch, config)
   └─> TRAIN (torch, real) ─> INFER (held-out val) ─> ANALYSIS (metrics)
            └────────────── ExperimentRun appended to ExperimentState ─┐
   ┌───────────────────────────────────────────────────────────────────┘
   OBSERVE:  planner sees COMPACT state (metrics only — prediction arrays excluded)
   REASON:   gpt-5.2 diagnoses (e.g. train>>val = overfitting) and picks ONE change
   ACT:      typed PlannerDecision.updated_params applied to arch/config (re-validated)
   ... repeat until stop
```

- **Code:** `agents/_experiment_loop.py` (loop + guards), `agents/_experiment_planner.py`
  (LLM planner + `ReActPlannerStrategy`), `prompts/_planner.py` (metrics-only prompt).
- **Strategy is swappable:** the loop takes any `PlannerStrategy` (`async decide(state)
  -> PlannerDecision`). A tree-search strategy (expand several candidate changes,
  evaluate, prune) can drop in without touching the loop — that's the design space to
  explore with Pranath.
- **Inference is not a decision point** — it only scores the trained model.

## Knobs the planner can turn (typed, re-validated on application)

- architecture: `name` (`resnet18` | `resnet34`)
- training: `learning_rate`, `batch_size`, `epochs`, `weight_decay`, `lr_scheduler`,
  `dropout`, `augment` (flip/rot90), `early_stop_patience` (10% train holdout,
  best-weights restore)

The regularization knobs are implemented for real in `tools/_torch_backends.py`.

## Stop conditions

Planner-decided (`report`/`stop`) **plus** code-side guards recorded as
`[loop guard]` decisions: gap ≤ threshold (default 0.06), no val improvement for 2
iterations, or the iteration budget (default 4).

## Bias control (the research point)

The planner prompt contains **no domain preferences** — no lensing-specific or
equivariance hints, no "known best" architectures. It is instructed to reason only
from the observed metrics and to prefer the smallest intervention. Whether the loop
converges toward the team's expectations on its own is exactly the question the
experiments should answer.

## Scaling note

`ExperimentPlanner.decide()` serializes a **compact** state view — prediction arrays
(`predictions`/`probabilities`/`true_labels`) are excluded; the planner reasons over
aggregate metrics. Full arrays stay in `ExperimentState` (code-side, authoritative).

## Combined design (added 2026-07-25, per Michael)

The planner now has TWO cooperating stages:

1. **Tree-based architecture search** (`agents/_architecture_search.py`): LLM
   generates ~10 buildable candidates -> LLM-as-judge takes the top-4 -> short REAL
   training ranks them -> code-side pruning keeps the best (one optional refinement
   round of variants). Candidates are constrained to the torch backend's buildable
   space (`resnet`/`cnn` with typed depths/widths) and validated code-side.
2. **Closed-loop hyperparameter tuning** (this document's ReAct loop) then tunes
   the winning architecture.

Orchestrated end to end by `scripts/run_ai_scientist.py` (all knobs configurable,
every phase timed). LLM default is gpt-5.6-luna via the Responses API
(`dlens.config.build_llm`; gpt-5.2 switchable with DLENS_LLM).

## Results

See [EXPERIMENT_PLANNER_RESULTS.md](./EXPERIMENT_PLANNER_RESULTS.md) for the real
closed-loop run on DeepLense Model_I data (overfitting-reduction trajectory), and
[ARCHITECTURE_SEARCH_RESULTS.md](./ARCHITECTURE_SEARCH_RESULTS.md) for the combined
tree-search + tuning runs with the per-phase runtime breakdown.

## Open items (for discussion)

- Tree-search strategy as an alternative `PlannerStrategy` (cost/benefit vs ReAct).
- Routing `design_model` decisions through the ModelDesignAgent (currently the loop
  applies typed architecture deltas directly — the agent's recommender carries a
  baseline bias we deliberately keep out of the planner's loop).
- Larger decision space: LR schedules, longer budgets, dataset-size decisions
  (`simulate` action is plumbed but unused in this prototype).
