# agents/_experiment_loop.py
"""ExperimentLoop — the closed feedback loop of the AI-Scientist pipeline (prototype).

ReAct-style: each iteration OBSERVEs the latest run's metrics (train vs val accuracy,
AUC, confusion, per-class), REASONs with the planner strategy (one concrete, typed
change), then ACTs — applies the change to the architecture / training config and
re-runs train -> infer -> analysis on the real backends. Inference is never a
decision point; it only scores the trained model.

The planning strategy is deliberately swappable (``PlannerStrategy`` protocol): the
default is a ReAct LLM planner; an agentic tree search could implement the same
protocol later — that design space is intentionally left open.

Stop conditions (code-side guards, besides the planner's own report/stop):
  * train/val gap closes below ``gap_threshold`` **and** the model still works
    (see below) — a small gap alone is not convergence
  * val accuracy stops improving for ``patience`` consecutive iterations
  * ``max_iterations`` budget reached

A closed gap only counts as convergence if validation accuracy is acceptable.
An intervention can close the gap by *destroying* the model — e.g. early
stopping fires while the network is still on its initial plateau, leaving train
and val both at chance level with a tiny gap between them. That is a failed
intervention, not a converged experiment, so the loop keeps iterating and lets
the planner react to the collapse instead of reporting success.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from dlens.schemas._downstream import DatasetRef
from dlens.schemas._experiment import (
    ExperimentRun,
    ExperimentState,
    PlannerAction,
    PlannerDecision,
)
from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig
from dlens.tools._analysis import compute_analysis
from dlens.tools._inference import InferBackend
from dlens.tools._training import TrainBackend

_TERMINAL = {PlannerAction.REPORT, PlannerAction.STOP}
# Typed knobs the planner may change (validated on application).
_ARCH_KEYS = ("name", "family", "physics_informed")
_CFG_KEYS = (
    "loss", "optimizer", "learning_rate", "batch_size", "epochs", "weight_decay",
    "lr_scheduler", "dropout", "augment", "early_stop_patience",
)


@runtime_checkable
class PlannerStrategy(Protocol):
    """Strategy interface: read the experiment history, return the next decision."""

    name: str

    async def decide(self, state: ExperimentState) -> PlannerDecision:
        ...


class ExperimentLoop:
    """design-delta -> train -> infer -> analysis -> plan, until stop."""

    def __init__(
        self,
        *,
        strategy: PlannerStrategy,
        train_backend: TrainBackend,
        infer_backend: InferBackend,
    ) -> None:
        self.strategy = strategy
        self.train_backend = train_backend
        self.infer_backend = infer_backend

    async def run(
        self,
        *,
        hypothesis: str,
        train_ref: DatasetRef,
        val_ref: DatasetRef,
        architecture: ArchitectureSpec,
        training_config: TrainingConfig,
        max_iterations: int = 4,
        gap_threshold: float = 0.06,
        min_val_delta: float = 0.005,
        patience: int = 2,
        min_val_accuracy: Optional[float] = None,
        min_val_above_chance: float = 0.05,
        max_val_regression: float = 0.10,
    ) -> ExperimentState:
        state = ExperimentState(hypothesis=hypothesis, max_iterations=max_iterations)
        arch, cfg = architecture, training_config
        best_val, no_improve = -1.0, 0

        # Convergence floor for validation accuracy. Default: chance level for the
        # class count plus a margin — a classifier within `min_val_above_chance` of
        # random guessing has collapsed, not converged. Chance is the only
        # task-independent reference point available here, which is why it (and not
        # some absolute accuracy target) sets the floor; pass `min_val_accuracy` to
        # impose a stricter, task-specific bound.
        chance = 1.0 / max(val_ref.num_classes, 2)
        val_floor = (
            min_val_accuracy if min_val_accuracy is not None else chance + min_val_above_chance
        )

        for i in range(max_iterations):
            state.current_iteration = i
            print(f"\n=== iteration {i}: arch={arch.name} dropout={cfg.dropout} "
                  f"augment={cfg.augment} wd={cfg.weight_decay} epochs={cfg.epochs} "
                  f"es_patience={cfg.early_stop_patience} ===", flush=True)

            tr = self.train_backend.train(train_ref, arch, cfg)
            ir = self.infer_backend.infer(tr.weights_path, val_ref)
            ar = compute_analysis(ir, val_ref.class_names)
            run = ExperimentRun(
                iteration=i, architecture=arch, training_config=cfg,
                train_result=tr, infer_result=ir, analysis_result=ar,
            )
            state.runs.append(run)

            train_acc = float(tr.metrics.get("train_accuracy", float("nan")))
            val_acc = float(ir.accuracy if ir.accuracy is not None else float("nan"))
            gap = train_acc - val_acc
            auc = f"{ar.macro_auc:.4f}" if ar.macro_auc is not None else "n/a"
            print(f"    train={train_acc:.4f}  val={val_acc:.4f}  gap={gap:.4f}  auc={auc}",
                  flush=True)

            # --- code-side stop guards (recorded as loop-guard decisions) ---
            prev_best = best_val
            if val_acc > best_val + min_val_delta:
                best_val, no_improve = val_acc, 0
            else:
                no_improve += 1

            # A closed gap is only convergence if the model still works: reject a
            # collapse (val at/near chance) or a large regression against the best
            # val accuracy seen so far — both mean the intervention did harm.
            collapsed = val_acc < val_floor
            regressed = prev_best >= 0.0 and (prev_best - val_acc) > max_val_regression

            guard: Optional[str] = None
            if gap <= gap_threshold and not (collapsed or regressed):
                guard = f"train/val gap {gap:.3f} <= threshold {gap_threshold} — gap closed."
            elif gap <= gap_threshold:
                why = (
                    f"val {val_acc:.4f} below floor {val_floor:.3f} (chance {chance:.3f})"
                    if collapsed
                    else f"val {val_acc:.4f} regressed {prev_best - val_acc:.4f} from best "
                         f"{prev_best:.4f} (limit {max_val_regression})"
                )
                print(f"    [guard] gap {gap:.3f} <= {gap_threshold} but {why} — "
                      "failed intervention, not convergence; continuing.", flush=True)
                if no_improve >= patience:
                    guard = f"val accuracy has not improved for {patience} iterations."
            elif no_improve >= patience:
                guard = f"val accuracy has not improved for {patience} iterations."
            if i == max_iterations - 1 and guard is None:
                guard = "iteration budget reached."

            if guard is not None:
                run.planner_decision = PlannerDecision(
                    action=PlannerAction.REPORT, rationale=f"[loop guard] {guard}"
                )
                break

            # --- REASON: one concrete change from the strategy ---
            decision = await self.strategy.decide(state)
            run.planner_decision = decision
            print(f"    planner[{self.strategy.name}]: {decision.action.value} "
                  f"{decision.updated_params}\n    reasoning: {decision.rationale[:300]}",
                  flush=True)
            if decision.action in _TERMINAL:
                break

            # --- ACT: apply the typed deltas (re-validated) ---
            arch, cfg = _apply(decision.updated_params, arch, cfg)

        return state


def best_run_by_val_accuracy(state: ExperimentState) -> Optional[ExperimentRun]:
    """The iteration with the highest validation accuracy (``None`` if no runs).

    Runners should report this rather than ``state.runs[-1]``: the last iteration
    may be a failed intervention, which must not be presented as the outcome of
    the tuning loop. Ties keep the earliest iteration (simpler config wins).
    """
    if not state.runs:
        return None

    def _val(run: ExperimentRun) -> float:
        acc = run.infer_result.accuracy
        return float(acc) if acc is not None else float("-inf")

    return max(state.runs, key=_val)


def _apply(
    params: dict, arch: ArchitectureSpec, cfg: TrainingConfig
) -> tuple[ArchitectureSpec, TrainingConfig]:
    """Apply planner deltas with full re-validation; unknown keys are ignored.

    Accepts both flat params ({"dropout": 0.3}) and the grouped form the LLM may
    emit ({"training": {"dropout": 0.3}, "architecture": {"name": ...}}).
    """
    flat = dict(params or {})
    arch_updates: dict = {}
    cfg_updates: dict = {}
    for group_key, sink in (("architecture", arch_updates), ("training", cfg_updates)):
        group = flat.pop(group_key, None)
        if isinstance(group, dict):
            sink.update(group)
    arch_updates.update(flat)
    cfg_updates.update(flat)
    arch_updates = {k: v for k, v in arch_updates.items() if k in _ARCH_KEYS}
    cfg_updates = {k: v for k, v in cfg_updates.items() if k in _CFG_KEYS}
    if arch_updates:
        arch = ArchitectureSpec(**{**arch.model_dump(), **arch_updates})
    if cfg_updates:
        cfg = TrainingConfig(**{**cfg.model_dump(), **cfg_updates})
    return arch, cfg
