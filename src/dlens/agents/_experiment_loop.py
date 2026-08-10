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
  * train/val gap closes below ``gap_threshold``
  * val accuracy stops improving for ``patience`` consecutive iterations
  * ``max_iterations`` budget reached
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
    ) -> ExperimentState:
        state = ExperimentState(hypothesis=hypothesis, max_iterations=max_iterations)
        arch, cfg = architecture, training_config
        best_val, no_improve = -1.0, 0

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
            guard: Optional[str] = None
            if gap <= gap_threshold:
                guard = f"train/val gap {gap:.3f} <= threshold {gap_threshold} — gap closed."
            elif val_acc > best_val + min_val_delta:
                best_val, no_improve = val_acc, 0
            else:
                no_improve += 1
                if no_improve >= patience:
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
