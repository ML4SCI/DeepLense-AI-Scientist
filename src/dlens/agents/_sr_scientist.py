"""Structured LLM suggestions for SR search and tuning.

Suggestions never override code-side buildability, metric ranking, or budgets.
"""

from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent, OutputSchema
from dlens.config import build_model_from_env
from dlens.sr_losses import ALL_LOSSES, allowed_losses, default_loss
from dlens.sr_models import FAMILIES

Family = Literal[*FAMILIES]
Knob = Literal["learning_rate", "loss_id", "loss_l1_weight", "augment", "width"]
LossID = Literal[*ALL_LOSSES]



def _sr_model_settings() -> dict[str, int] | None:
    """Apply an optional output cap to every SR LLM call."""
    raw = os.getenv("DLENS_MODEL_MAX_TOKENS")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("DLENS_MODEL_MAX_TOKENS must be a positive integer") from exc
    if value < 1:
        raise ValueError("DLENS_MODEL_MAX_TOKENS must be a positive integer")
    return {"max_tokens": value}


class SRCandidate(BaseModel):
    family: Family
    width: int = Field(ge=8, le=128)
    depth: int = Field(ge=1, le=6)
    loss_id: LossID
    screening_epochs: int = Field(ge=1, le=100, description="Epoch budget before promotion")
    reasoning: str = Field(description="Why this specific architecture belongs in the batch")


class SRBatchCandidates(OutputSchema):
    candidates: list[SRCandidate] = Field(min_length=5, max_length=5)
    reasoning: str = Field(description="Why this batch follows from the previous round's winner")


class SRPlannerDecision(OutputSchema):
    knob: Knob
    value: float | int | bool | str
    reasoning: str = Field(description="Brief justification using the previous round and current best")


class SREpochBudget(OutputSchema):
    epochs: int = Field(ge=1, le=100)
    reasoning: str = Field(description="Why this architecture needs this training budget")


class SRJudgeDecision(OutputSchema):
    selected_ids: list[str] = Field(description="Ordered IDs selected from the eligible candidates")
    candidate_reasons: dict[str, str] = Field(
        description="A concise evidence-based assessment for every eligible candidate ID")
    reasoning: str = Field(description="Overall selection rationale based on measured evidence")


def planner_context(history: list[dict], config: dict, best: dict,
                    baseline: dict) -> dict:
    """Expose the last measured round and the checkpoint used next."""
    def summary(run: dict) -> dict:
        losses = run.get("train_loss") or []
        return {"id": run["id"], "stage": run.get("stage", "prior"),
                "architecture": run["architecture"],
                "psnr_db": run["psnr_db"], "ssim": run["ssim"],
                "parameters": run["parameters"],
                "learning_rate": run["learning_rate"],
                "loss_l1_weight": run["loss_l1_weight"],
                "loss_id": run.get("loss_id", default_loss(run["architecture"]["family"])),
                "augment": run["augment"], "epochs": run["epochs"],
                "train_loss_last": losses[-1] if losses else None,
                "resumed_from": run.get("resumed_from")}

    if not history:
        raise ValueError("planner requires at least one completed run")
    compact = [summary(run) for run in history]
    return {"task": "Choose one change for the next tuning run using the previous round's result",
            "selection_baseline": baseline,
            "allowed_losses": allowed_losses(config["family"]),
            "previous_round": compact[-1],
            "current_best": summary(best),
            "current_config": config,
            "measured_history": compact}


def proposal_context(history: list[dict], best: dict | None, baseline: dict,
                     allowed_families: list[str], round_number: int,
                     rejected: list[str], winner_selected: bool = True,
                     judge_feedback: dict | None = None,
                     screening_epoch_bounds: tuple[int, int] = (1, 10)) -> dict:
    """Give the generator one completed selection before a whole batch."""
    compact = [{"id": run["id"], "architecture": run["architecture"],
                "psnr_db": run["psnr_db"], "ssim": run["ssim"],
                "loss_id": run.get("loss_id", default_loss(run["architecture"]["family"])),
                "parameters": run["parameters"]} for run in history]
    winner = ({"id": best["id"], "architecture": best["architecture"],
               "psnr_db": best["psnr_db"], "ssim": best["ssim"]}
              if best is not None else None)
    measured_architectures = list({
        (run["architecture"]["family"], run["architecture"]["width"],
         run["architecture"]["depth"], run.get("loss_id", default_loss(run["architecture"]["family"]))):
        {"architecture": run["architecture"],
         "loss_id": run.get("loss_id", default_loss(run["architecture"]["family"]))}
        for run in history
    }.values())
    return {"task": "Propose 5 different x2 super-resolution experiments together for one batch round",
            "round_number": round_number,
            "previous_cycle_winner": winner if winner_selected else None,
            "previous_cycle_leader": winner,
            "previous_judge_feedback": judge_feedback,
            "latest_measured_run": compact[-1] if compact else None,
            "current_best": winner,
            "selection_baseline": baseline,
            "screening_epoch_bounds": {"min": screening_epoch_bounds[0],
                                       "max": screening_epoch_bounds[1]},
            "allowed_families": allowed_families,
            "allowed_losses_by_family": {family: allowed_losses(family) for family in allowed_families},
            "already_measured": measured_architectures,
            "rejected_proposals": rejected,
            "measured_history": compact[-72:],
            "total_measured_runs": len(compact)}


class SRGenerator(DLensBaseAgent):
    def __init__(self) -> None:
        super().__init__(
            config=BaseAgentConfig(
                name="SRArchitectureGenerator",
                description="Propose a batch of 5 buildable x2 architectures after each completed selection.",
                custom_system_prompt=(
                    "Propose exactly 5 distinct candidates together from allowed_families. "
                    "Do not use outcomes from any candidate in the same batch. Use previous_cycle_winner "
                    "as primary feedback; if null, use previous_cycle_leader. In the first round "
                    "both may be null. Use previous_judge_feedback, selection_baseline, "
                    "and measured_history; rejected models are negative evidence, "
                    "not winners. "
                    "Give a specific evidence-based reasoning for every candidate. "
                    "Choose loss_id from allowed_losses_by_family for every candidate. "
                    "Choose screening_epochs separately for every candidate, within "
                    "screening_epoch_bounds, based on architecture and compute needs. "
                    "Vary family, width, depth, and loss. Do not repeat an already_measured "
                    "or rejected configuration. "
                    "Every candidate predicts an HR image, not class labels. Do not use test data."
                ), model=build_model_from_env()), output_type=SRBatchCandidates,
            model_settings=_sr_model_settings())

    async def propose_batch(self, history: list[dict], best: dict | None,
                            baseline: dict, allowed_families: list[str],
                            round_number: int, winner_selected: bool = True,
                            judge_feedback: dict | None = None,
                            screening_epoch_bounds: tuple[int, int] = (1, 10)) -> SRBatchCandidates:
        return (await self.arun(json.dumps(proposal_context(
            history, best, baseline, allowed_families, round_number,
            [], winner_selected, judge_feedback, screening_epoch_bounds)))).output


class SREpochPlanner(DLensBaseAgent):
    def __init__(self) -> None:
        super().__init__(
            config=BaseAgentConfig(
                name="SREpochPlanner",
                description="Choose an architecture-specific promotion or tuning epoch budget.",
                custom_system_prompt=(
                    "Choose an integer epochs value within epoch_bounds, inclusive. "
                    "Use the architecture, prior validation scores, training loss, and "
                    "optimizer steps to justify the compute budget. Promotion continues "
                    "a screened checkpoint; tuning continues the best checkpoint. "
                    "Do not claim a model has passed the baseline unless both measured "
                    "metrics exceed it. Do not use test data."
                ), model=build_model_from_env()), output_type=SREpochBudget,
            model_settings=_sr_model_settings())

    async def decide(self, stage: str, candidate: dict, baseline: dict,
                     history: list[dict], epoch_bounds: tuple[int, int]) -> SREpochBudget:
        evidence = {"task": "Choose additional training epochs", "stage": stage,
                    "epoch_bounds": {"min": epoch_bounds[0], "max": epoch_bounds[1]},
                    "candidate": {key: candidate.get(key) for key in
                                  ("id", "architecture", "loss_id", "psnr_db", "ssim",
                                   "epochs", "epochs_this_stage", "optimizer_steps",
                                   "train_loss", "learning_rate")},
                    "selection_baseline": baseline,
                    "recent_history": [{key: run.get(key) for key in
                                        ("id", "architecture", "psnr_db", "ssim",
                                         "optimizer_steps", "epochs")}
                                       for run in history[-12:]]}
        return (await self.arun(json.dumps(evidence))).output


class SRPlanner(DLensBaseAgent):
    def __init__(self) -> None:
        super().__init__(
            config=BaseAgentConfig(
                name="SRExperimentPlanner",
                description="Suggest one typed tuning change from compact validation metrics.",
                custom_system_prompt=(
                    "Change exactly one of learning_rate, loss_id, loss_l1_weight, augment, "
                    "or width. Epochs are chosen separately by the epoch planner. "
                    "The previous_round is the most recent completed "
                    "experiment; current_best is the checkpoint the next run starts from. "
                    "Use both and the measured history before proposing the next change. "
                    "Explain the change using the previous round's PSNR and SSIM. "
                    "The proposed value must differ from the current value. "
                    "Choose a loss_id compatible with the current family. "
                    "Respect both PSNR and SSIM. Never inspect test data."
                ), model=build_model_from_env()), output_type=SRPlannerDecision,
            model_settings=_sr_model_settings())

    async def decide(self, history: list[dict], config: dict,
                     best: dict, baseline: dict) -> SRPlannerDecision:
        return (await self.arun(json.dumps(planner_context(
            history, config, best, baseline)))).output


class SRJudge(DLensBaseAgent):
    def __init__(self) -> None:
        super().__init__(
            config=BaseAgentConfig(
                name="SRValidationJudge",
                description="Select SR models from measured validation evidence after screening or tuning.",
                custom_system_prompt=(
                    "You are the decision maker for the eligible models in this stage. "
                    "Select exactly requested_count distinct IDs from eligible_candidates, "
                    "or all eligible candidates when fewer are available. "
                    "At screening, balance PSNR, SSIM, image-level consistency, training "
                    "behavior, compute, and useful architectural diversity. At final "
                    "selection, choose one best validated checkpoint for this round. "
                    "Follow selection_rule in the evidence: screening may promote "
                    "underbaseline models for additional training when enabled, but final "
                    "winners must strictly exceed selection_baseline on both metrics. "
                    "Never select rejected candidates or invent IDs or measurements. "
                    "Explain every eligible candidate in candidate_reasons, including "
                    "why an unselected model lost. Give an explicit, concise rationale; "
                    "do not provide private chain-of-thought. Do not use test data."
                ), model=build_model_from_env()), output_type=SRJudgeDecision,
            model_settings=_sr_model_settings())

    async def decide(self, evidence: dict) -> SRJudgeDecision:
        return (await self.arun(json.dumps(evidence))).output
