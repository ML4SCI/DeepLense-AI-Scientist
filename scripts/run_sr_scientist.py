"""Run closed 5-model SR proposal batches with three-model tuning and selection.

Only train/ and val/ are opened. Test evaluation is a separate final action.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dlens.config import ModelSettings
from dlens.data.sr_dataset import SRPairDataset
from dlens.schemas._super_resolution import SRDatasetRef
from dlens.sr_baseline import load_baseline
from dlens.sr_losses import allowed_losses, default_loss, validate_loss
from dlens.sr_metrics import METRIC_POLICY, evaluate_batches, matches_metric_policy, summarize_scores
from dlens.sr_models import FAMILIES, SRArchitecture, build_sr_model, canonical_sr_family
from dlens.sr_selection import passes_baseline, rank_runs, rank_with_fallback, ranking_details
from dlens.sr_training import SRTrainer
from dlens.sr_visuals import fixed_sample_ids, render_comparison

PROPOSALS_PER_ROUND = 5


def _ref(root: Path, lr_size: int) -> SRDatasetRef:
    return SRDatasetRef(root=str(root), lr_shape=(lr_size, lr_size),
                        hr_shape=(2 * lr_size, 2 * lr_size), scale_factor=2)


def _evaluate(model: nn.Module | None, loader: DataLoader, device: torch.device,
              max_pairs: int | None = None) -> tuple[list[dict], dict]:
    if model is not None:
        model.eval()

    def predict(lr: torch.Tensor, _hr: torch.Tensor, _offset: int) -> torch.Tensor:
        return model(lr)

    rows = [row for _, _, _, batch in evaluate_batches(
        loader, device, predict if model is not None else None, max_pairs=max_pairs,
    ) for row in batch]
    return rows, summarize_scores(rows)


def _run(spec: SRArchitecture, *, train_set: SRPairDataset, val_loader: DataLoader,
         device: torch.device, output: Path, epochs: int, batch_size: int,
         learning_rate: float, loss_l1_weight: float, augment: bool,
         max_train_batches: int | None, max_val_pairs: int | None, seed: int,
         run_id: str, stage: str = "search", resume_from: dict | None = None,
         loss_id: str | None = None) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_set.augment = augment
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    model = build_sr_model(spec).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    loss_id = loss_id or default_loss(spec.family)
    trainer = SRTrainer(model, spec, learning_rate, device, loss_id=loss_id)
    if resume_from is not None:
        if resume_from["architecture"] != asdict(spec):
            raise ValueError("continuation requires the same architecture")
        saved = torch.load(resume_from["checkpoint"], map_location=device, weights_only=True)
        trainer.load_training_state(saved)
        trainer.set_learning_rate(learning_rate)
        torch.set_rng_state(saved["rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in saved:
            # map_location=device moves these ByteTensors to CUDA, but PyTorch's
            # RNG restoration API requires CPU ByteTensors.
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
    losses = []
    stage_steps = 0
    stage_samples = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = count = 0
        steps_per_epoch = min(len(train_loader), max_train_batches or len(train_loader))
        for lr, hr, _ in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            total += trainer.step(lr, hr, loss_l1_weight)
            count += 1
            stage_steps += 1
            stage_samples += lr.shape[0]
            if count == 1 or count % 10 == 0 or count == steps_per_epoch:
                print(f"{run_id}: epoch {epoch + 1}/{epochs} batch "
                      f"{count}/{steps_per_epoch} loss={total / count:.6f}", flush=True)
            if max_train_batches is not None and count >= max_train_batches:
                break
        losses.append(total / count)
        print(f"{run_id}: epoch {epoch + 1}/{epochs} loss={losses[-1]:.6f}", flush=True)
    train_seconds = time.perf_counter() - started
    rows, stats = _evaluate(model, val_loader, device, max_val_pairs)
    folder = output / run_id
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "val_per_image.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("sample_id", "psnr_db", "ssim"))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = folder / "model.pt"
    total_epochs = epochs + (resume_from["epochs"] if resume_from else 0)
    total_steps = stage_steps + (resume_from.get("optimizer_steps", 0) if resume_from else 0)
    training_state = trainer.training_state()
    torch.save({"model": model.state_dict(), "architecture": asdict(spec),
                "epochs": total_epochs, "learning_rate": learning_rate,
                "loss_l1_weight": loss_l1_weight, "augment": augment,
                "loss_id": loss_id,
                "seed": seed, "objective": trainer.objective,
                "loss_config": trainer.loss_config,
                "rng_state": torch.get_rng_state(),
                **({"cuda_rng_state": torch.cuda.get_rng_state_all()} if device.type == "cuda" else {}),
                **training_state}, checkpoint)
    result = {"id": run_id, "architecture": asdict(spec), "parameters": parameters,
            "stage": stage, "resumed_from": resume_from["id"] if resume_from else None,
            "epochs": total_epochs, "epochs_this_stage": epochs,
            "optimizer_steps": total_steps, "steps_this_stage": stage_steps,
            "samples_this_stage": stage_samples,
            "train_seconds": train_seconds,
            "cumulative_train_seconds": train_seconds + (resume_from.get("cumulative_train_seconds", 0) if resume_from else 0),
            "learning_rate": learning_rate,
            "loss_l1_weight": loss_l1_weight, "augment": augment,
            "loss_id": loss_id,
            "objective": trainer.objective,
            "loss_config": trainer.loss_config,
            "train_loss": losses, "validation_pairs": len(rows),
            "psnr_db": stats["psnr_db"]["mean"], "ssim": stats["ssim"]["mean"],
            "validation": stats, "checkpoint": str(checkpoint),
            "per_image_csv": str(folder / "val_per_image.csv")}
    (folder / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _best_per_family(runs: list[dict], baseline: dict) -> list[dict]:
    """Compare each family's best validated checkpoint exactly once."""
    families = {}
    for run in runs:
        families.setdefault(run["architecture"]["family"], []).append(run)
    best = [ranked[0] for family_runs in families.values()
            if (ranked := rank_with_fallback(family_runs, baseline))]
    return rank_with_fallback(best, baseline)


def _next_architecture(runs: list[dict], baseline: dict,
                       allowed_families: list[str], image_size: int,
                       selected_winner: dict | None = None,
                       additional_seen: set[tuple[str, int, int]] | None = None,
                       default_width: int = 32, default_depth: int = 2) -> SRArchitecture:
    """Deterministic feedback planner when no LLM is configured."""
    seen = {(run["architecture"]["family"], run["architecture"]["width"],
             run["architecture"]["depth"]) for run in runs}
    seen.update(additional_seen or set())
    ranked = _best_per_family(runs, baseline) if runs else []
    sources = ([selected_winner] if selected_winner is not None else []) + [
        run for run in ranked if selected_winner is None or run["id"] != selected_winner["id"]]
    for source in sources:
        family = source["architecture"]["family"]
        if family not in allowed_families:
            continue
        width, depth = source["architecture"]["width"], source["architecture"]["depth"]
        widths = dict.fromkeys((width, min(width * 2, 128), max(width // 2, 8),
                                min(width + 16, 128), min(width + 8, 128),
                                max(width - 8, 8), 48, 96, 128, 24, 16, 8))
        depths = dict.fromkeys((min(depth + 1, 6), depth, min(depth + 2, 6),
                                max(depth - 1, 1), 1, 2, 3, 4, 5, 6))
        for candidate_depth in depths:
            for candidate_width in widths:
                signature = (family, candidate_width, candidate_depth)
                if signature not in seen:
                    return SRArchitecture(*signature, image_size=image_size)
    for family in allowed_families:
        for depth in dict.fromkeys((default_depth, 1, 2, 3, 4, 5, 6)):
            for width in dict.fromkeys((default_width, 16, 24, 32, 48, 64, 96, 128, 8)):
                signature = (family, width, depth)
                if signature not in seen:
                    return SRArchitecture(*signature, image_size=image_size)
    raise ValueError("no unmeasured architecture remains in the allowed families")


def _proposal_batch(history: list[dict], winner: dict | None, baseline: dict,
                    families: list[str], round_number: int, image_size: int,
                    width: int, depth: int, use_llm: bool,
                    judge_feedback: dict | None = None,
                    screening_epoch_bounds: tuple[int, int] = (1, 10)) -> tuple[list[SRArchitecture], dict]:
    """Plan all 5 unique architectures before any model in the batch is trained."""
    count = PROPOSALS_PER_ROUND
    measured = {(run["architecture"]["family"], run["architecture"]["width"],
                 run["architecture"]["depth"],
                 run.get("loss_id", default_loss(run["architecture"]["family"])))
                for run in history}
    chosen: list[SRArchitecture] = []
    reserved: set[tuple[str, int, int, str]] = set()
    log = {"source": "deterministic",
           "reasoning": "Choose diverse unmeasured configurations using the prior round winner and measured history.",
           "planner_inputs": {
               "previous_winner_id": winner["id"] if winner else None,
               "previous_winner_metrics": ({key: winner[key] for key in ("psnr_db", "ssim")}
                                           if winner else None),
               "previous_judge_feedback": judge_feedback,
               "baseline": baseline, "measured_run_count": len(history),
               "recent_measurements": [
                   {"id": run["id"], "architecture": run["architecture"],
                    "psnr_db": run["psnr_db"], "ssim": run["ssim"]}
                   for run in history[-72:]],
           },
           "candidate_reasons": [], "loss_ids": [], "screening_epochs": [], "rejected": []}
    if use_llm:
        from dlens.agents._sr_scientist import SRGenerator

        proposal = asyncio.run(SRGenerator().propose_batch(
            history, winner, baseline, families, round_number,
            winner_selected=winner is not None and
            passes_baseline(winner, baseline),
            judge_feedback=judge_feedback,
            screening_epoch_bounds=screening_epoch_bounds))
        log["source"] = "llm"
        log["reasoning"] = proposal.reasoning
        for candidate in proposal.candidates:
            signature = (candidate.family, candidate.width, candidate.depth)
            experiment = (*signature, candidate.loss_id)
            epochs = getattr(candidate, "screening_epochs", screening_epoch_bounds[1])
            if (candidate.family not in families or experiment in measured or
                    experiment in reserved or candidate.loss_id not in allowed_losses(candidate.family) or
                    type(epochs) is not int or not screening_epoch_bounds[0] <= epochs <= screening_epoch_bounds[1]):
                log["rejected"].append({"architecture": signature,
                                        "loss_id": candidate.loss_id,
                                        "reason": "disallowed, duplicate, or invalid screening epoch budget"})
                continue
            chosen.append(SRArchitecture(*signature, image_size=image_size))
            log["candidate_reasons"].append(candidate.reasoning)
            log["loss_ids"].append(candidate.loss_id)
            log["screening_epochs"].append(epochs)
            reserved.add(experiment)
    rotation = (round_number - 1) % len(families)
    ordered = families[rotation:] + families[:rotation]
    if winner is not None and winner["architecture"]["family"] in ordered:
        family = winner["architecture"]["family"]
        ordered = [family] + [item for item in ordered if item != family]
    while len(chosen) < count:
        family = ordered[len(chosen) % len(ordered)]
        spec = _next_architecture(
            history, baseline, [family], image_size,
            winner if winner is not None and winner["architecture"]["family"] == family else None,
            {item[:3] for item in reserved}, width, depth)
        chosen.append(spec)
        if winner is None:
            reason = (f"Initial batch exploration of {family}; no previous round winner exists. "
                      f"Configuration width={spec.width}, depth={spec.depth} is unmeasured.")
        elif family == winner["architecture"]["family"]:
            reason = (f"Explore a new width/depth variant of previous winner {winner['id']} "
                      f"({winner['psnr_db']:.4f} dB PSNR, {winner['ssim']:.6f} SSIM).")
        else:
            reason = (f"Compare unmeasured {family} against previous winner {winner['id']} "
                      f"({winner['psnr_db']:.4f} dB PSNR, {winner['ssim']:.6f} SSIM).")
        log["candidate_reasons"].append(reason)
        choices = tuple(loss for loss in allowed_losses(family)
                        if loss != "l1_mse")
        log["loss_ids"].append(choices[(round_number + len(chosen) - 2) % len(choices)])
        log["screening_epochs"].append(screening_epoch_bounds[1])
        reserved.add((spec.family, spec.width, spec.depth, log["loss_ids"][-1]))
    return chosen, log


def _epoch_budget(stage: str, candidate: dict, baseline: dict,
                  history: list[dict], bounds: tuple[int, int], use_llm: bool) -> dict:
    """Choose and validate additional epochs for one architecture and stage."""
    lower, upper = bounds
    if not 1 <= lower <= upper <= 100:
        raise ValueError(f"invalid {stage} epoch bounds: {bounds}")
    if not use_llm:
        return {"source": "fixed", "epochs": upper,
                "reasoning": "Use the configured epoch budget without an LLM."}
    from dlens.agents._sr_scientist import SREpochPlanner

    answer = asyncio.run(SREpochPlanner().decide(stage, candidate, baseline, history, bounds))
    if type(answer.epochs) is not int or not lower <= answer.epochs <= upper:
        raise ValueError(f"{stage} epoch planner returned {answer.epochs}; expected [{lower}, {upper}]")
    if not answer.reasoning.strip():
        raise ValueError(f"{stage} epoch planner gave no rationale")
    return {"source": "llm", "epochs": answer.epochs, "reasoning": answer.reasoning}


def _tuning_config(winner: dict, default_epochs: int) -> dict:
    """Start each proposal from the current best checkpoint's settings."""
    architecture = winner["architecture"]
    return {"learning_rate": winner["learning_rate"],
            "loss_l1_weight": winner["loss_l1_weight"],
            "loss_id": winner.get("loss_id", default_loss(architecture["family"])),
            "augment": winner["augment"],
            "epochs": (winner["epochs_this_stage"]
                       if winner.get("stage") == "tuning" else default_epochs),
            "family": architecture["family"], "width": architecture["width"],
            "depth": architecture["depth"]}


def _next_config(config: dict, latest: dict, prior: dict, iteration: int) -> dict:
    """Feedback-driven, one-change-at-a-time SR planner with typed knobs."""
    config = dict(config)
    if all(latest[k] < prior[k] for k in ("psnr_db", "ssim")):
        config["learning_rate"] *= 0.5
    else:
        knobs = (("augment", "learning_rate", "width")
                 if config.get("family") == "realesrgan" else
                 ("loss_id", "augment", "learning_rate", "width"))
        knob = knobs[(iteration - 1) % len(knobs)]
        if knob == "loss_id":
            choices = tuple(loss for loss in allowed_losses(config["family"])
                            if loss != "l1_mse")
            config[knob] = choices[(choices.index(config[knob]) + 1) % len(choices)] if config[knob] in choices else choices[0]
        elif knob == "augment":
            config[knob] = not config[knob]
        elif knob == "learning_rate":
            config[knob] *= 0.5
        else:
            config[knob] = min(config[knob] * 2, 128)
    return config


def _apply_llm_change(config: dict, decision) -> dict:
    """Validate the typed proposal before it can affect an experiment."""
    result = dict(config)
    key, value = decision.knob, decision.value
    if result.get("family") == "sr3" and key == "loss_l1_weight":
        raise ValueError("loss_l1_weight does not apply to diffusion noise prediction")
    if key == "augment":
        if type(value) is not bool:
            raise ValueError("augment must be boolean")
    elif key == "learning_rate":
        if type(value) not in (int, float) or not 1e-6 <= value <= 1e-2:
            raise ValueError("learning_rate must be in [1e-6, 1e-2]")
    elif key == "loss_l1_weight":
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError("loss_l1_weight must be in [0, 1]")
        if result.get("loss_id") != "l1_mse":
            raise ValueError("loss_l1_weight only applies to l1_mse")
    elif key == "loss_id":
        if type(value) is not str:
            raise ValueError("loss_id must be a string")
        validate_loss(result["family"], value)
    elif key == "epochs":
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError("epochs must be an integer in [1, 100]")
    elif key == "width":
        if type(value) is not int or not 8 <= value <= 128:
            raise ValueError("width must be an integer in [8, 128]")
    else:
        raise ValueError(f"unsupported knob: {key}")
    if result[key] == value:
        raise ValueError(f"planner proposed no change: {key}={value}")
    result[key] = value
    return result


def _load_run_record(path: Path) -> dict:
    record = json.loads(path.read_text())

    def migrate(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("family"), str):
                value["family"] = canonical_sr_family(value["family"])
            if isinstance(value.get("families"), list):
                value["families"] = [canonical_sr_family(name) for name in value["families"]]
            if isinstance(value.get("family_best_ids"), dict):
                value["family_best_ids"] = {
                    canonical_sr_family(name): run_id for name, run_id in value["family_best_ids"].items()
                }
            for child in value.values():
                migrate(child)
        elif isinstance(value, list):
            for child in value:
                migrate(child)

    migrate(record)
    return record


def _load_prior_history(path: Path | None, baseline: dict,
                        baseline_pairs: int, baseline_name: str = "Bicubic") -> list[dict]:
    """Use prior results as LLM context only, after validation-policy checks."""
    if path is None:
        return []
    record = _load_run_record(path)
    policy = record.get("metric_policy", {})
    if not matches_metric_policy(policy) or record.get("baseline_pairs") != baseline_pairs:
        raise ValueError("prior run used different validation count or metric settings")
    if record.get("baseline_name", "Bicubic") != baseline_name:
        raise ValueError("prior run used a different baseline model")
    if any(abs(record["baseline"][key] - baseline[key]) > 1e-6
           for key in ("psnr_db", "ssim")):
        raise ValueError("prior run used different validation pairs or baseline")
    return record["runs"]


def _promotion_sources(screened_runs: list[dict], baseline: dict,
                       top_k: int) -> list[dict]:
    """Reject failed baseline checks before choosing promotion candidates."""
    return rank_runs(screened_runs, baseline)[:top_k]


def _judge_evidence(run: dict) -> dict:
    """Summarize the measured record and image-level validation distribution."""
    scores = {"psnr_db": [], "ssim": []}
    if run.get("per_image_csv"):
        with Path(run["per_image_csv"]).open(newline="") as file:
            for row in csv.DictReader(file):
                for metric, values in scores.items():
                    values.append(float(row[metric]))
    distribution = {}
    for metric, values in scores.items():
        ordered = sorted(values)
        if run.get("per_image_csv") and not ordered:
            raise ValueError(f"empty validation CSV for {run['id']}")
        if ordered:
            distribution[metric] = {
                "min": ordered[0], "median": ordered[len(ordered) // 2],
                "p10": ordered[int(0.1 * (len(ordered) - 1))],
                "p90": ordered[int(0.9 * (len(ordered) - 1))], "max": ordered[-1],
            }
    return {"id": run["id"], "architecture": run["architecture"],
            "loss_id": run.get("loss_id", default_loss(run["architecture"]["family"])),
            "psnr_db": run["psnr_db"], "ssim": run["ssim"],
            "validation": run.get("validation"), "per_image_distribution": distribution,
            "validation_pairs": len(scores["psnr_db"]) or run.get("validation_pairs"),
            "parameters": run["parameters"], "epochs": run.get("epochs"),
            "train_seconds": run.get("train_seconds"),
            "train_loss": run.get("train_loss", []),
            "stage": run.get("stage"), "resumed_from": run.get("resumed_from")}


def _judge_select(candidates: list[dict], baseline: dict, requested_count: int,
                  stage: str, use_llm: bool,
                  allow_underbaseline_promotion: bool = False) -> dict:
    """Allow exploratory promotion at screening; enforce the baseline for winners."""
    exploratory = stage == "screening" and allow_underbaseline_promotion
    eligible = (rank_with_fallback(candidates, baseline) if exploratory
                else rank_runs(candidates, baseline))
    eligible_ids = {run["id"] for run in eligible}
    rejected = [{"id": run["id"], "psnr_db": run["psnr_db"],
                 "ssim": run["ssim"], "reason": "PSNR and SSIM must both strictly exceed baseline"}
                for run in candidates if run["id"] not in eligible_ids]
    evidence = {"stage": stage, "selection_baseline": baseline,
                "requested_count": min(requested_count, len(eligible)),
                "eligible_candidates": [_judge_evidence(run) for run in eligible],
                "rejected_candidates": rejected,
                "selection_rule": (
                    "Screening selects promising models for more training; only final winners must "
                    "strictly exceed both baseline means." if exploratory else
                    "Every selected model must strictly exceed both baseline means.")}
    if not eligible:
        return {"source": "baseline_gate", "inputs": evidence, "selected_ids": [],
                "candidate_reasons": {},
                "reasoning": "All candidates failed at least one baseline metric; no model proceeds."}
    if use_llm:
        from dlens.agents._sr_scientist import SRJudge

        answer = asyncio.run(SRJudge().decide(evidence))
        selected_ids = answer.selected_ids
        reasons = answer.candidate_reasons
        reasoning = answer.reasoning
        source = "llm"
    else:
        selected_ids = [run["id"] for run in eligible[:requested_count]]
        reasons = {run["id"]: (
            "Selected for additional training by validation rank." if run["id"] in selected_ids
            else "Outside the requested promotion rank cutoff.") for run in eligible}
        reasoning = ("Select promising models for additional training by validation rank."
                     if exploratory else
                     "Select eligible models by PSNR and SSIM rank sum; parameter count breaks ties.")
        source = "deterministic"
    if (len(selected_ids) != min(requested_count, len(eligible)) or
            len(set(selected_ids)) != len(selected_ids) or
            not set(selected_ids) <= eligible_ids):
        raise ValueError(f"judge returned invalid or below-baseline IDs at {stage}")
    if set(reasons) != eligible_ids or any(not value.strip() for value in reasons.values()):
        raise ValueError(f"judge did not explain every eligible candidate at {stage}")
    if not reasoning.strip():
        raise ValueError(f"judge supplied no overall rationale at {stage}")
    return {"source": source, "inputs": evidence, "selected_ids": selected_ids,
            "candidate_reasons": reasons, "reasoning": reasoning}


def _judge_table(candidates: list[dict], baseline: dict,
                 selected_ids: list[str], reasons: dict[str, str] | None = None,
                 exploratory: bool = False) -> list[dict]:
    """Record the measured gate and the judge's stated selection reasons."""
    details = {item["id"]: item for item in ranking_details(candidates, baseline)}
    positions = {run["id"]: index for index, run in enumerate(
        rank_with_fallback(candidates, baseline), start=1)}
    selected = set(selected_ids)
    rows = []
    for run in candidates:
        detail = details[run["id"]]
        passed = detail["passes_baseline"]
        reason = (
            f"PSNR {'>' if run['psnr_db'] > baseline['psnr_db'] else '<='} baseline; "
            f"SSIM {'>' if run['ssim'] > baseline['ssim'] else '<='} baseline. "
            f"{'Eligible rank sum ' + str(detail['rank_sum']) if passed else 'Below baseline; may train further' if exploratory else 'Rejected immediately'}. "
            f"{'Selected' if run['id'] in selected else 'Not selected'} at this stage."
        )
        if reasons and run["id"] in reasons and (passed or exploratory):
            reason += f" Judge: {reasons[run['id']]}"
        rows.append({"id": run["id"], "architecture": run["architecture"],
                     "psnr_db": run["psnr_db"], "ssim": run["ssim"],
                     "parameters": run["parameters"],
                     "psnr_above_baseline": run["psnr_db"] - baseline["psnr_db"],
                     "ssim_above_baseline": run["ssim"] - baseline["ssim"],
                     "passes_baseline": passed,
                     "metric_ranks": detail["metric_ranks"],
                     "rank_sum": detail["rank_sum"],
                     "selection_order": positions[run["id"]],
                     "selected": run["id"] in selected, "reason": reason})
    return rows


def _complete_selection_cycle(
    runs: list[dict], promotion_sources: list[dict], *, cycle_number: int,
    train_set: SRPairDataset, val_loader: DataLoader, device: torch.device,
    args: argparse.Namespace, baseline: dict, prior_runs: list[dict], llm_log: dict,
    run_trial=None, decision_store: dict | None = None, persist=None,
    final_judge=None, epoch_decisions: dict | None = None,
) -> dict:
    """Promote and tune the selected screened models before choosing a round winner."""
    promoted_ids = []
    tuning_ids = []
    candidate_best = []
    candidate_details = []
    execute = run_trial or _run
    decisions = decision_store if decision_store is not None else {}
    budgets = epoch_decisions if epoch_decisions is not None else {}

    def planned_epochs(key: str, stage: str, candidate: dict,
                       bounds: tuple[int, int], history: list[dict]) -> int:
        if key not in budgets:
            try:
                decision = _epoch_budget(stage, candidate, baseline, history,
                                         bounds, args.use_llm)
            except ValueError as exc:
                decision = {"source": "invalid_llm_fallback", "epochs": bounds[1],
                            "reasoning": f"Rejected epoch proposal ({exc}); used configured maximum."}
            budgets[key] = {"stage": stage, "candidate_id": candidate["id"],
                            "bounds": list(bounds), **decision}
            if persist is not None:
                persist()
            print(f"{key}: {stage} epochs={decision['epochs']} "
                  f"({decision['source']}). {decision['reasoning']}", flush=True)
        return budgets[key]["epochs"]

    for source in promotion_sources:
        candidate_runs = [source]
        candidate_promoted_ids = []
        candidate_tuning_ids = []
        promotion_ceiling = args.promotion_epochs * (
            args.special_promotion_multiplier
            if source["architecture"]["family"] in {"sr3", "realesrgan"} else 1)
        promotion_epochs = planned_epochs(
            f"round_{cycle_number:02d}_promotion_budget_{source['id']}",
            "promotion", source,
            (getattr(args, "min_promotion_epochs", args.promotion_epochs), promotion_ceiling),
            prior_runs + candidate_runs)
        for label, multiplier in (("base", 1.0), ("half", 0.5)):
            spec = SRArchitecture(**source["architecture"])
            run = execute(
                spec, train_set=train_set, val_loader=val_loader, device=device,
                output=args.out_dir, epochs=promotion_epochs,
                batch_size=args.batch_size,
                learning_rate=source["learning_rate"] * multiplier,
                loss_l1_weight=source["loss_l1_weight"], augment=source["augment"],
                loss_id=source.get("loss_id", default_loss(spec.family)),
                max_train_batches=args.max_train_batches, max_val_pairs=args.max_val_pairs,
                seed=args.seed, run_id=f"round_{cycle_number:02d}_promote_{source['id']}_{label}",
                stage="promotion", resume_from=source)
            if run_trial is None:
                runs.append(run)
            candidate_runs.append(run)
            promoted_ids.append(run["id"])
            candidate_promoted_ids.append(run["id"])
        winner = rank_with_fallback(candidate_runs, baseline)[0]
        no_improve = 0
        previous = prior = winner
        for iteration in range(args.max_tuning_runs):
            run_id = f"round_{cycle_number:02d}_tune_{source['id']}_{iteration + 1:02d}"
            if run_id in decisions:
                entry = decisions[run_id]
                config = entry["config_after"]
            else:
                before = _tuning_config(winner, args.tune_epochs)
                config = before
                decision_source = "initial"
                reason = "Start this model's tuning from its best screened or promoted checkpoint."
                if iteration:
                    if args.use_llm:
                        from dlens.agents._sr_scientist import SRPlanner

                        try:
                            decision = asyncio.run(SRPlanner().decide(
                                prior_runs + candidate_runs, config, winner, baseline))
                            config = _apply_llm_change(config, decision)
                            decision_source = "llm"
                            reason = decision.reasoning
                            llm_log.setdefault("planner_decisions", []).append(
                                {"round": cycle_number, "candidate_id": source["id"],
                                 **decision.model_dump()})
                        except ValueError as exc:
                            llm_log.setdefault("rejected_planner_decisions", []).append(
                                {"round": cycle_number, "candidate_id": source["id"],
                                 "reason": str(exc)})
                            config = _next_config(config, previous, prior, iteration)
                            decision_source = "deterministic_fallback"
                            reason = f"Rejected LLM change ({exc}); used validation feedback."
                    else:
                        config = _next_config(config, previous, prior, iteration)
                        decision_source = "deterministic"
                        reason = ("Both previous validation metrics declined; halve learning rate."
                                  if all(previous[k] < prior[k] for k in ("psnr_db", "ssim"))
                                  else "Change one tuning setting using the previous validation result.")
                config = dict(config)
                config["epochs"] = planned_epochs(
                    f"round_{cycle_number:02d}_tuning_budget_{source['id']}_{iteration + 1:02d}",
                    "tuning", winner,
                    (getattr(args, "min_tune_epochs", args.tune_epochs), args.tune_epochs),
                    prior_runs + candidate_runs)
                entry = {"round": cycle_number, "candidate_id": source["id"],
                         "run_id": run_id, "source": decision_source,
                         "reasoning": reason, "previous_run_id": previous["id"],
                         "best_before_id": winner["id"],
                         "previous_scores": {key: previous[key] for key in ("psnr_db", "ssim")},
                         "best_before_scores": {key: winner[key] for key in ("psnr_db", "ssim")},
                         "config_before": before, "config_after": config}
                decisions[run_id] = entry
                if persist is not None:
                    persist()
                print(f"{run_id} planner ({decision_source}): {reason} "
                      f"settings={config}", flush=True)
            spec = SRArchitecture(winner["architecture"]["family"], width=config["width"],
                                  depth=config["depth"], image_size=args.lr_size)
            resume = winner if asdict(spec) == winner["architecture"] else None
            run = execute(spec, train_set=train_set, val_loader=val_loader, device=device,
                       output=args.out_dir, epochs=config["epochs"],
                       batch_size=args.batch_size, learning_rate=config["learning_rate"],
                       loss_l1_weight=config["loss_l1_weight"], augment=config["augment"],
                       loss_id=config["loss_id"],
                       max_train_batches=args.max_train_batches,
                       max_val_pairs=args.max_val_pairs, seed=args.seed,
                       run_id=run_id,
                       stage="tuning", resume_from=resume)
            prior, previous = previous, run
            if run_trial is None:
                runs.append(run)
            candidate_runs.append(run)
            tuning_ids.append(run["id"])
            candidate_tuning_ids.append(run["id"])
            current = rank_with_fallback(candidate_runs, baseline)[0]
            if current["id"] == run["id"]:
                winner = run
                no_improve = 0
                entry["became_candidate_best"] = True
            else:
                no_improve += 1
                entry["became_candidate_best"] = False
            entry["result_scores"] = {key: run[key] for key in ("psnr_db", "ssim")}
            entry["baseline_rejected"] = not passes_baseline(run, baseline)
            entry["resumed_from"] = run.get("resumed_from")
            if persist is not None:
                persist()
            if no_improve >= args.patience:
                break
        candidate_best.append(rank_with_fallback(candidate_runs, baseline)[0])
        candidate_details.append({"screened_id": source["id"],
                                  "promoted_ids": candidate_promoted_ids,
                                  "tuning_ids": candidate_tuning_ids,
                                  "tuning_decisions": [decisions[run_id] for run_id in candidate_tuning_ids],
                                  "best_id": candidate_best[-1]["id"]})
    if final_judge is not None:
        verdict = final_judge(candidate_best)
    else:
        verdict = _judge_select(candidate_best, baseline, 1, "round_final", False)
    eligible_final_ids = {run["id"] for run in rank_runs(candidate_best, baseline)}
    if (len(verdict["selected_ids"]) != min(1, len(eligible_final_ids)) or
            not set(verdict["selected_ids"]) <= eligible_final_ids):
        raise ValueError("round judge selected an invalid or below-baseline winner")
    selected_id = verdict["selected_ids"][0] if verdict["selected_ids"] else None
    return {"cycle": cycle_number,
            "promotion_source_ids": [source["id"] for source in promotion_sources],
            "promoted_ids": promoted_ids, "tuning_ids": tuning_ids,
            "candidates": candidate_details,
            "judge_final_decision": verdict,
            "epoch_decisions": dict(budgets),
            "judge_final": _judge_table(candidate_best, baseline,
                                        verdict["selected_ids"],
                                        verdict["candidate_reasons"]),
            "leader_id": selected_id,
            "selected_id": selected_id,
            "winner_psnr_db": next((run["psnr_db"] for run in candidate_best
                                    if run["id"] == selected_id), None),
            "winner_ssim": next((run["ssim"] for run in candidate_best
                                 if run["id"] == selected_id), None)}


_RESUME_CONFIG_KEYS = (
    "train_split", "lr_size", "width", "depth", "families", "search_epochs",
    "min_search_epochs", "min_promotion_epochs", "min_tune_epochs",
    "promotion_epochs", "promotion_top_k", "special_promotion_multiplier",
    "tune_epochs", "max_tuning_runs", "patience", "batch_size",
    "learning_rate", "max_train_batches", "max_val_pairs", "seed", "use_llm",
    "allow_underbaseline_promotion",
)


def _write_state(path: Path, state: dict) -> None:
    """Atomically save progress after every planned decision and completed run."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _restore_state(path: Path, args: argparse.Namespace, baseline: dict,
                   baseline_pairs: int, baseline_name: str,
                   train_ids: list[str]) -> dict:
    state = _load_run_record(path)
    if state.get("state_version") != 2:
        raise ValueError("resume state uses the old selection workflow; start a new campaign with the LLM judge")
    if state.get("config", {}).get("proposals_per_round") != PROPOSALS_PER_ROUND:
        raise ValueError("resume state uses a different proposal batch size; start a new campaign")
    if not matches_metric_policy(state.get("metric_policy")) or state.get("baseline_pairs") != baseline_pairs:
        raise ValueError("resume state uses a different validation policy or pair count")
    if state.get("baseline_name") != baseline_name or any(
        abs(state["baseline"][key] - baseline[key]) > 1e-6
        for key in ("psnr_db", "ssim")
    ):
        raise ValueError("resume state uses a different validation baseline")
    if state.get("training_sample_ids") != train_ids:
        raise ValueError("resume state uses different training sample IDs")
    for key in _RESUME_CONFIG_KEYS:
        current = list(getattr(args, key)) if key == "families" else getattr(args, key)
        saved = state["config"].get(key, False) if key == "allow_underbaseline_promotion" else state["config"][key]
        if saved != current:
            raise ValueError(f"resume setting changed: {key}")
    if args.proposal_rounds < len(state["selection_cycles"]):
        raise ValueError("total round target is less than the number already completed")
    for run in state["runs"]:
        checkpoint = args.out_dir / run["id"] / "model.pt"
        per_image = args.out_dir / run["id"] / "val_per_image.csv"
        if not checkpoint.is_file() or not per_image.is_file():
            raise FileNotFoundError(f"resume files missing for {run['id']}: {checkpoint}")
        run["checkpoint"] = str(checkpoint)
        run["per_image_csv"] = str(per_image)
    state["config"]["proposal_rounds"] = args.proposal_rounds
    state["config"]["out_dir"] = str(args.out_dir)
    state["metric_policy"] = METRIC_POLICY
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path,
                        help="Baseline summary and per-image CSV measured on this dataset")
    parser.add_argument("--prior-run-json", type=Path,
                        help="Completed prior run to show the LLM; never replaces current validation")
    parser.add_argument("--train-split", choices=("train", "train_small"), default="train_small")
    parser.add_argument("--lr-size", type=int, default=75)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--families", nargs="+", type=canonical_sr_family,
                        choices=FAMILIES, default=FAMILIES,
                        help="Allowed proposal families; defaults to every supported family")
    parser.add_argument("--search-epochs", type=int, default=10)
    parser.add_argument("--min-search-epochs", type=int, default=1)
    parser.add_argument("--proposal-rounds", type=int, default=10,
                        help="5-model batch rounds; each batch uses the previous round winner")
    parser.add_argument("--rounds-this-run", type=int,
                        help="Stop after this many additional completed rounds; use 1 for Kaggle")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from sr_scientist_state.json in the output directory")
    parser.add_argument("--promotion-epochs", type=int, default=30,
                        help="Additional epochs for each promoted learning-rate branch")
    parser.add_argument("--min-promotion-epochs", type=int, default=1)
    parser.add_argument("--promotion-top-k", type=int, default=3,
                        help="Number of screened models to promote and tune per round (default: 3)")
    parser.add_argument("--special-promotion-multiplier", type=int, default=1,
                        help="Optional promotion epoch multiplier for diffusion and adversarial families (default: 1)")
    parser.add_argument("--tune-epochs", type=int, default=30)
    parser.add_argument("--min-tune-epochs", type=int, default=1)
    parser.add_argument("--max-tuning-runs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-pairs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable LLM batch generator, tuning planner, and judge")
    parser.add_argument("--allow-underbaseline-promotion", action="store_true",
                        help="At screening, promote promising models for more training even if below baseline; final winner still must beat it")
    args = parser.parse_args()
    if any(value < 1 for value in (args.lr_size, args.width, args.depth, args.search_epochs,
                                  args.promotion_epochs, args.promotion_top_k,
                                  args.special_promotion_multiplier,
                                  args.tune_epochs, args.max_tuning_runs, args.patience,
                                  args.batch_size)) or args.learning_rate <= 0:
        parser.error("sizes, epochs, run limits, patience and batch size must be positive")
    if any(x is not None and x < 1 for x in (args.max_train_batches, args.max_val_pairs)):
        parser.error("optional batch/pair limits must be positive")
    if args.proposal_rounds < 1:
        parser.error("proposal rounds must be positive")
    if args.rounds_this_run is not None and args.rounds_this_run < 1:
        parser.error("rounds-this-run must be positive")
    if args.promotion_top_k > PROPOSALS_PER_ROUND:
        parser.error(f"promotion top-k cannot exceed {PROPOSALS_PER_ROUND} models per round")
    for stage, lower, upper in (("screening", args.min_search_epochs, args.search_epochs),
                                ("promotion", args.min_promotion_epochs,
                                 args.promotion_epochs),
                                ("tuning", args.min_tune_epochs, args.tune_epochs)):
        if not 1 <= lower <= upper <= 100:
            parser.error(f"{stage} epoch bounds must satisfy 1 <= minimum <= maximum <= 100")
    if args.promotion_epochs * args.special_promotion_multiplier > 100:
        parser.error("special-family promotion maximum must be at most 100 epochs")
    model_settings = ModelSettings.from_env() if args.use_llm else None
    llm_identity = ({"provider": model_settings.provider, "model": model_settings.model,
                     "base_url": model_settings.base_url} if model_settings else None)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    train_set = SRPairDataset(_ref(args.splits_root / args.train_split, args.lr_size))
    train_ids = [sample_id for _, _, sample_id in train_set._pairs]
    val_set = SRPairDataset(_ref(args.splits_root / "val", args.lr_size))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows, baseline_stats = _evaluate(None, val_loader, device, args.max_val_pairs)
    with (args.out_dir / "bicubic_val_per_image.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("sample_id", "psnr_db", "ssim"))
        writer.writeheader()
        writer.writerows(baseline_rows)
    bicubic_reference = {k: baseline_stats[k]["mean"] for k in ("psnr_db", "ssim")}
    baseline, baseline_source = load_baseline(args.baseline_dir, baseline_rows)
    state_path = args.out_dir / "sr_scientist_state.json"
    if state_path.exists():
        if not args.resume:
            parser.error("output directory has saved progress; pass --resume to continue")
        state = _restore_state(state_path, args, baseline, len(baseline_rows),
                               baseline_source["name"], train_ids)
        if state.get("llm", {}).get("model_identity") != llm_identity:
            raise ValueError("resume model provider or model ID changed")
    else:
        if args.resume:
            parser.error("--resume requested but sr_scientist_state.json is missing")
        prior_runs = _load_prior_history(args.prior_run_json, baseline,
                                         len(baseline_rows), baseline_source["name"])
        state = {"state_version": 2,
                 "status": "running",
                 "selection_rule": (
                     "screening may promote underbaseline models for additional training; "
                     "every final winner must strictly exceed both baseline means; "
                     "final campaign winner uses equal PSNR and SSIM ranks with parameter count as tie breaker"
                     if args.allow_underbaseline_promotion else
                     "reject every model unless both means strictly exceed the supplied baseline; the judge chooses eligible models for promotion and each round winner; final campaign winner uses equal PSNR and SSIM ranks with parameter count as tie breaker"),
                 "config": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()} | {
                                "proposals_per_round": PROPOSALS_PER_ROUND},
                 "metric_policy": METRIC_POLICY,
                 "baseline_name": baseline_source["name"],
                 "baseline_source": baseline_source,
                 "bicubic_reference": bicubic_reference,
                 "baseline": baseline, "baseline_pairs": len(baseline_rows),
                 "training_sample_ids": train_ids,
                 "prior_history": prior_runs,
                 "runs": [], "proposal_rounds": [], "selection_cycles": [],
                 "active_round": None,
                 "llm": {"model_identity": llm_identity,
                         "prior_run_json": str(args.prior_run_json) if prior_runs else None,
                         "prior_runs_seen": len(prior_runs)},
                 "test_split_used": False,
                 "stopping": {"patience": args.patience,
                              "max_tuning_runs": args.max_tuning_runs}}
        _write_state(state_path, state)
    runs = state["runs"]
    prior_runs = state["prior_history"]
    llm_log = state["llm"]
    proposal_log = state["proposal_rounds"]
    selection_cycles = state["selection_cycles"]
    run_lookup = {run["id"]: run for run in runs}
    state["status"] = "running"
    state["baseline_source"] = baseline_source
    state["bicubic_reference"] = bicubic_reference

    def persist() -> None:
        _write_state(state_path, state)

    visuals = state.setdefault("visualizations", {})
    visual_ids = visuals.setdefault(
        "sample_ids", fixed_sample_ids([row["sample_id"] for row in baseline_rows]))
    if not set(visual_ids) <= {row["sample_id"] for row in baseline_rows}:
        raise ValueError("saved comparison images are not in the validation set")
    visual_rounds = visuals.setdefault("rounds", {})
    persist()

    def ensure_plot(filename: str, selected_runs: list[dict], title: str) -> str:
        relative = f"visualizations/{filename}"
        destination = args.out_dir / relative
        if selected_runs and not destination.is_file():
            print(f"Creating validation comparison: {relative}", flush=True)
            render_comparison(val_set, visual_ids, selected_runs,
                              args.out_dir / "bicubic_val_per_image.csv",
                              destination, device, title)
            print(f"Saved validation comparison: {relative}", flush=True)
        return relative

    def ensure_run_plot(run: dict) -> None:
        relative = ensure_plot(f"{run['id']}.png", [run],
                               f"{run['id']}: validation comparison after {run['stage']}")
        visuals.setdefault("runs", {})[run["id"]] = relative
        persist()

    def ensure_stage_plot(number: int, stage: str, run_ids: list[str]) -> None:
        if not run_ids:
            return
        relative = ensure_plot(f"round_{number:02d}_{stage}.png",
                               [run_lookup[run_id] for run_id in run_ids],
                               f"Round {number}: {stage} stage")
        visual_rounds.setdefault(str(number), {})[stage] = relative
        persist()

    def ensure_screening_plot(number: int, screened_ids: list[str]) -> None:
        if len(screened_ids) != PROPOSALS_PER_ROUND:
            return
        relative = ensure_plot(f"round_{number:02d}_screening.png",
                               [run_lookup[run_id] for run_id in screened_ids],
                               f"Round {number}: all five proposed models after screening")
        visual_rounds.setdefault(str(number), {})["screening"] = relative
        persist()

    def ensure_finalist_plot(cycle: dict) -> None:
        candidates = [run_lookup[item["best_id"]] for item in cycle["candidates"]]
        if candidates:
            number = cycle["cycle"]
            relative = ensure_plot(f"round_{number:02d}_finalists.png", candidates,
                                   f"Round {number}: best checkpoint for each finalist")
            visual_rounds.setdefault(str(number), {})["finalists"] = relative
            persist()

    # An interrupted run may have finished training or judging before a plot was saved.
    for completed_run in runs:
        ensure_run_plot(completed_run)
    for completed_cycle in selection_cycles:
        ensure_screening_plot(completed_cycle["cycle"], completed_cycle["screened_ids"])
        ensure_stage_plot(completed_cycle["cycle"], "promotion", completed_cycle["promoted_ids"])
        ensure_stage_plot(completed_cycle["cycle"], "tuning", completed_cycle["tuning_ids"])
        ensure_stage_plot(completed_cycle["cycle"], "winner",
                          [completed_cycle["selected_id"]] if completed_cycle["selected_id"] else [])
        ensure_finalist_plot(completed_cycle)
    if state["active_round"]:
        active_saved = state["active_round"]
        ensure_screening_plot(active_saved["round"], active_saved["screened_ids"])

    def run_trial(spec: SRArchitecture, **kwargs) -> dict:
        run_id = kwargs["run_id"]
        if run_id in run_lookup:
            existing = run_lookup[run_id]
            if existing["architecture"] != asdict(spec) or existing["stage"] != kwargs["stage"]:
                raise ValueError(f"saved run {run_id} differs from the requested experiment")
            return existing
        completed_path = args.out_dir / run_id / "run.json"
        if completed_path.is_file():
            recovered = _load_run_record(completed_path)
            expected = {"architecture": asdict(spec), "stage": kwargs["stage"],
                        "epochs_this_stage": kwargs["epochs"],
                        "learning_rate": kwargs["learning_rate"],
                        "loss_l1_weight": kwargs["loss_l1_weight"],
                        "augment": kwargs["augment"],
                        "resumed_from": (kwargs["resume_from"]["id"]
                                         if kwargs.get("resume_from") else None)}
            if any(recovered.get(key) != value for key, value in expected.items()):
                raise ValueError(f"completed run {run_id} differs from resume plan")
            if recovered.get("loss_id", default_loss(spec.family)) != kwargs.get("loss_id", default_loss(spec.family)):
                raise ValueError(f"completed run {run_id} uses a different loss")
            if ((args.out_dir / run_id / "model.pt").is_file() and
                    (args.out_dir / run_id / "val_per_image.csv").is_file()):
                recovered["checkpoint"] = str(args.out_dir / run_id / "model.pt")
                recovered["per_image_csv"] = str(args.out_dir / run_id / "val_per_image.csv")
                recovered["passes_baseline"] = passes_baseline(recovered, baseline)
                runs.append(recovered)
                run_lookup[run_id] = recovered
                persist()
                ensure_run_plot(recovered)
                return recovered
        run = _run(spec, **kwargs)
        run["passes_baseline"] = passes_baseline(run, baseline)
        print(f"{run_id}: loss={run.get('loss_id', default_loss(spec.family))} "
              f"val_psnr={run['psnr_db']:.4f}dB val_ssim={run['ssim']:.6f} "
              f"beats_baseline={run['passes_baseline']}", flush=True)
        runs.append(run)
        run_lookup[run_id] = run
        persist()
        ensure_run_plot(run)
        return run

    allowed_families = list(dict.fromkeys(args.families))
    round_limit = min(args.proposal_rounds,
                      len(selection_cycles) + (args.rounds_this_run or args.proposal_rounds))
    for round_number in range(len(selection_cycles) + 1, round_limit + 1):
        previous_selection = selection_cycles[-1] if selection_cycles else None
        winner_id = previous_selection["selected_id"] if previous_selection else None
        winner = run_lookup.get(winner_id)
        active = state["active_round"]
        if active is None:
            feedback = ({"screening": previous_selection["judge_screening_decision"],
                         "final": previous_selection["judge_final_decision"]}
                        if previous_selection else None)
            specs, batch_plan = _proposal_batch(
                prior_runs + runs, winner, baseline, allowed_families, round_number,
                args.lr_size, args.width, args.depth, args.use_llm, feedback,
                (args.min_search_epochs, args.search_epochs))
            active = {"round": round_number, "specs": [asdict(spec) for spec in specs],
                      "batch_plan": batch_plan, "winner_before_id": (
                          previous_selection["selected_id"] if previous_selection else None),
                      "leader_before_id": (
                          previous_selection["leader_id"] if previous_selection else None),
                      "screened_ids": [], "tuning_decisions": {}, "epoch_decisions": {}}
            state["active_round"] = active
            persist()
            for index, (spec, loss_id, epochs) in enumerate(
                    zip(specs, batch_plan.get("loss_ids", [default_loss(s.family) for s in specs]),
                        batch_plan.get("screening_epochs", [args.search_epochs] * len(specs))),
                    start=1):
                print(f"Round {round_number} proposal {index}: {spec.family} "
                      f"width={spec.width} depth={spec.depth} loss={loss_id} "
                      f"screening_epochs={epochs}", flush=True)
        elif active["round"] != round_number:
            raise ValueError("saved active round does not follow completed rounds")
        specs = [SRArchitecture(**spec) for spec in active["specs"]]
        batch_plan = active["batch_plan"]
        screened = []
        for index, spec in enumerate(specs, start=1):
            run = run_trial(
                spec, train_set=train_set, val_loader=val_loader, device=device,
                output=args.out_dir,
                epochs=batch_plan.get("screening_epochs", [args.search_epochs] * len(specs))[index - 1],
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                loss_l1_weight=1.0, augment=False,
                loss_id=batch_plan.get("loss_ids", [default_loss(item.family) for item in specs])[index - 1],
                max_train_batches=args.max_train_batches,
                max_val_pairs=args.max_val_pairs, seed=args.seed,
                run_id=f"round_{round_number:02d}_screen_{index:02d}",
                stage="search")
            screened.append(run)
            if run["id"] not in active["screened_ids"]:
                active["screened_ids"].append(run["id"])
                persist()
        ensure_screening_plot(round_number, active["screened_ids"])
        if "selected_candidate_ids" not in active:
            shortlist = _judge_select(screened, baseline, args.promotion_top_k,
                                      "screening", args.use_llm,
                                      args.allow_underbaseline_promotion)
            active["judge_screening_decision"] = shortlist
            active["selected_candidate_ids"] = shortlist["selected_ids"]
            active["judge_screening"] = _judge_table(
                screened, baseline, active["selected_candidate_ids"],
                shortlist["candidate_reasons"],
                exploratory=args.allow_underbaseline_promotion)
            persist()
            print(f"Round {round_number} promotion shortlist: "
                  f"{active['selected_candidate_ids'] or 'none passed the baseline'}. "
                  f"Reason: {shortlist['reasoning']}", flush=True)
        promotion_sources = [run_lookup[run_id] for run_id in active["selected_candidate_ids"]]

        def final_judge(candidates: list[dict], round_state: dict = active) -> dict:
            if "judge_final_decision" not in round_state:
                round_state["judge_final_decision"] = _judge_select(
                    candidates, baseline, 1, "round_final", args.use_llm)
                persist()
            return round_state["judge_final_decision"]

        selection = _complete_selection_cycle(
            runs, promotion_sources, cycle_number=round_number, train_set=train_set,
            val_loader=val_loader, device=device, args=args, baseline=baseline,
            prior_runs=prior_runs, llm_log=llm_log, run_trial=run_trial,
            decision_store=active["tuning_decisions"], persist=persist,
            final_judge=final_judge,
            epoch_decisions=active.setdefault("epoch_decisions", {}))
        selection["screened_ids"] = [run["id"] for run in screened]
        selection["judge_screening"] = active["judge_screening"]
        selection["judge_screening_decision"] = active["judge_screening_decision"]
        selection_cycles.append(selection)
        print(f"Round {round_number} winner: "
              f"{selection['selected_id'] or 'none passed the baseline'}. "
              f"Reason: {selection['judge_final_decision']['reasoning']}", flush=True)
        proposal_log.append({"round": round_number,
                             "winner_before_id": active["winner_before_id"],
                             "leader_before_id": active["leader_before_id"],
                             "winner_after_id": selection["selected_id"],
                             "leader_after_id": selection["leader_id"],
                             **batch_plan,
                             "proposals": [{"id": run["id"],
                                            "architecture": run["architecture"],
                                            "loss_id": run.get("loss_id", default_loss(run["architecture"]["family"])),
                                            "psnr_db": run["psnr_db"], "ssim": run["ssim"],
                                            "reasoning": batch_plan["candidate_reasons"][index]}
                                           for index, run in enumerate(screened)],
                             "selected_candidate_ids": active["selected_candidate_ids"]})
        state["active_round"] = None
        persist()
        ensure_stage_plot(round_number, "promotion", selection["promoted_ids"])
        ensure_stage_plot(round_number, "tuning", selection["tuning_ids"])
        ensure_stage_plot(round_number, "winner",
                          [selection["selected_id"]] if selection["selected_id"] else [])
        ensure_finalist_plot(selection)
    family_best = _best_per_family(runs, baseline) if runs else []
    round_winners = [next(run for run in runs if run["id"] == cycle["selected_id"])
                     for cycle in selection_cycles if cycle["selected_id"]]
    final = rank_runs(round_winners, baseline)
    state.update({"status": ("complete" if len(selection_cycles) == args.proposal_rounds
                             else "ready_to_resume"),
                  "rounds_completed": len(selection_cycles),
                  "ranking": ranking_details(runs, baseline),
                  "promoted_ids": [run_id for cycle in selection_cycles
                                   for run_id in cycle["promoted_ids"]],
                  "family_best_ids": {run["architecture"]["family"]: run["id"]
                                      for run in family_best},
                  "total_train_seconds": sum(run.get("train_seconds", 0) for run in runs),
                  "total_optimizer_steps": sum(run.get("steps_this_stage", 0) for run in runs),
                  "selected_id": final[0]["id"] if final else None,
                  "judge_final": _judge_table(round_winners, baseline,
                                               [final[0]["id"]] if final else [])})
    persist()
    if len(selection_cycles) == args.proposal_rounds and round_winners:
        visuals["final"] = ensure_plot("round_winners.png", round_winners,
                                       "Final comparison: winner from each round")
        persist()
    (args.out_dir / "sr_scientist_run.json").write_text(json.dumps(state, indent=2) + "\n")
    print(f"Completed rounds: {len(selection_cycles)}/{args.proposal_rounds}")
    print(f"Selected: {state['selected_id'] or 'none passed both baseline checks'}")
    print(f"Log: {args.out_dir / 'sr_scientist_run.json'}")


if __name__ == "__main__":
    main()
