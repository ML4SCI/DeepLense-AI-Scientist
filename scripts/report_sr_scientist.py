"""Write a complete, readable audit of an SR Scientist run JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dlens.sr_selection import rank_runs, ranking_details


def render(record: dict) -> str:
    baseline_name = record.get("baseline_name", "Bicubic")
    lines = ["# Super resolution experiment report", "",
             "## Run setup", "",
             f"- Status: {record.get('status', 'unknown')}",
             f"- Completed rounds: {record.get('rounds_completed', len(record.get('selection_cycles', [])))}",
             f"- Training split: `{record['config']['train_split']}`",
             f"- Validation pairs per model: {record['baseline_pairs']}",
             f"- Architecture families: {', '.join(record['config'].get('families', []))}",
             f"- Screening epochs: {record['config']['search_epochs']}",
             f"- Batch rounds: {record['config'].get('proposal_rounds', 0)}",
             f"- Models proposed and screened per round: {record['config'].get('proposals_per_round', 12)}",
             (f"- Promotion and tuning: up to {record['config'].get('promotion_top_k', 0)} screened models, "
              f"{record['config'].get('promotion_epochs', 0)} additional epoch(s) per learning rate"),
             ("- Diffusion/adversarial promotion epoch multiplier: "
              f"{record['config'].get('special_promotion_multiplier', 1)}"),
             f"- LLM planner and judge enabled: {record['config']['use_llm']}",
             f"- LLM model: {record.get('llm', {}).get('model_identity')}",
             f"- Test split used during search or tuning: {record['test_split_used']}",
             f"- Selection rule: {record['selection_rule']}", "",
             f"## {baseline_name} selection baseline", "",
             f"- Mean PSNR: {record['baseline']['psnr_db']:.4f} dB",
             f"- Mean SSIM: {record['baseline']['ssim']:.6f}"]
    if record.get("baseline_source"):
        source = record["baseline_source"]
        if source.get("summary"):
            lines.append(f"- Source: `{source['summary']}`")
        if source.get("checkpoint"):
            lines.append(f"- Best checkpoint: `{source['checkpoint']}` "
                         f"(epoch {source.get('checkpoint_epoch', 'unknown')})")
        if source.get("inference_steps") is not None:
            lines.append(f"- Inference steps: {source['inference_steps']}")
    if record.get("bicubic_reference"):
        bicubic = record["bicubic_reference"]
        lines += ["", "## Bicubic reference", "",
                  f"- Mean PSNR: {bicubic['psnr_db']:.4f} dB",
                  f"- Mean SSIM: {bicubic['ssim']:.6f}"]
    lines += ["", "## How evidence was collected", "",
              "The runner pairs LR and HR `.npy` images by sample ID in the training and validation folders. It trains on the selected training split, evaluates every completed checkpoint on the same validation pairs, and saves image level PSNR and SSIM in each run's `val_per_image.csv`. It computes the mean and population standard deviation for both metrics. The supplied baseline metrics provide the selection threshold, and bicubic scores verify compatible validation pairs and metric settings. The test split is not read during this workflow."]
    visuals = record.get("visualizations", {})
    if visuals.get("sample_ids"):
        lines += ["", "## Fixed validation image comparisons", "",
                  ("These are visual examples from the same saved validation sample IDs in every plot. "
                  "Displayed pixels are clipped to [0, 1]; the printed PSNR and SSIM come from "
                  "the unmodified validation predictions. Model selection uses all validation pairs."),
                  f"- Sample IDs: {', '.join(f'`{sample_id}`' for sample_id in visuals['sample_ids'])}"]
        for number, paths in sorted(visuals.get("rounds", {}).items(),
                                    key=lambda item: int(item[0])):
            for stage, label in (("screening", "all five proposed models after screening"),
                                 ("promotion", "promoted checkpoints"),
                                 ("tuning", "tuned checkpoints"),
                                 ("winner", "selected round winner"),
                                 ("finalists", "best checkpoint for each finalist")):
                if paths.get(stage):
                    lines += ["", f"### Round {number}: {label}", "",
                              f"![Round {number} {stage}]({paths[stage]})"]
        if visuals.get("final"):
            lines += ["", "### Final round-winner comparison", "",
                      f"![Final round winners]({visuals['final']})"]
    lines += ["",
             "## Every model and tuning run", "",
             "| Run | Stage | Family | Loss | Width | Depth | Parameters | Total epochs | Steps this stage | Train seconds | Learning rate | L1 weight | Augment | Train loss | PSNR (mean ± SD) | SSIM (mean ± SD) | Passed baseline | Rank sum |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|"]
    detail = {item["id"]: item for item in ranking_details(
        record["runs"], record["baseline"])}
    for run in record["runs"]:
        rank = detail[run["id"]]
        val = run["validation"]
        lines.append(
            f"| {run['id']} | {run.get('stage', 'search')} | {run['architecture']['family']} | "
            f"{run.get('loss_id', 'legacy')} | "
            f"{run['architecture']['width']} | {run['architecture']['depth']} | "
            f"{run['parameters']} | {run['epochs']} | {run.get('steps_this_stage', '—')} | "
            f"{run.get('train_seconds', 0):.3f} | {run['learning_rate']:.3g} | "
            f"{run['loss_l1_weight']:.2f} | {run['augment']} | "
            f"{run['train_loss'][-1]:.5f} | "
            f"{val['psnr_db']['mean']:.3f} ± {val['psnr_db']['std']:.3f} | "
            f"{val['ssim']['mean']:.5f} ± {val['ssim']['std']:.5f} | "
            f"{rank['passes_baseline']} | {rank['rank_sum'] or '—'} |"
        )
    lines += ["", "## Workflow decisions", ""]
    search_runs = [run for run in record["runs"] if run.get("stage", "search") == "search"]
    if search_runs:
        eligible_search = rank_runs(search_runs, record["baseline"])
        lines.append(
            f"- Top eligible screened candidate overall: "
            f"`{eligible_search[0]['id']}`." if eligible_search
            else "- No screened candidate passed both baseline metrics.")
    if record.get("selection_cycles"):
        lines += ["", "Completed selection cycles:", ""]
        for cycle in record["selection_cycles"]:
            selected_id = cycle["selected_id"] or "none passed both baseline checks"
            lines.append(
                f"- Round {cycle['cycle']}: screened {len(cycle.get('screened_ids', []))} models; "
                f"selected {', '.join(cycle['promotion_source_ids'])} for promotion and tuning; "
                f"completed {len(cycle['tuning_ids'])} tuning run(s); "
                f"round winner `{selected_id}`.")
            for key, budget in cycle.get("epoch_decisions", {}).items():
                lines.append(
                    f"  - {budget['stage']} budget for `{budget['candidate_id']}`: "
                    f"{budget['epochs']} epoch(s), source {budget['source']}. "
                    f"{budget['reasoning']}")
    if record.get("proposal_rounds"):
        lines += ["", "Batch architecture feedback:", ""]
        for batch in record["proposal_rounds"]:
            if "proposals" not in batch:
                lines.append(
                    f"- Round {batch['round']}: previous winner `{batch.get('winner_before_id') or 'none'}`; "
                    f"proposed `{batch['result_id']}`; selected "
                    f"`{batch.get('winner_after_id') or 'none'}`.")
                continue
            lines.append(
                f"- Round {batch['round']}: previous winner `{batch['winner_before_id'] or 'none'}`; "
                f"{batch['source']} proposed {len(batch['proposals'])} models together. "
                f"Selected models: {', '.join(batch.get('selected_candidate_ids', batch.get('selected_six_ids', [])))}. "
                f"Round winner: `{batch['winner_after_id'] or 'none'}`. {batch['reasoning']}")
            inputs = batch.get("planner_inputs", {})
            lines.append(
                f"  - Planner saw {inputs.get('measured_run_count', 'unknown')} previous runs, "
                f"baseline {inputs.get('baseline')}, and previous winner metrics "
                f"{inputs.get('previous_winner_metrics')}.")
            if inputs.get("previous_judge_feedback"):
                feedback = inputs["previous_judge_feedback"]
                lines.append(
                    "  - Previous judge feedback: screening rationale "
                    f"{feedback['screening']['reasoning']} Final rationale "
                    f"{feedback['final']['reasoning']}")
            for index, proposal in enumerate(batch["proposals"]):
                spec = proposal["architecture"]
                lines.append(
                    f"  - `{proposal['id']}`: {spec['family']} width {spec['width']} "
                    f"depth {spec['depth']}, loss {proposal.get('loss_id', 'legacy')}; "
                    f"screening epochs {batch.get('screening_epochs', [None] * len(batch['proposals']))[index]}; "
                    f"PSNR {proposal['psnr_db']:.4f}, "
                    f"SSIM {proposal['ssim']:.6f}. Planner reason: "
                    f"{proposal.get('reasoning', 'not recorded')}"
                )
            if batch.get("rejected"):
                lines.append(f"  - Rejected duplicate or disallowed proposals: {batch['rejected']}.")
    if record.get("selection_cycles"):
        exploratory = record.get("config", {}).get("allow_underbaseline_promotion", False)
        lines += ["", "## Judge decisions and tuning history", "",
                  (("Screening may promote a model below the baseline for additional training. "
                    "Only a final winner must strictly exceed both baseline metrics. "
                    if exploratory else
                    "The baseline gate rejects a model whenever PSNR or SSIM does not "
                    "strictly exceed the saved baseline. ") +
                   "With LLM mode enabled, the judge supplies explicit reasons. "
                   "The report records those reasons, not private chain-of-thought."), ""]
        for cycle in record["selection_cycles"]:
            lines += [f"### Round {cycle['cycle']}", "",
                      (f"Screening judge compared all {len(cycle.get('screened_ids', []))} "
                       "models to the saved baseline:"), ""]
            screening = cycle.get("judge_screening_decision", {})
            if screening:
                lines.append(
                    f"- Judge source: {screening['source']}. Selected "
                    f"{len(screening['selected_ids'])} model(s) for further training. "
                    f"Rationale: {screening['reasoning']}")
            for row in cycle.get("judge_screening", []):
                lines.append(
                    f"- `{row['id']}`: PSNR {row['psnr_db']:.4f} "
                    f"({row['psnr_above_baseline']:+.4f} vs baseline), "
                    f"SSIM {row['ssim']:.6f} "
                    f"({row['ssim_above_baseline']:+.6f}); "
                    f"rank {row['selection_order']}. {row['reason']}")
            for candidate in cycle.get("candidates", []):
                lines += ["", f"Candidate `{candidate['screened_id']}`:", ""]
                for decision in candidate.get("tuning_decisions", []):
                    before = decision["config_before"]
                    after = decision["config_after"]
                    changes = [f"{key}: {before[key]} → {after[key]}" for key in before
                               if before[key] != after[key]]
                    lines.append(
                        f"- `{decision['run_id']}` ({decision['source']}): "
                        f"{', '.join(changes) if changes else 'initial settings'}. "
                        f"Reason: {decision['reasoning']} "
                        f"Observed: {decision.get('result_scores', 'pending')}; "
                        f"baseline rejected: {decision.get('baseline_rejected', 'pending')}; "
                        f"candidate best: {decision.get('became_candidate_best', 'pending')}.")
            lines += ["", "Final judge comparison for this round:", ""]
            final_decision = cycle.get("judge_final_decision", {})
            if final_decision:
                lines.append(
                    f"- Judge source: {final_decision['source']}. "
                    f"Rationale: {final_decision['reasoning']}")
            for row in cycle.get("judge_final", []):
                lines.append(
                    f"- `{row['id']}`: PSNR {row['psnr_db']:.4f}, "
                    f"SSIM {row['ssim']:.6f}, rank {row['selection_order']}. "
                    f"{row['reason']}")
    if record.get("active_round"):
        active = record["active_round"]
        lines += ["", "## Incomplete round", "",
                  (f"Round {active['round']} is saved for resume; "
                   f"{len(active.get('screened_ids', []))} of "
                   f"{record['config'].get('proposals_per_round', 12)} screening runs completed."), ""]
        plan = active.get("batch_plan", {})
        for index, (spec, reason) in enumerate(zip(
            active.get("specs", []), plan.get("candidate_reasons", []), strict=False),
            start=1,
        ):
            lines.append(
                f"- Proposal {index}: {spec['family']} width {spec['width']} "
                f"depth {spec['depth']}. Planner reason: {reason}")
        for decision in active.get("tuning_decisions", {}).values():
            lines.append(
                f"- Pending or completed tuning decision `{decision['run_id']}`: "
                f"{decision['reasoning']} Result: {decision.get('result_scores', 'pending')}.")
        for stage in ("judge_screening_decision", "judge_final_decision"):
            if stage in active:
                verdict = active[stage]
                lines.append(
                    f"- Saved {stage}: selected {verdict['selected_ids']}. "
                    f"Rationale: {verdict['reasoning']}")
    if record.get("promoted_ids"):
        lines.append(f"- Promoted runs: {', '.join(f'`{name}`' for name in record['promoted_ids'])}.")
    for run in record["runs"]:
        if run.get("stage") == "search":
            lines.append(f"- `{run['id']}`: batch proposal was screened.")
        elif run.get("stage") == "proposal":
            lines.append(f"- `{run['id']}`: proposed architecture was short-trained.")
        elif run.get("stage") == "promotion":
            lines.append(f"- `{run['id']}`: continued `{run['resumed_from']}` for "
                         f"{run['epochs_this_stage']} more epoch(s) at learning rate {run['learning_rate']:.3g}.")
        else:
            lines.append(f"- `{run['id']}`: tuning run; resumed from "
                         f"`{run.get('resumed_from')}`.")
    llm = record.get("llm", {})
    if llm.get("planner_decisions"):
        lines += ["", "LLM planner proposals:", ""]
        for decision in llm["planner_decisions"]:
            lines.append(f"- {decision['knob']} → {decision['value']}: {decision.get('reasoning', '')}")
    lines += ["", "## Selection", ""]
    if record.get("family_best_ids"):
        lines.append("Best checkpoint by family across all rounds:")
        for family, run_id in record["family_best_ids"].items():
            lines.append(f"- {family}: `{run_id}`")
        lines += [(f"Total training time: {record.get('total_train_seconds', 0):.1f} seconds; "
                   f"optimizer steps: {record.get('total_optimizer_steps', 0)}."), ""]
    if record.get("judge_final"):
        lines += ["Final comparison across round winners:", ""]
        for row in record["judge_final"]:
            lines.append(
                f"- `{row['id']}`: PSNR {row['psnr_db']:.4f}, "
                f"SSIM {row['ssim']:.6f}, rank {row['selection_order']}. "
                f"{row['reason']}")
        lines.append("")
    selected = next((run for run in record["runs"]
                     if run["id"] == record.get("selected_id")), None)
    if selected:
        lines += [f"Selected run: **{selected['id']}** ({selected['architecture']['family']}).",
                  f"Checkpoint: `{selected['checkpoint']}`.",
                  f"Validation PSNR: {selected['psnr_db']:.4f} dB; SSIM: {selected['ssim']:.6f}."]
    elif record.get("active_round") or not record.get("selection_cycles"):
        lines.append("No completed round winner yet. Resume the saved campaign to finish selection.")
    else:
        lines.append(f"**No model passed both {baseline_name} baseline checks.** No final model was selected.")
    lines += ["", "## Reproduction files", "",
              "- `sr_scientist_run.json`: complete configurations, metrics, and ranking.",
              "- `sr_scientist_state.json`: resumable progress and saved planner decisions.",
              "- `bicubic_val_per_image.csv`: bicubic reference score for each validation image.",
              "- External baseline CSV and summary: scores and provenance for the selection threshold.",
              "- `<run>/val_per_image.csv`: each candidate's image-level scores.",
              "- `<run>/model.pt`: saved weights and configuration for each candidate.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = render(json.loads(args.run_json.read_text()))
    out = args.out or args.run_json.with_name("SR_EXPERIMENT_REPORT.md")
    out.write_text(report)
    print(out)


if __name__ == "__main__":
    main()
