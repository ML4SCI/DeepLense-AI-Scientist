"""SR candidate buildability and two-metric selection rules."""

import json
import os
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pydantic_ai.models.test import TestModel

from dlens.agents._sr_scientist import (
    SRGenerator,
    SRCandidate,
    SREpochPlanner,
    SRJudge,
    SRPlanner,
    planner_context,
    proposal_context,
)
from dlens.sr_baseline import METRIC_POLICY
from dlens.sr_losses import (
    ALL_LOSSES,
    DIFFUSION_LOSSES,
    RECONSTRUCTION_LOSSES,
    allowed_losses,
    reconstruction_loss,
    validate_loss,
)
from dlens.sr_models import FAMILIES, SRArchitecture, build_sr_model
from dlens.sr_selection import rank_runs, ranking_details
from dlens.sr_training import SRTrainer
from scripts.report_sr_scientist import render
from scripts.run_sr_scientist import (
    _apply_llm_change,
    _epoch_budget,
    _best_per_family,
    _complete_selection_cycle,
    _judge_select,
    _judge_table,
    _next_architecture,
    _next_config,
    _promotion_sources,
    _proposal_batch,
    _restore_state,
    _run,
    _tuning_config,
    _write_state,
    main,
)


class SRScientistTests(unittest.TestCase):
    def test_llm_output_cap_reaches_all_sr_agents(self) -> None:
        with (patch.dict(os.environ, {"DLENS_MODEL_MAX_TOKENS": "4096"}),
              patch("dlens.agents._sr_scientist.build_model_from_env",
                    return_value=TestModel())):
            for agent_type in (SRGenerator, SRPlanner, SREpochPlanner, SRJudge):
                with self.subTest(agent=agent_type.__name__):
                    self.assertEqual(agent_type().model_settings, {"max_tokens": 4096})

    def test_loss_menu_has_finite_gradients_and_family_guards(self) -> None:
        target = torch.rand(1, 1, 16, 16)
        for loss_id in RECONSTRUCTION_LOSSES:
            with self.subTest(loss_id=loss_id):
                prediction = torch.rand_like(target, requires_grad=True)
                loss = reconstruction_loss(prediction, target, loss_id)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(allowed_losses("sr3"), DIFFUSION_LOSSES)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_loss("sr3", "l1_perceptual")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_loss("cnn", "l1_perceptual")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_loss("cnn", "l1_adversarial")

    def test_planner_validates_loss_choice(self) -> None:
        config = {"family": "cnn", "loss_id": "l1", "loss_l1_weight": 1.0}
        changed = _apply_llm_change(config, SimpleNamespace(knob="loss_id", value="l1_edge"))
        self.assertEqual(changed["loss_id"], "l1_edge")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            _apply_llm_change(config, SimpleNamespace(knob="loss_id", value="noise_mse"))

    def test_all_supported_families_produce_trainable_hr_images(self) -> None:
        x = torch.rand(1, 1, 15, 15)
        for family in FAMILIES:
            with self.subTest(family=family):
                model = build_sr_model(SRArchitecture(
                    family, width=8, depth=1, patch=5, image_size=15))
                result = model(x)
                self.assertEqual(tuple(result.shape), (1, 1, 30, 30))
                result.mean().backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_srcnn_refines_the_upscaled_image(self) -> None:
        model = build_sr_model(SRArchitecture("srcnn", width=16, depth=2))
        convolutions = [layer for layer in model.features.modules()
                        if isinstance(layer, nn.Conv2d)]
        self.assertEqual([layer.kernel_size for layer in convolutions],
                         [(9, 9), (1, 1), (1, 1), (5, 5)])
        seen_shapes = []
        handle = convolutions[0].register_forward_hook(
            lambda _module, inputs, _output: seen_shapes.append(tuple(inputs[0].shape)))
        try:
            prediction = model(torch.rand(2, 1, 11, 13))
        finally:
            handle.remove()
        self.assertEqual(seen_shapes, [(2, 1, 22, 26)])
        self.assertEqual(tuple(prediction.shape), (2, 1, 22, 26))

    def test_llm_schema_uses_supported_families_and_losses(self) -> None:
        fields = SRCandidate.model_json_schema()["properties"]
        self.assertEqual(set(fields["family"]["enum"]), set(FAMILIES))
        self.assertEqual(set(fields["loss_id"]["enum"]), set(ALL_LOSSES))

    def test_sr3_validation_is_independent_of_batch_position(self) -> None:
        torch.manual_seed(42)
        model = build_sr_model(SRArchitecture("sr3", width=8, depth=1)).eval()
        lr = torch.rand(2, 1, 8, 8)
        with torch.inference_mode():
            batched = model(lr)
            separate = model(lr[1:])
        torch.testing.assert_close(batched[1:], separate, rtol=1e-5, atol=2e-4)

    def test_special_training_objectives_complete_a_step(self) -> None:
        lr = torch.rand(1, 1, 15, 15)
        hr = torch.rand(1, 1, 30, 30)
        for family, objective in (("edsr", "reconstruction_l1_mse"),
                                  ("realesrgan", "reconstruction_plus_adversarial"),
                                  ("sr3", "conditional_noise_mse")):
            with self.subTest(family=family):
                spec = SRArchitecture(family, width=8, depth=1, image_size=15)
                model = build_sr_model(spec)
                trainer = SRTrainer(model, spec, 3e-4, torch.device("cpu"))
                self.assertEqual(trainer.objective, objective)
                self.assertTrue(torch.isfinite(torch.tensor(trainer.step(lr, hr, 1.0))))
                if family == "realesrgan":
                    self.assertIn("discriminator", trainer.checkpoint_extra())
                    state = {"model": model.state_dict(), **trainer.training_state()}
                    restored = SRTrainer(build_sr_model(spec), spec, 1e-4,
                                         torch.device("cpu"))
                    restored.load_training_state(state)
                    restored.set_learning_rate(1.5e-4)
                    self.assertEqual(restored.optimizer.param_groups[0]["lr"], 1.5e-4)
                    self.assertEqual(restored.discriminator_optimizer.param_groups[0]["lr"], 1.5e-4)
                    self.assertTrue(restored.optimizer.state)
                    self.assertTrue(restored.discriminator_optimizer.state)
                if family == "sr3":
                    model.eval()
                    with torch.inference_mode():
                        self.assertTrue(torch.equal(model(lr), model(lr)))

    def test_both_metrics_must_exceed_bicubic(self) -> None:
        runs = [
            {"id": "psnr_only", "psnr_db": 32.0, "ssim": 0.79, "parameters": 10},
            {"id": "both", "psnr_db": 31.0, "ssim": 0.81, "parameters": 20},
        ]
        baseline = {"psnr_db": 30.0, "ssim": 0.80}
        self.assertEqual([r["id"] for r in rank_runs(runs, baseline)], ["both"])
        self.assertFalse(ranking_details(runs, baseline)[0]["passes_baseline"])

    def test_equal_rank_sums_choose_smaller_model(self) -> None:
        runs = [
            {"id": "a", "psnr_db": 31.0, "ssim": 0.83, "parameters": 100},
            {"id": "b", "psnr_db": 30.5, "ssim": 0.85, "parameters": 50},
        ]
        self.assertEqual(rank_runs(runs, {"psnr_db": 30.0, "ssim": 0.8})[0]["id"], "b")

    def test_family_best_keeps_one_checkpoint_per_family(self) -> None:
        runs = [
            {"id": "cnn_1", "architecture": {"family": "cnn"}, "psnr_db": 31.0,
             "ssim": 0.82, "parameters": 100},
            {"id": "cnn_2", "architecture": {"family": "cnn"}, "psnr_db": 32.0,
             "ssim": 0.83, "parameters": 100},
            {"id": "vit_1", "architecture": {"family": "vit"}, "psnr_db": 29.0,
             "ssim": 0.79, "parameters": 200},
        ]
        best = _best_per_family(runs, {"psnr_db": 30.0, "ssim": 0.8})
        self.assertEqual([run["id"] for run in best], ["cnn_2", "vit_1"])

    def test_failed_tuning_change_does_not_accumulate_on_next_attempt(self) -> None:
        winner = {"architecture": {"family": "cnn", "width": 32, "depth": 2},
                  "learning_rate": 1.5e-4, "loss_l1_weight": 1.0,
                  "augment": False, "stage": "promotion", "epochs_this_stage": 5}
        scores = {"psnr_db": 31.0, "ssim": 0.82}
        first = _next_config(_tuning_config(winner, 4), scores, scores, 1)
        self.assertEqual(first["loss_id"], "l1")
        second = _next_config(_tuning_config(winner, 4), scores, scores, 2)
        self.assertTrue(second["augment"])
        self.assertEqual(second["loss_id"], "l1_mse")
        self.assertEqual(second["loss_l1_weight"], 1.0)
        self.assertEqual(second["learning_rate"], 1.5e-4)

    def test_planner_sees_previous_round_and_current_best(self) -> None:
        best = {"id": "promote_cnn_base", "stage": "promotion",
                "architecture": {"family": "cnn", "width": 32, "depth": 2},
                "psnr_db": 31.0, "ssim": 0.82, "parameters": 100,
                "learning_rate": 3e-4, "loss_l1_weight": 1.0,
                "augment": False, "epochs": 10, "train_loss": [0.02]}
        previous = {**best, "id": "tune_01", "stage": "tuning",
                    "psnr_db": 30.5, "ssim": 0.81, "train_loss": [0.03],
                    "resumed_from": best["id"]}
        context = planner_context([best, previous], _tuning_config(best, 4),
                                  best, {"psnr_db": 30.0, "ssim": 0.8})
        self.assertEqual(context["previous_round"]["id"], "tune_01")
        self.assertEqual(context["current_best"]["id"], "promote_cnn_base")
        self.assertEqual(context["previous_round"]["resumed_from"], "promote_cnn_base")
        self.assertEqual(context["current_config"]["learning_rate"], 3e-4)

    def test_ten_architecture_proposals_use_updated_measured_history(self) -> None:
        baseline = {"psnr_db": 30.0, "ssim": 0.8}
        history = [{"id": "search_cnn",
                    "architecture": {"family": "cnn", "width": 32, "depth": 2,
                                     "patch": 5, "image_size": 15},
                    "psnr_db": 31.0, "ssim": 0.82, "parameters": 100}]
        signatures = set()
        for round_number in range(1, 11):
            best = _best_per_family(history, baseline)[0]
            context = proposal_context(history, best, baseline, ["cnn"], round_number, [])
            self.assertEqual(context["previous_cycle_winner"]["id"], best["id"])
            self.assertEqual(context["latest_measured_run"]["id"], history[-1]["id"])
            spec = _next_architecture(history, baseline, ["cnn"], 15)
            signature = (spec.family, spec.width, spec.depth)
            self.assertNotIn(signature, signatures)
            signatures.add(signature)
            history.append({"id": f"proposal_{round_number:02d}",
                            "architecture": {"family": spec.family, "width": spec.width,
                                             "depth": spec.depth, "patch": spec.patch,
                                             "image_size": spec.image_size},
                            "psnr_db": 31.0 + round_number / 100,
                            "ssim": 0.82 + round_number / 1000,
                            "parameters": 100 + round_number})
        self.assertEqual(len(signatures), 10)

    def test_fallback_proposal_starts_from_previous_cycle_winner(self) -> None:
        baseline = {"psnr_db": 30.0, "ssim": 0.8}
        runs = [
            {"id": "cnn", "architecture": {"family": "cnn", "width": 32,
                                           "depth": 2}, "psnr_db": 32.0,
             "ssim": 0.85, "parameters": 100},
            {"id": "vit", "architecture": {"family": "vit", "width": 32,
                                           "depth": 2}, "psnr_db": 31.0,
             "ssim": 0.82, "parameters": 100},
        ]
        proposal = _next_architecture(runs, baseline, ["cnn", "vit"], 15, runs[1])
        self.assertEqual(proposal.family, "vit")
        context = proposal_context(runs, runs[1], baseline, ["cnn", "vit"],
                                   1, [], winner_selected=False)
        self.assertIsNone(context["previous_cycle_winner"])
        self.assertEqual(context["previous_cycle_leader"]["id"], "vit")

    def test_selection_cycle_finishes_before_next_proposal_context(self) -> None:
        architecture = {"family": "cnn", "width": 8, "depth": 1,
                        "patch": 5, "image_size": 15}
        seed = {"id": "search_cnn", "architecture": architecture,
                "psnr_db": 31.0, "ssim": 0.81, "parameters": 100,
                "learning_rate": 3e-4, "loss_l1_weight": 1.0,
                "augment": False, "epochs_this_stage": 5, "stage": "search"}
        runs = [seed]
        args = SimpleNamespace(promotion_epochs=5, special_promotion_multiplier=2,
                               out_dir=Path("/tmp"), batch_size=1,
                               max_train_batches=1, max_val_pairs=1, seed=42,
                               tune_epochs=4, max_tuning_runs=1, patience=3,
                               use_llm=False, lr_size=15)

        def fake_run(spec, **kwargs):
            return {**seed, "id": kwargs["run_id"], "stage": kwargs["stage"],
                    "psnr_db": 32.0 if kwargs["stage"] == "tuning" else 31.5,
                    "ssim": 0.82 if kwargs["stage"] == "tuning" else 0.815,
                    "resumed_from": kwargs["resume_from"]["id"] if kwargs["resume_from"] else None}

        with patch("scripts.run_sr_scientist._run", side_effect=fake_run):
            cycle = _complete_selection_cycle(
                runs, [seed], cycle_number=0, train_set=None, val_loader=None,
                device=torch.device("cpu"), args=args,
                baseline={"psnr_db": 30.0, "ssim": 0.8},
                prior_runs=[], llm_log={})
        self.assertEqual(len(cycle["promoted_ids"]), 2)
        self.assertEqual(len(cycle["tuning_ids"]), 1)
        self.assertEqual(cycle["selected_id"], "round_00_tune_search_cnn_01")
        winner = next(run for run in runs if run["id"] == cycle["selected_id"])
        context = proposal_context(runs, winner, {"psnr_db": 30.0, "ssim": 0.8},
                                   ["cnn"], 1, [], winner_selected=True)
        self.assertEqual(context["previous_cycle_winner"]["id"], cycle["selected_id"])
        self.assertEqual(context["latest_measured_run"]["id"], cycle["tuning_ids"][-1])

    def test_batch_proposes_five_before_screening_and_selects_three_models(self) -> None:
        baseline = {"psnr_db": 30.0, "ssim": 0.8}
        first_context = proposal_context([], None, baseline, list(FAMILIES),
                                         1, [], winner_selected=False)
        self.assertIsNone(first_context["previous_cycle_winner"])
        self.assertIsNone(first_context["previous_cycle_leader"])
        specs, log = _proposal_batch([], None, baseline, list(FAMILIES),
                                     1, 15, 8, 1, False)
        signatures = [(spec.family, spec.width, spec.depth) for spec in specs]
        self.assertEqual(len(signatures), 5)
        self.assertEqual(len(set(signatures)), 5)
        self.assertEqual(log["source"], "deterministic")
        self.assertEqual(len(log["loss_ids"]), 5)
        for spec, loss_id in zip(specs, log["loss_ids"]):
            self.assertIn(loss_id, allowed_losses(spec.family))
        screened = [{"id": f"screen_{index:02d}",
                     "architecture": {"family": spec.family, "width": spec.width,
                                      "depth": spec.depth},
                     "psnr_db": 30.0 + index / 10, "ssim": 0.8 + index / 100,
                     "parameters": 100}
                    for index, spec in enumerate(specs, start=1)]
        selected = _promotion_sources(screened, baseline, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual({run["id"] for run in selected},
                         {f"screen_{index:02d}" for index in range(3, 6)})
        next_specs, _ = _proposal_batch(screened, selected[0], baseline,
                                       list(FAMILIES), 2, 15, 8, 1, False)
        self.assertEqual(len(next_specs), 5)
        self.assertEqual(next_specs[0].family, selected[0]["architecture"]["family"])
        self.assertFalse(set(signatures) & {
            (spec.family, spec.width, spec.depth) for spec in next_specs})

    def test_generator_assigns_separate_screening_epoch_budgets(self) -> None:
        families = ["cnn", "srcnn", "resnet", "vit", "hybrid"]
        proposed = [SimpleNamespace(family=family, width=32, depth=2,
                                    loss_id=allowed_losses(family)[0],
                                    screening_epochs=2 + index,
                                    reasoning="Architecture-specific compute budget")
                    for index, family in enumerate(families)]
        answer = SimpleNamespace(candidates=proposed, reasoning="Diverse architectures")
        with patch("dlens.agents._sr_scientist.SRGenerator") as generator:
            generator.return_value.propose_batch = AsyncMock(return_value=answer)
            specs, log = _proposal_batch([], None, {"psnr_db": 30.0, "ssim": 0.8},
                                         families, 1, 15, 32, 2, True,
                                         screening_epoch_bounds=(2, 6))
        self.assertEqual(len(specs), 5)
        self.assertEqual(log["screening_epochs"], [2, 3, 4, 5, 6])

    def test_epoch_planner_validates_promotion_and_tuning_budgets(self) -> None:
        candidate = {"id": "model", "architecture": {"family": "edsr"},
                     "psnr_db": 31.0, "ssim": 0.75}
        baseline = {"psnr_db": 34.0, "ssim": 0.82}
        answer = SimpleNamespace(epochs=13, reasoning="Needs more training")
        with patch("dlens.agents._sr_scientist.SREpochPlanner") as planner:
            planner.return_value.decide = AsyncMock(return_value=answer)
            promotion = _epoch_budget("promotion", candidate, baseline, [], (10, 25), True)
            self.assertEqual(promotion["epochs"], 13)
            answer.epochs = 26
            with self.assertRaisesRegex(ValueError, "expected \\[10, 25\\]"):
                _epoch_budget("promotion", candidate, baseline, [], (10, 25), True)
            answer.epochs = 42
            tuning = _epoch_budget("tuning", candidate, baseline, [], (30, 50), True)
            self.assertEqual(tuning["epochs"], 42)

    def test_first_tuning_run_uses_planned_epochs(self) -> None:
        source = {"id": "screen_a", "architecture": {"family": "cnn", "width": 8,
                  "depth": 1, "patch": 5, "image_size": 15},
                  "psnr_db": 31.0, "ssim": 0.81, "parameters": 100,
                  "learning_rate": 3e-4, "loss_l1_weight": 1.0,
                  "loss_id": "l1", "augment": False, "epochs_this_stage": 3,
                  "stage": "search"}
        args = SimpleNamespace(promotion_epochs=25, min_promotion_epochs=10,
                               special_promotion_multiplier=1, out_dir=Path("/tmp"),
                               batch_size=1, max_train_batches=1, max_val_pairs=1,
                               seed=42, tune_epochs=50, min_tune_epochs=30,
                               max_tuning_runs=1, patience=3, use_llm=True, lr_size=15)
        calls = []
        def fake_run(spec, **kwargs):
            calls.append((kwargs["stage"], kwargs["epochs"]))
            return {**source, "id": kwargs["run_id"], "stage": kwargs["stage"],
                    "epochs_this_stage": kwargs["epochs"],
                    "psnr_db": 32.0, "ssim": 0.83}
        budgets = {}
        with patch("scripts.run_sr_scientist._epoch_budget",
                   side_effect=lambda stage, *rest: {"source": "llm",
                       "epochs": 13 if stage == "promotion" else 42,
                       "reasoning": "Measured architecture needs this budget"}):
            cycle = _complete_selection_cycle(
                [source], [source], cycle_number=1, train_set=None,
                val_loader=None, device=torch.device("cpu"), args=args,
                baseline={"psnr_db": 30.0, "ssim": 0.8}, prior_runs=[],
                llm_log={}, run_trial=fake_run, epoch_decisions=budgets)
        self.assertEqual(calls, [("promotion", 13), ("promotion", 13), ("tuning", 42)])
        self.assertEqual(len(cycle["epoch_decisions"]), 2)

    def test_every_selected_model_gets_its_own_tuning_budget(self) -> None:
        baseline = {"psnr_db": 30.0, "ssim": 0.8}
        screened = [
            {"id": f"screen_{index:02d}",
             "architecture": {"family": "cnn", "width": 8 * index,
                              "depth": 1, "patch": 5, "image_size": 15},
             "psnr_db": 30.0 + index / 10, "ssim": 0.8 + index / 100,
             "parameters": 100 + index, "learning_rate": 3e-4,
             "loss_l1_weight": 1.0, "augment": False,
             "epochs_this_stage": 5, "stage": "search"}
            for index in range(1, 4)]
        self.assertEqual(len(_promotion_sources(screened, baseline, 3)), 3)
        args = SimpleNamespace(promotion_epochs=5, special_promotion_multiplier=2,
                               out_dir=Path("/tmp"), batch_size=1,
                               max_train_batches=1, max_val_pairs=1, seed=42,
                               tune_epochs=4, max_tuning_runs=2, patience=3,
                               use_llm=False, lr_size=15)

        def fake_run(spec, **kwargs):
            parent = kwargs["resume_from"]
            return {**parent, "id": kwargs["run_id"], "stage": kwargs["stage"],
                    "architecture": {"family": spec.family, "width": spec.width,
                                     "depth": spec.depth, "patch": spec.patch,
                                     "image_size": spec.image_size},
                    "psnr_db": parent["psnr_db"] + 0.01,
                    "ssim": parent["ssim"] + 0.001,
                    "epochs_this_stage": kwargs["epochs"],
                    "resumed_from": parent["id"]}

        with patch("scripts.run_sr_scientist._run", side_effect=fake_run):
            cycle = _complete_selection_cycle(
                list(screened), screened, cycle_number=1, train_set=None,
                val_loader=None, device=torch.device("cpu"), args=args,
                baseline=baseline, prior_runs=[], llm_log={},
                final_judge=lambda candidates: {
                    "source": "llm", "inputs": {},
                    "selected_ids": [candidates[0]["id"]],
                    "candidate_reasons": {item["id"]: "Validated evidence"
                                          for item in candidates},
                    "reasoning": "Select the first eligible candidate"})
        self.assertEqual(len(cycle["candidates"]), 3)
        self.assertEqual(len(cycle["promoted_ids"]), 6)
        self.assertEqual(len(cycle["tuning_ids"]), 6)
        self.assertTrue(all(len(item["tuning_ids"]) == 2
                            for item in cycle["candidates"]))
        self.assertEqual(cycle["selected_id"], cycle["candidates"][0]["best_id"])
        self.assertEqual(cycle["judge_final_decision"]["source"], "llm")

    def test_judge_audit_explains_baseline_and_selection(self) -> None:
        baseline = {"psnr_db": 30.0, "ssim": 0.8}
        candidates = [
            {"id": "a", "architecture": {"family": "cnn"}, "psnr_db": 31.0,
             "ssim": 0.79, "parameters": 10},
            {"id": "b", "architecture": {"family": "vit"}, "psnr_db": 30.5,
             "ssim": 0.82, "parameters": 20},
        ]
        rows = _judge_table(candidates, baseline, ["b"])
        self.assertFalse(rows[0]["passes_baseline"])
        self.assertTrue(rows[1]["passes_baseline"])
        self.assertTrue(rows[1]["selected"])
        self.assertIn("SSIM <= baseline", rows[0]["reason"])
        self.assertIn("Selected", rows[1]["reason"])

    def test_llm_judge_cannot_select_below_baseline_or_invent_ids(self) -> None:
        baseline = {"psnr_db": 30.0, "ssim": 0.8}
        candidates = [
            {"id": f"model_{i}", "architecture": {"family": "cnn"},
             "psnr_db": 31.0 + i / 10, "ssim": 0.81 + i / 100,
             "parameters": 100, "stage": "search"}
            for i in range(7)]
        candidates.append({"id": "failed", "architecture": {"family": "vit"},
                           "psnr_db": 31.0, "ssim": 0.8, "parameters": 200})
        reasons = {run["id"]: "Measured validation evidence" for run in candidates[:-1]}
        answer = SimpleNamespace(selected_ids=[f"model_{i}" for i in range(1, 4)],
                                 candidate_reasons=reasons, reasoning="Three strongest fits")
        with patch("dlens.agents._sr_scientist.SRJudge") as judge:
            judge.return_value.decide = AsyncMock(return_value=answer)
            result = _judge_select(candidates, baseline, 3, "screening", True)
            self.assertEqual(result["source"], "llm")
            self.assertNotIn("failed", result["selected_ids"])
            self.assertEqual(len(result["inputs"]["rejected_candidates"]), 1)
            evidence = judge.return_value.decide.call_args.args[0]
            self.assertEqual(len(evidence["eligible_candidates"]), 7)
            answer.selected_ids = ["failed"] + answer.selected_ids[1:]
            with self.assertRaisesRegex(ValueError, "below-baseline IDs"):
                _judge_select(candidates, baseline, 3, "screening", True)
            answer.selected_ids = ["model_0"]
            final = _judge_select(candidates, baseline, 1, "round_final", True)
            self.assertEqual(final["selected_ids"], ["model_0"])

    def test_no_model_passes_baseline_so_judge_skips_tuning(self) -> None:
        baseline = {"psnr_db": 30.0, "ssim": 0.8}
        candidates = [{"id": "failed", "architecture": {"family": "cnn"},
                       "psnr_db": 30.0, "ssim": 0.9, "parameters": 100}]
        self.assertEqual(_promotion_sources(candidates, baseline, 3), [])
        with patch("dlens.agents._sr_scientist.SRJudge") as judge:
            result = _judge_select(candidates, baseline, 3, "screening", True)
            judge.assert_not_called()
        self.assertEqual(result["selected_ids"], [])
        self.assertEqual(result["source"], "baseline_gate")
        cycle = _complete_selection_cycle(
            [], [], cycle_number=1, train_set=None, val_loader=None,
            device=torch.device("cpu"), args=SimpleNamespace(),
            baseline=baseline, prior_runs=[], llm_log={})
        self.assertIsNone(cycle["selected_id"])
        self.assertEqual(cycle["promoted_ids"], [])

    def test_exploratory_screening_promotes_below_baseline_but_final_gate_remains(self) -> None:
        baseline = {"psnr_db": 34.0, "ssim": 0.82}
        candidates = [
            {"id": "a", "architecture": {"family": "edsr"}, "psnr_db": 32.3,
             "ssim": 0.78, "parameters": 100},
            {"id": "b", "architecture": {"family": "rcan"}, "psnr_db": 31.5,
             "ssim": 0.75, "parameters": 100},
        ]
        screening = _judge_select(candidates, baseline, 2, "screening", False,
                                  allow_underbaseline_promotion=True)
        self.assertEqual(screening["selected_ids"], ["a", "b"])
        self.assertEqual(screening["inputs"]["rejected_candidates"], [])
        table = _judge_table(candidates, baseline, screening["selected_ids"],
                             screening["candidate_reasons"], exploratory=True)
        self.assertTrue(all(row["selected"] and not row["passes_baseline"] for row in table))
        self.assertIn("may train further", table[0]["reason"])
        answer = SimpleNamespace(selected_ids=["b", "a"],
                                 candidate_reasons={"a": "Strong PSNR", "b": "Diverse family"},
                                 reasoning="Both warrant longer training")
        with patch("dlens.agents._sr_scientist.SRJudge") as judge:
            judge.return_value.decide = AsyncMock(return_value=answer)
            llm_screening = _judge_select(candidates, baseline, 2, "screening", True,
                                          allow_underbaseline_promotion=True)
        self.assertEqual(llm_screening["selected_ids"], ["b", "a"])
        final = _judge_select(candidates, baseline, 1, "round_final", False,
                              allow_underbaseline_promotion=True)
        self.assertEqual(final["selected_ids"], [])

    def test_all_rejected_round_completes_without_promotion(self) -> None:
        class Pairs:
            def __init__(self) -> None:
                self._pairs = [(None, None, "one")]

        def fake_run(spec, **kwargs):
            folder = kwargs["output"] / kwargs["run_id"]
            folder.mkdir(parents=True)
            (folder / "model.pt").write_bytes(b"weights")
            (folder / "val_per_image.csv").write_text("sample_id,psnr_db,ssim\none,29,0.7\n")
            return {"id": kwargs["run_id"], "architecture": asdict(spec),
                    "stage": "search", "parameters": 100,
                    "psnr_db": 29.0, "ssim": 0.7}

        baseline_rows = [{"sample_id": "one", "psnr_db": 28.0, "ssim": 0.6}]
        baseline_stats = {"psnr_db": {"mean": 28.0}, "ssim": {"mean": 0.6}}
        with TemporaryDirectory() as directory:
            out = Path(directory) / "results"
            command = ["run_sr_scientist.py", "--splits-root", directory,
                       "--out-dir", str(out), "--baseline-dir", directory, "--proposal-rounds", "1"]
            with (patch("scripts.run_sr_scientist.SRPairDataset", return_value=Pairs()),
                  patch("scripts.run_sr_scientist.DataLoader", return_value=object()),
                  patch("scripts.run_sr_scientist._evaluate",
                        return_value=(baseline_rows, baseline_stats)),
                  patch("scripts.run_sr_scientist.load_baseline",
                        return_value=({"psnr_db": 30.0, "ssim": 0.8},
                                      {"name": "Iterative Auto Encoder"})),
                  patch("scripts.run_sr_scientist.render_comparison"),
                  patch("scripts.run_sr_scientist._run", side_effect=fake_run) as trainer,
                  patch.object(sys, "argv", command)):
                main()
                self.assertEqual(trainer.call_count, 5)
            state = json.loads((out / "sr_scientist_state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertIsNone(state["selected_id"])
            self.assertEqual(state["selection_cycles"][0]["promotion_source_ids"], [])
            self.assertTrue(all(not run["passes_baseline"] for run in state["runs"]))

    def test_resume_state_rewrites_checkpoint_paths_and_legacy_srcnn_name(self) -> None:
        from scripts.run_sr_scientist import _RESUME_CONFIG_KEYS

        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            new = root / "new"
            new.mkdir()
            (new / "trial").mkdir()
            (new / "trial" / "model.pt").write_bytes(b"checkpoint")
            (new / "trial" / "val_per_image.csv").write_text("sample_id\n")
            settings = {key: 1 for key in _RESUME_CONFIG_KEYS}
            settings["families"] = ["srcnn"]
            args = SimpleNamespace(**settings, out_dir=new, proposal_rounds=10)
            state = {"state_version": 2, "metric_policy": METRIC_POLICY,
                "baseline_pairs": 1, "baseline_name": "Iterative Auto Encoder",
                "baseline": {"psnr_db": 30.0, "ssim": 0.8},
                "training_sample_ids": ["one"], "config": {**settings, "families": ["scnn"], "proposal_rounds": 10,
                                                     "proposals_per_round": 5},
                "active_round": {"specs": [{"family": "scnn"}]},
                "selection_cycles": [],
                "runs": [{"id": "trial", "architecture": {"family": "scnn"},
                          "checkpoint": str(old / "trial" / "model.pt"),
                          "per_image_csv": str(old / "trial" / "val_per_image.csv")}]}
            path = new / "sr_scientist_state.json"
            _write_state(path, state)
            restored = _restore_state(path, args, state["baseline"], 1,
                                      "Iterative Auto Encoder", ["one"])
            self.assertEqual(restored["runs"][0]["checkpoint"],
                             str(new / "trial" / "model.pt"))
            self.assertEqual(restored["config"]["families"], ["srcnn"])
            self.assertEqual(restored["runs"][0]["architecture"]["family"], "srcnn")
            self.assertEqual(restored["active_round"]["specs"][0]["family"], "srcnn")
            self.assertEqual(SRArchitecture("scnn").family, "srcnn")
            with self.assertRaisesRegex(ValueError, "training sample IDs"):
                _restore_state(path, args, state["baseline"], 1,
                               "Iterative Auto Encoder", ["different"])

    def test_one_round_then_resume_uses_previous_winner_without_retraining(self) -> None:
        class Pairs:
            def __init__(self) -> None:
                self._pairs = [(None, None, "one")]

        interrupted = {"once": True}

        def fake_run(spec, **kwargs):
            run_id = kwargs["run_id"]
            if run_id == "round_02_screen_04" and interrupted["once"]:
                interrupted["once"] = False
                raise RuntimeError("simulated session interruption")
            folder = kwargs["output"] / run_id
            folder.mkdir(parents=True)
            checkpoint = folder / "model.pt"
            per_image = folder / "val_per_image.csv"
            checkpoint.write_bytes(b"weights")
            per_image.write_text("sample_id,psnr_db,ssim\none,31,0.81\n")
            score = 31 + (int(run_id.split("_")[1]) / 10)
            result = {"id": run_id, "architecture": asdict(spec),
                      "stage": kwargs["stage"], "parameters": 100,
                      "psnr_db": score, "ssim": 0.81 + score / 1000,
                      "epochs": kwargs["epochs"],
                      "epochs_this_stage": kwargs["epochs"],
                      "learning_rate": kwargs["learning_rate"],
                      "loss_l1_weight": kwargs["loss_l1_weight"],
                      "augment": kwargs["augment"],
                      "resumed_from": (kwargs["resume_from"]["id"]
                                       if kwargs.get("resume_from") else None),
                      "checkpoint": str(checkpoint), "per_image_csv": str(per_image),
                      "train_seconds": 0.01, "steps_this_stage": 1,
                      "train_loss": [0.01],
                      "validation": {"psnr_db": {"mean": score, "std": 0.0},
                                     "ssim": {"mean": 0.81 + score / 1000,
                                              "std": 0.0}}}
            (folder / "run.json").write_text(json.dumps(result))
            return result

        baseline_rows = [{"sample_id": "one", "psnr_db": 29.0, "ssim": 0.7}]
        baseline_stats = {"psnr_db": {"mean": 29.0}, "ssim": {"mean": 0.7}}
        with TemporaryDirectory() as directory:
            out = Path(directory) / "results"
            command = ["run_sr_scientist.py", "--splits-root", directory,
                       "--out-dir", str(out), "--baseline-dir", directory, "--proposal-rounds", "2",
                       "--rounds-this-run", "1", "--max-tuning-runs", "1"]
            with (patch("scripts.run_sr_scientist.SRPairDataset", return_value=Pairs()),
                  patch("scripts.run_sr_scientist.DataLoader", return_value=object()),
                  patch("scripts.run_sr_scientist._evaluate",
                        return_value=(baseline_rows, baseline_stats)),
                  patch("scripts.run_sr_scientist.load_baseline",
                        return_value=({"psnr_db": 30.0, "ssim": 0.8},
                                      {"name": "Iterative Auto Encoder"})),
                  patch("scripts.run_sr_scientist.render_comparison"),
                  patch("scripts.run_sr_scientist._run", side_effect=fake_run) as trainer):
                with patch.object(sys, "argv", command):
                    main()
                first_state = json.loads((out / "sr_scientist_state.json").read_text())
                self.assertEqual(first_state["rounds_completed"], 1)
                self.assertEqual(first_state["status"], "ready_to_resume")
                self.assertEqual(trainer.call_count, 14)
                with (patch.object(sys, "argv", command + ["--resume"]),
                      self.assertRaisesRegex(RuntimeError, "session interruption")):
                    main()
                interrupted_state = json.loads((out / "sr_scientist_state.json").read_text())
                self.assertEqual(interrupted_state["rounds_completed"], 1)
                self.assertEqual(len(interrupted_state["active_round"]["screened_ids"]), 3)
                self.assertIn("Planner reason:", render(interrupted_state))
                with patch.object(sys, "argv", command + ["--resume"]):
                    main()
                second_state = json.loads((out / "sr_scientist_state.json").read_text())
                self.assertEqual(second_state["rounds_completed"], 2)
                self.assertEqual(second_state["status"], "complete")
                self.assertEqual(trainer.call_count, 29)
                self.assertEqual(second_state["proposal_rounds"][1]["winner_before_id"],
                                 first_state["selection_cycles"][0]["selected_id"])
                report = render(second_state)
                self.assertIn("How evidence was collected", report)
                self.assertIn("Screening judge compared all 5 models", report)
                self.assertIn("Final comparison across round winners", report)

    def test_run_continues_checkpoint_and_counts_additional_steps(self) -> None:
        class Pairs(list):
            augment = False

        lr = torch.rand(1, 15, 15)
        hr = F.interpolate(lr[None], scale_factor=2, mode="bicubic",
                           align_corners=False)[0]
        pairs = Pairs([(lr, hr, "sample")])
        with TemporaryDirectory() as directory:
            kwargs = {"train_set": pairs, "val_loader": DataLoader(pairs, batch_size=1),
                      "device": torch.device("cpu"), "output": Path(directory), "epochs": 1,
                      "batch_size": 1, "learning_rate": 3e-4, "loss_l1_weight": 1.0,
                      "loss_id": "l1_edge",
                      "augment": False, "max_train_batches": 1, "max_val_pairs": 1, "seed": 42}
            spec = SRArchitecture("cnn", width=8, depth=1, image_size=15)
            first = _run(spec, run_id="screen", **kwargs)
            second = _run(spec, run_id="continue", stage="promotion",
                          resume_from=first, **kwargs)
            uninterrupted = _run(spec, run_id="two_epochs", **{**kwargs, "epochs": 2})
            self.assertEqual(second["resumed_from"], "screen")
            self.assertEqual(second["epochs"], 2)
            self.assertEqual(second["optimizer_steps"], 2)
            self.assertEqual(second["steps_this_stage"], 1)
            checkpoint = torch.load(second["checkpoint"], weights_only=True)
            self.assertEqual(checkpoint["epochs"], 2)
            self.assertEqual(checkpoint["loss_config"]["loss_id"], "l1_edge")
            self.assertIn("optimizer", checkpoint)
            self.assertEqual(len(checkpoint["optimizer"]["state"]),
                             len(torch.load(first["checkpoint"], weights_only=True)["optimizer"]["state"]))
            direct = torch.load(uninterrupted["checkpoint"], weights_only=True)
            for name, tensor in checkpoint["model"].items():
                self.assertTrue(torch.equal(tensor, direct["model"][name]), name)


if __name__ == "__main__":
    unittest.main()
