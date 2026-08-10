"""End-to-end AI-Scientist pipeline runner.

Chains the whole workflow from a hypothesis:

    simulate each class  ->  assemble a labelled dataset  ->  ExperimentLoop
    (design -> train -> infer -> analysis -> planner, looping)  ->  report

Offline (default): scripted planner/report + mock backends — no GPU, LLM, or network.
Live: drives the planner + report with the model from your env (set an API model).

    # offline
    uv run python scripts/run_pipeline.py

    # live, OpenAI (Aatmaj's key):
    DLENS_MODEL_PROVIDER=openai DLENS_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... \
        uv run python scripts/run_pipeline.py --live

Notes: the data stage uses the simulation tool directly with the mock backend
(deterministic, no lenstronomy); train/infer use mock backends so the run needs no
GPU. The LLM reasoning lives in the planner + report agents — that's what `--live`
exercises against a real model.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import tempfile
from pathlib import Path

from dlens.agents._experiment_loop import ExperimentLoop
from dlens.agents._experiment_planner import ExperimentPlanner
from dlens.agents._report import ReportAgent
from dlens.agents._scripted_planner import make_scripted_planner_model, make_scripted_report_model
from dlens.config import ModelSettings, build_model_from_env
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._simulation import SimConfig, SimModelConfig, SubstructureType
from dlens.tools._inference import MockInferBackend
from dlens.tools._sim_backends import get_backend
from dlens.tools._sim_runner import execute_simulation
from dlens.tools._training import MockTrainBackend

CLASSES = [SubstructureType.NO_SUBSTRUCTURE, SubstructureType.CDM, SubstructureType.VORTEX]
RULE = "-" * 70


def banner(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def simulate_and_assemble(sim_backend, root: Path, n_images: int = 6) -> DatasetRef:
    """Simulate each substructure class and assemble them into one labelled dataset."""
    raw, dest = root / "raw", root / "dataset"
    dest.mkdir(parents=True, exist_ok=True)
    outputs = []
    for cls in CLASSES:
        cfg = SimConfig(
            substructure_type=cls,
            model_config_name=SimModelConfig.MODEL_II,
            num_images=n_images,
            z_source=1.5,
            axion_mass=1e-23 if cls == SubstructureType.VORTEX else None,
        )
        out = execute_simulation(cfg, backend=sim_backend, output_root=str(raw), make_preview=False)
        outputs.append(out)
        for fname in out.filenames:
            shutil.copy(Path(out.output_dir) / fname, dest / fname)
        print(f"  simulated {cls.value:8} -> {out.num_generated} images ({out.image_shape})")
    return DatasetRef(
        root=str(dest),
        class_names=[c.value for c in CLASSES],
        image_shape=outputs[0].image_shape,
        num_samples=sum(o.num_generated for o in outputs),
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end AI-Scientist pipeline runner")
    parser.add_argument("--live", action="store_true", help="Use the model from env (API) for planner + report.")
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    banner("AI-Scientist pipeline — end to end")
    if args.live:
        print(f"  Mode: LIVE — {ModelSettings.from_env().describe()}")
        model = build_model_from_env()
        planner = ExperimentPlanner(model=model)
        report_agent = ReportAgent(model=model)
    else:
        print("  Mode: OFFLINE — scripted planner/report, mock backends (no LLM/GPU/network).")
        planner = ExperimentPlanner(model=make_scripted_planner_model(report_after=1))
        report_agent = ReportAgent(model=make_scripted_report_model())

    hypothesis = "Can a simple CNN baseline separate no_sub / cdm / vortex lensing images?"

    with tempfile.TemporaryDirectory() as tmp:
        banner("1. Simulate + assemble dataset")
        dataset = simulate_and_assemble(get_backend("mock", seed=7), Path(tmp))
        print(f"  dataset: {dataset.num_classes} classes {dataset.class_names}, "
              f"{dataset.num_samples} samples, {dataset.image_shape}")

        banner("2. Experiment loop (design -> train -> infer -> analysis -> plan)")
        loop = ExperimentLoop(
            planner=planner, report_agent=report_agent,
            train_backend=MockTrainBackend(), infer_backend=MockInferBackend(),
        )
        state, report = await loop.run(
            hypothesis=hypothesis, dataset=dataset, max_iterations=args.max_iterations
        )
        for run in state.runs:
            acc = run.analysis_result.accuracy if run.analysis_result else None
            decision = run.planner_decision.action.value if run.planner_decision else "-"
            print(f"  iter {run.iteration}: arch={run.architecture.name:9} "
                  f"acc={acc:.3f}  ->  planner: {decision}")

    banner("3. Report")
    if report is not None:
        print(f"  {report.title}")
        print(f"  best_accuracy   : {report.best_accuracy}")
        print(f"  final_architecture: {report.final_architecture}")
        print(f"  conclusion      : {report.conclusion}")
    else:
        print("  (planner chose to stop without a report)")

    ok = len(state.runs) >= 1 and (report is not None)
    banner("RESULT")
    print("  PASSED" if ok else "  (no report produced)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
