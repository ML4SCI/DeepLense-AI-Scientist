# agents/_architecture_search.py
"""Tree-based architecture exploration for the AI Scientist (prototype).

The design discussed with Michael (2026-07-25):

    GENERATE  ~N candidate architectures (LLM, structured ArchitectureSpecs)
    JUDGE     LLM-as-judge ranks them from the specs alone; take the top-k
    RUN       train the top-k for REAL but short (cheap ranking signal)
    PRUNE     keep the best by REAL validation metrics (code-side, not the LLM)
    (refine)  optionally one more shallow round: variants of the winner -> train -> prune

The winner then goes to the EXISTING closed-loop hyperparameter tuner
(``ExperimentLoop``) — tree search picks the architecture, the ReAct loop tunes it.

Bias rule: prompts contain no preferred-architecture hints; candidates are
constrained to the torch backend's buildable space and validated code-side.
Every phase is timed (``timings``) for the demo-runtime estimate.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from dlens.agents._base import BaseAgentConfig, DLensBaseAgent, OutputSchema
from dlens.config import build_llm
from dlens.prompts._arch_search import ARCH_GENERATOR_PROMPT, ARCH_JUDGE_PROMPT
from dlens.schemas._downstream import DatasetRef
from dlens.schemas._model_design import ArchitectureSpec, TrainingConfig
from dlens.tools._analysis import compute_analysis
from dlens.tools._inference import InferBackend
from dlens.tools._training import TrainBackend


class CandidateProposals(OutputSchema):
    """Generator output: a diverse set of buildable candidate architectures."""

    candidates: list[ArchitectureSpec] = Field(description="The proposed candidates.")


class JudgeVerdict(OutputSchema):
    """Judge output: candidate indices ranked best-first (before training)."""

    ranking: list[int] = Field(description="Candidate indices, most promising first.")


class CandidateEval(BaseModel):
    """Real short-training result for one candidate."""

    name: str
    params_m: float = Field(description="Parameter count in millions.")
    train_accuracy: float
    val_accuracy: float
    gap: float
    macro_auc: Optional[float]
    train_seconds: float


class ArchSearchResult(BaseModel):
    """Full, demo-friendly log of one architecture search."""

    proposed: list[ArchitectureSpec]
    dropped_unbuildable: list[str] = Field(default_factory=list)
    judge_ranking: list[int]
    judge_reasoning: str
    rounds: list[list[CandidateEval]]
    winner: ArchitectureSpec
    winner_eval: CandidateEval
    timings: dict[str, float]


class ArchitectureGenerator(DLensBaseAgent):
    """Proposes candidate ArchitectureSpecs (structured output, no tools)."""

    def __init__(self, *, model: Any | None = None, retries: int = 2) -> None:
        super().__init__(
            config=BaseAgentConfig(
                name="ArchitectureGenerator",
                description="Proposes diverse, buildable candidate CNN architectures for a task.",
                custom_system_prompt=ARCH_GENERATOR_PROMPT,
                model=model or build_llm(),
            ),
            output_type=CandidateProposals,
            retries=retries,
        )


class ArchitectureJudge(DLensBaseAgent):
    """Ranks candidate specs before training (LLM-as-judge)."""

    def __init__(self, *, model: Any | None = None, retries: int = 2) -> None:
        super().__init__(
            config=BaseAgentConfig(
                name="ArchitectureJudge",
                description="Ranks candidate architectures for expected generalization on a task.",
                custom_system_prompt=ARCH_JUDGE_PROMPT,
                model=model or build_llm(),
            ),
            output_type=JudgeVerdict,
            retries=retries,
        )


def _params_m(spec: ArchitectureSpec) -> float:
    """Real parameter count (builds the model; torch required)."""
    from dlens.tools._torch_backends import build_model

    return round(sum(p.numel() for p in build_model(spec).parameters()) / 1e6, 3)


class ArchitectureSearch:
    """Generate -> judge -> short-train -> prune (optionally refine once)."""

    name = "tree-search"

    def __init__(
        self,
        *,
        generator: ArchitectureGenerator | None = None,
        judge: ArchitectureJudge | None = None,
        train_backend: TrainBackend,
        infer_backend: InferBackend,
        num_candidates: int = 10,
        top_k: int = 4,
        refine_variants: int = 3,
        rounds: int = 2,
        candidate_epochs: int = 4,
    ) -> None:
        self.generator = generator or ArchitectureGenerator()
        self.judge = judge or ArchitectureJudge()
        self.train_backend = train_backend
        self.infer_backend = infer_backend
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.refine_variants = refine_variants
        self.rounds = max(1, rounds)
        self.candidate_epochs = candidate_epochs

    async def search(
        self, *, task_description: str, train_ref: DatasetRef, val_ref: DatasetRef
    ) -> ArchSearchResult:
        timings: dict[str, float] = {}

        # --- GENERATE ---
        t0 = time.monotonic()
        gen = await self.generator.arun(
            f"{task_description}\nPropose exactly {self.num_candidates} candidates."
        )
        proposed: list[ArchitectureSpec] = gen.output.candidates
        timings["generate_s"] = round(time.monotonic() - t0, 1)

        # --- FILTER (code-side buildability) ---
        dropped = [c.name for c in proposed if not c.is_buildable()]
        candidates = [c for c in proposed if c.is_buildable()]
        print(f"[search] generated {len(proposed)} candidates "
              f"({len(dropped)} dropped as unbuildable) in {timings['generate_s']}s", flush=True)
        for i, c in enumerate(candidates):
            print(f"    [{i}] {c.name:<18} {c.family.value:<7} depths={c.depths} "
                  f"widths={c.widths} (~{_params_m(c)}M params)", flush=True)

        # --- JUDGE ---
        t0 = time.monotonic()
        listing = "\n".join(
            f"[{i}] name={c.name} family={c.family.value} depths={c.depths} "
            f"widths={c.widths} params={_params_m(c)}M"
            for i, c in enumerate(candidates)
        )
        verdict = await self.judge.arun(
            f"{task_description}\n\nCandidates:\n{listing}\n\nRank all candidate indices."
        )
        ranking = [i for i in verdict.output.ranking if 0 <= i < len(candidates)]
        timings["judge_s"] = round(time.monotonic() - t0, 1)
        top = [candidates[i] for i in ranking[: self.top_k]]
        print(f"[search] judge ranking: {ranking} -> top-{self.top_k}: "
              f"{[c.name for c in top]} ({timings['judge_s']}s)", flush=True)
        print(f"    judge reasoning: {verdict.output.reasoning[:300]}", flush=True)

        # --- RUN + PRUNE (round 1) ---
        rounds_evals: list[list[CandidateEval]] = []
        evals = await self._evaluate(top, train_ref, val_ref, timings, tag="round1")
        rounds_evals.append(evals)
        best_spec, best_eval = self._best(top, evals)
        print(f"[search] round-1 winner: {best_eval.name} "
              f"(val={best_eval.val_accuracy:.4f})", flush=True)

        # --- optional shallow refinement round ---
        if self.rounds > 1 and self.refine_variants > 0:
            t0 = time.monotonic()
            table = "\n".join(
                f"{e.name}: val_accuracy={e.val_accuracy:.4f} gap={e.gap:.4f} "
                f"params={e.params_m}M" for e in evals
            )
            gen2 = await self.generator.arun(
                f"{task_description}\n\nShort-training results of the first round:\n{table}\n\n"
                f"The current best is '{best_eval.name}'. Propose exactly "
                f"{self.refine_variants} VARIATIONS of that architecture (adjust depth/"
                f"width/capacity around it) that might generalize better."
            )
            # Dedupe: drop variants structurally identical to anything already evaluated
            # (seeded training would just reproduce the same numbers).
            seen = {(c.family, tuple(c.depths or []), tuple(c.widths or [])) for c in top}
            variants = []
            for c in gen2.output.candidates:
                key = (c.family, tuple(c.depths or []), tuple(c.widths or []))
                if c.is_buildable() and key not in seen:
                    seen.add(key)
                    variants.append(c)
            variants = variants[: self.refine_variants]
            timings["refine_generate_s"] = round(time.monotonic() - t0, 1)
            print(f"[search] refinement variants: {[c.name for c in variants]} "
                  f"({timings['refine_generate_s']}s)", flush=True)
            if variants:
                evals2 = await self._evaluate(variants, train_ref, val_ref, timings, tag="round2")
                rounds_evals.append(evals2)
                cand_all = [best_spec] + variants
                eval_all = [best_eval] + evals2
                best_spec, best_eval = self._best(cand_all, eval_all)
                print(f"[search] final winner after refinement: {best_eval.name} "
                      f"(val={best_eval.val_accuracy:.4f})", flush=True)

        timings["search_total_s"] = round(sum(v for k, v in timings.items() if k != "search_total_s"), 1)
        return ArchSearchResult(
            proposed=proposed,
            dropped_unbuildable=dropped,
            judge_ranking=ranking,
            judge_reasoning=verdict.output.reasoning,
            rounds=rounds_evals,
            winner=best_spec,
            winner_eval=best_eval,
            timings=timings,
        )

    async def _evaluate(
        self, specs: list[ArchitectureSpec], train_ref: DatasetRef, val_ref: DatasetRef,
        timings: dict[str, float], tag: str,
    ) -> list[CandidateEval]:
        cfg = TrainingConfig(loss="cross_entropy", epochs=self.candidate_epochs, batch_size=32)
        evals: list[CandidateEval] = []
        for spec in specs:
            t0 = time.monotonic()
            tr = self.train_backend.train(train_ref, spec, cfg)
            ir = self.infer_backend.infer(tr.weights_path, val_ref)
            ar = compute_analysis(ir, val_ref.class_names)
            dt = round(time.monotonic() - t0, 1)
            train_acc = float(tr.metrics.get("train_accuracy", float("nan")))
            val_acc = float(ir.accuracy or float("nan"))
            evals.append(CandidateEval(
                name=spec.name, params_m=_params_m(spec),
                train_accuracy=train_acc, val_accuracy=val_acc,
                gap=round(train_acc - val_acc, 4), macro_auc=ar.macro_auc,
                train_seconds=dt,
            ))
            timings[f"{tag}_{spec.name}_s"] = dt
            print(f"    [{tag}] {spec.name:<18} val={val_acc:.4f} gap={train_acc - val_acc:+.4f} "
                  f"auc={ar.macro_auc:.4f} ({dt}s)", flush=True)
        return evals

    @staticmethod
    def _best(
        specs: list[ArchitectureSpec], evals: list[CandidateEval]
    ) -> tuple[ArchitectureSpec, CandidateEval]:
        idx = max(range(len(evals)), key=lambda i: evals[i].val_accuracy)
        return specs[idx], evals[idx]
