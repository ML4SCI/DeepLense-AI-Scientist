# agents/_architecture_search.py
"""Tree-based architecture exploration for the AI Scientist (prototype).

The design discussed with Michael (2026-07-25):

    for round r in 1..R:
        GENERATE  ~N candidate architectures (LLM, structured ArchitectureSpecs);
                  from round 2 on, the prompt carries EVERY measured result so
                  far, so later rounds are informed rather than blind resampling
        JUDGE     LLM-as-judge ranks them from the specs alone; take the top-k
        RUN       train the top-k for REAL but short (cheap ranking signal)
        PRUNE     keep the best by REAL validation metrics (code-side, not the LLM)
    carry the best architecture seen across all rounds

Restructured 2026-08-13 at the mentor's request. It previously ran one full round
plus a shallow refinement round that could only produce variants of the winner;
the winner now re-enters the full generation stage, so a later round is free to
leave the incumbent's neighbourhood entirely.

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
from dlens.prompts._arch_search import (
    ARCH_GENERATOR_ITERATIVE_SUFFIX,
    ARCH_GENERATOR_PROMPT,
    ARCH_JUDGE_PROMPT,
)
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


class RoundRecord(BaseModel):
    """Everything one search round did, for the write-up and for auditing."""

    round_index: int = Field(description="1-based round number.")
    proposed: list[ArchitectureSpec] = Field(description="Candidates generated this round.")
    dropped_unbuildable: list[str] = Field(default_factory=list)
    dropped_duplicate: list[str] = Field(
        default_factory=list, description="Structurally identical to something already tried."
    )
    judge_ranking: list[int] = Field(description="Judge order over this round's kept candidates.")
    judge_reasoning: str = ""
    shortlist: list[str] = Field(default_factory=list, description="Names the judge promoted.")
    evals: list["CandidateEval"] = Field(default_factory=list)
    round_winner: str = ""
    generate_s: float = 0.0
    judge_s: float = 0.0
    train_s: float = 0.0


class ArchSearchResult(BaseModel):
    """Full, demo-friendly log of one architecture search."""

    proposed: list[ArchitectureSpec]
    dropped_unbuildable: list[str] = Field(default_factory=list)
    judge_ranking: list[int]
    judge_reasoning: str
    rounds: list[list[CandidateEval]]
    round_records: list[RoundRecord] = Field(
        default_factory=list, description="Per-round detail; `rounds` is the eval-only view."
    )
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
    """Iterative beam search: R x (generate -> judge -> short-train -> prune).

    The winner does not merely seed a refinement round; every round after the
    first re-enters full generation with all measured results so far, so the
    search can abandon the incumbent's neighbourhood entirely.
    """

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
        refine_variants: int = 3,   # deprecated: the shallow refinement round is
                                    # gone; kept so existing call sites still work
        rounds: int = 3,
        candidate_epochs: int = 4,
    ) -> None:
        self.generator = generator or ArchitectureGenerator()
        self.judge = judge or ArchitectureJudge()
        self.train_backend = train_backend
        self.infer_backend = infer_backend
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.refine_variants = refine_variants  # unused; see __init__ signature
        self.rounds = max(1, rounds)
        self.candidate_epochs = candidate_epochs

    async def search(
        self, *, task_description: str, train_ref: DatasetRef, val_ref: DatasetRef
    ) -> ArchSearchResult:
        timings: dict[str, float] = {}
        records: list[RoundRecord] = []
        rounds_evals: list[list[CandidateEval]] = []

        # Everything measured so far, in proposal order, so round r>1 can be told
        # what has already been tried and how it scored.
        history: list[tuple[ArchitectureSpec, CandidateEval]] = []
        seen: set[tuple] = set()
        all_proposed: list[ArchitectureSpec] = []
        all_dropped: list[str] = []
        best_spec: ArchitectureSpec | None = None
        best_eval: CandidateEval | None = None

        for rnd in range(1, self.rounds + 1):
            # --- GENERATE ---
            prompt = f"{task_description}\nPropose exactly {self.num_candidates} candidates."
            if history:
                table = "\n".join(
                    f"- {sp.name} (family={sp.family.value}, depths={sp.depths}, "
                    f"widths={sp.widths}, {ev.params_m}M params): "
                    f"val_accuracy={ev.val_accuracy:.4f}, train_accuracy={ev.train_accuracy:.4f}, "
                    f"gap={ev.gap:+.4f}"
                    for sp, ev in history
                )
                prompt = f"{task_description}" + ARCH_GENERATOR_ITERATIVE_SUFFIX.format(
                    history=table, n=self.num_candidates
                )
            t0 = time.monotonic()
            gen = await self.generator.arun(prompt)
            generate_s = round(time.monotonic() - t0, 1)
            timings[f"round{rnd}_generate_s"] = generate_s
            proposed = gen.output.candidates
            all_proposed.extend(proposed)

            # --- FILTER: buildable, then structurally novel ---
            dropped_unbuildable = [c.name for c in proposed if not c.is_buildable()]
            dropped_duplicate: list[str] = []
            candidates: list[ArchitectureSpec] = []
            for c in proposed:
                if not c.is_buildable():
                    continue
                key = (c.family, tuple(c.depths or []), tuple(c.widths or []))
                if key in seen:
                    dropped_duplicate.append(c.name)
                    continue
                seen.add(key)
                candidates.append(c)
            all_dropped.extend(dropped_unbuildable)

            print(f"\n[search] === ROUND {rnd}/{self.rounds} === generated "
                  f"{len(proposed)} ({len(dropped_unbuildable)} unbuildable, "
                  f"{len(dropped_duplicate)} duplicate) in {generate_s}s", flush=True)
            for i, c in enumerate(candidates):
                print(f"    [{i}] {c.name:<34} {c.family.value:<12} depths={c.depths} "
                      f"widths={c.widths} (~{_params_m(c)}M)", flush=True)
            if not candidates:
                print("[search] no usable candidates this round; stopping early", flush=True)
                break

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
            # The judge may omit indices; append anything it left out so the
            # shortlist is always well defined.
            ranking += [i for i in range(len(candidates)) if i not in ranking]
            judge_s = round(time.monotonic() - t0, 1)
            timings[f"round{rnd}_judge_s"] = judge_s
            top = [candidates[i] for i in ranking[: self.top_k]]
            print(f"[search] judge ranking: {ranking} -> top-{self.top_k}: "
                  f"{[c.name for c in top]} ({judge_s}s)", flush=True)
            print(f"    judge reasoning: {verdict.output.reasoning[:300]}", flush=True)

            # --- RUN + PRUNE ---
            t0 = time.monotonic()
            evals = await self._evaluate(top, train_ref, val_ref, timings, tag=f"round{rnd}")
            train_s = round(time.monotonic() - t0, 1)
            rounds_evals.append(evals)
            history.extend(zip(top, evals))

            round_spec, round_eval = self._best(top, evals)
            if best_eval is None or round_eval.val_accuracy > best_eval.val_accuracy:
                best_spec, best_eval = round_spec, round_eval
            print(f"[search] round-{rnd} winner: {round_eval.name} "
                  f"(val={round_eval.val_accuracy:.4f}); best so far: {best_eval.name} "
                  f"(val={best_eval.val_accuracy:.4f})", flush=True)

            records.append(RoundRecord(
                round_index=rnd, proposed=proposed,
                dropped_unbuildable=dropped_unbuildable, dropped_duplicate=dropped_duplicate,
                judge_ranking=ranking, judge_reasoning=verdict.output.reasoning,
                shortlist=[c.name for c in top], evals=evals,
                round_winner=round_eval.name,
                generate_s=generate_s, judge_s=judge_s, train_s=train_s,
            ))

        if best_spec is None or best_eval is None:
            raise RuntimeError("architecture search produced no evaluable candidate")

        timings["search_total_s"] = round(
            sum(v for k, v in timings.items() if k != "search_total_s"), 1
        )
        print(f"\n[search] FINAL winner after {len(records)} round(s): {best_eval.name} "
              f"(val={best_eval.val_accuracy:.4f}, {best_eval.params_m}M params)", flush=True)
        return ArchSearchResult(
            proposed=all_proposed,
            dropped_unbuildable=all_dropped,
            judge_ranking=records[0].judge_ranking if records else [],
            judge_reasoning=records[0].judge_reasoning if records else "",
            rounds=rounds_evals,
            round_records=records,
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
