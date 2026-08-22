# Agent reliability: what the AI-Scientist loop's decisions are actually worth

Everything here measures the agent's **decision-making**, not its accuracy. The
classification numbers are reported elsewhere (`docs/PAPER_RUN_RESULTS.md`,
`docs/ARCHITECTURE_SEARCH_RESULTS.md`); this document asks a different question:
when the loop chooses to keep training, or ranks one architecture above another,
is that choice reproducible, is it diagnostic, and does it predict anything?

Five experiments. E1 and E2 hold an observation fixed and vary only LLM
sampling. E3 and E3b test whether the judge's ranking corresponds to reality.
E4 tests whether the cheap proxy the search ranks on agrees with the expensive
measurement it stands in for.

Raw artifacts: `docs/agent_reliability/`. Harnesses:
`scripts/run_planner_reliability.py`, `scripts/run_judge_reliability.py`,
`scripts/run_search_validity.py`. All runs gpt-5.2, real training, nothing mocked.

---

## 0. The headline: the loop's stopping decision is not the agent's

This is the finding I'd lead the paper with, and I found it by accident while
verifying E1.

In `src/dlens/agents/_experiment_loop.py`, the termination guard is evaluated
**before** the planner is ever asked:

- **lines 144–160** — compute `gap`, and set `guard` if the gap closed, if val
  accuracy stalled, or if the iteration budget is spent.
- **lines 162–166** — `if guard is not None:` construct a
  `PlannerDecision(action=PlannerAction.REPORT, rationale=f"[loop guard] {guard}")`
  and `break`.
- **line 169** — `decision = await self.strategy.decide(state)`, i.e. the LLM
  planner, is only reached when no guard fired.

So at the iteration where a run terminates, **the LLM planner is not consulted at
all.** The guard writes a synthetic decision into `run.planner_decision`, and that
synthetic decision is what lands in the trajectory JSON. Every terminating
decision in the recorded runs carries the tell:

```
resnet34_ref iter1  rationale: '[loop guard] train/val gap 0.007 <= threshold 0.06 — gap closed.'
search_winner iter1 rationale: '[loop guard] train/val gap 0.027 <= threshold 0.06 — gap closed.'
```

Read naively — and the trajectory format invites this — those look like the agent
deciding it was finished. They are a threshold comparison in six lines of
imperative code.

E1 shows the guard is doing real work rather than rubber-stamping what the
planner would have said anyway. Given the **already-converged** state
(`resnet34_ref` iteration 1, gap 0.0066), the planner asked directly chose to
**keep training in 20 of 20 trials**, adding augmentation and early stopping to a
model whose gap was already closed. It never once said stop.

The honest claim is: *the loop terminates sensibly, and the LLM planner is not
the reason it does.* Any evaluation that reads `planner_decision` from a
trajectory and attributes it to the agent is measuring the guard.

---

## 1. E1 — planner decision reliability under repeated identical trials

Fixed observation, 20 trials, fresh sampling each time. Four states. Two are
**verbatim recorded** observations; two are **constructed probes** assembled from
recorded metrics, because no recorded run ever produced an underfitting or
collapsed state. Both are labelled in the artifact, with exact numbers so a
reader can rebuild them.

| State | Provenance | train / val / gap / AUC |
|---|---|---|
| overfit | **VERBATIM RECORDED** — `paper_run` `resnet34_ref` iteration 0 | 0.9839 / 0.7517 / 0.2322 / 0.8922 |
| healthy | **VERBATIM RECORDED** — `paper_run` `resnet34_ref` iteration 1 | 0.8233 / 0.8167 / 0.0066 / 0.9283 |
| underfit | **CONSTRUCTED PROBE** — val, AUC and confusion verbatim from `ai_scientist_run` tuning iteration 0; train set to 0.6500 so both are low with no gap | 0.6500 / 0.6294 / 0.0206 / 0.8068 |
| collapsed | **CONSTRUCTED PROBE** — train/val from the loop-guard regression (chance for 3 classes); confusion built consistent with chance | 0.3740 / 0.3320 / 0.0420 / 0.5100 |

### Result: the action distribution is degenerate

**`action=train` in 80 of 80 trials. Action entropy 0.000 bits on every state.**
Not once did the planner return `report`, `stop`, `design_model` or `simulate` —
on any state, including the collapsed one.

The only thing that varies is which knob it turns:

| State | Modal knob set | Distribution | Knob-set entropy |
|---|---|---|---|
| overfit | `augment` + `early_stop_patience` | 18/20; `augment` alone 2/20 | 0.469 bits |
| healthy | `augment` + `early_stop_patience` | 15/20; `augment` alone 5/20 | 0.811 bits |
| underfit | `augment` + `early_stop_patience` | 15/20; `augment` 4/20; `early_stop`+`lr` 1/20 | 0.992 bits |
| collapsed | `early_stop_patience` + `learning_rate` | 9/20; `augment`+`early_stop` 7/20; `augment` 3/20; `lr` 1/20 | 1.675 bits |

### Result: consistent, but not diagnostic

`augment + early_stop_patience` is the modal response to overfitting (18/20),
health (15/20) **and underfitting (15/20) alike**. The planner is highly
reproducible and largely state-independent.

On the underfit probe — train 0.6500, val 0.6294, essentially no gap, i.e. a
model that has not learned enough — **19 of 20 trials added regularization**.
That is the wrong direction: augmentation and early stopping make an
underfitting model worse. Exactly one trial raised the learning rate.

Only the collapsed probe shifts the distribution meaningfully (learning rate
appears in 10/20), and even there the planner never proposes redesigning the
model or stopping; it proposes another training run at chance-level accuracy.

I score decisions with a coarse rule (`ok` / `noop` / `harmful`) recorded per
trial: 20/20 `harmful` on healthy (churn on a converged model), 19/20 `harmful`
on underfit (regularizing an underfit model). The rule is a convenience for
tabulation, not ground truth — every rationale string is kept in the artifact so
the labels can be re-read by hand.

---

## 2. E2 — judge ranking stability

Ten architecture candidates taken verbatim from the pilot's `experiment_b`
search, identical prompt, 20 trials.

- **Mean pairwise Kendall tau 0.927** over all 190 ranking pairs
  (min 0.778, max 1.000, sd 0.051).
- 8 distinct rankings in 20 trials.
- The bottom four (`cnn_s3_small`, `resnet_s2_tiny`, `cnn_s2_minimal`,
  `resnet_s4_large`) come out in **identical order every single trial**.
- Top-4 membership — the cut that matters, since only the top-4 are trained:
  `resnet_s3_narrow` 20/20, `resnet_s3_deeper` 20/20, `resnet_s2_wide` 19/20,
  `cnn_s3_wide` 17/20, `cnn_s4_mid` 2/20, `resnet_s4_balanced` 2/20.
- Rank 1 alternates between two candidates: `resnet_s3_deeper` 12/20,
  `resnet_s3_narrow` 8/20.

The judge is stable. Anyone who ran only this experiment would conclude the
pre-training filter is reliable.

---

## 3. E3 — does the judge's ranking predict the outcome?

For every recorded search that has both a judge ranking and round-1 shortlist
results, correlate the judge's pre-training ranking against the post-training
validation accuracy of the candidates it shortlisted. Three runs, four trained
candidates each, **12 data points**.

| Run | Kendall tau | Spearman | Judge's #1 was empirically best? |
|---|---|---|---|
| `paper_run` | **−0.667** | −0.800 | **No** |
| `ai_scientist_run` | +0.667 | +0.800 | Yes |
| `expB_pilot` | **+1.000** | +1.000 | Yes |

The sign is not consistent. Mean tau +0.333, but with n=4 per run and 3 runs
this cannot be distinguished from zero in either direction.

The bigger problem is structural: **only the judge-selected top-4 were ever
trained**, so this correlation is restricted to the set the judge already liked.
The 5–6 candidates it rejected in each run have no empirical outcome at all.
That is what E3b fixes.

---

## 4. E3b — train the architectures the judge rejected

Run on `paper_run`, deliberately: it is the search where the judge was
anti-correlated, so it is the stress test. The six candidates the judge rejected
were trained under exactly the shortlist protocol — `train_small` (1,800 images),
val 1,800, 10 epochs, batch 32, seed 0, same backends. 287 s total.

| Candidate | Judge rank | Status | Val acc | Params |
|---|---|---|---|---|
| `cnn_medium_s4` | #4 | shortlisted | **0.7733** | 2.539 M |
| `resnet_mediumplus_s4` | #5 | **rejected** | **0.7517** | 4.735 M |
| `resnet_large_s4` | #10 | **rejected** | **0.7417** | 31.46 M |
| `resnet_small_s4` | #2 | shortlisted | 0.7189 | 2.797 M |
| `cnn_small_deep_s4` | #3 | shortlisted | 0.7022 | 1.911 M |
| `resnet_deep_narrow_s4` | #6 | **rejected** | 0.6589 | 3.34 M |
| `cnn_wide_s4` | #7 | **rejected** | 0.5233 | 4.651 M |
| `resnet_wide_shallow_s3` | **#1** | shortlisted | **0.5211** | 1.227 M |
| `resnet_tiny_s3` | #8 | **rejected** | 0.3850 | 0.078 M |
| `cnn_tiny_s3` | #9 | **rejected** | 0.3444 | 0.024 M |

Read the top and bottom of that table together:

- **The judge's #1 pick ranks 8th of 10 empirically.** `resnet_wide_shallow_s3`
  (val 0.5211) was beaten by four of the six candidates the judge discarded.
- **The judge's #10 pick — its last-ranked candidate — ranks 3rd of 10.**
  `resnet_large_s4` (0.7417) beat three of the four the judge shortlisted. The
  judge's stated reason for burying it was over-capacity overfitting; at 31.46 M
  params on 1,800 images its gap was +0.0866, smaller than three shortlisted
  models'.
- No rejected candidate beat the best shortlisted one, so the search's eventual
  winner survived. But it survived by one slot: **`cnn_medium_s4` was the judge's
  #4, the last candidate to make the top-4 cut.** With `top_k=3` the search
  would have discarded its own eventual winner.

With all ten candidates trained, the unrestricted correlation is **Kendall tau
+0.244, Spearman +0.236** — weakly positive, driven almost entirely by the judge
correctly burying the two sub-0.1 M models that genuinely underfit. Above that
floor the ranking carries little signal.

**Not run: the `expB_pilot` rejected set.** It is the search where the judge
scored a perfect tau +1.000, so it would test the judge at its best rather than
stress it, and at 7,200 images it costs ~23 minutes. Skipped deliberately; the
`paper_run` result is the informative one.

---

## 5. E4 — the low-fidelity ranking the search actually depends on

The search ranks its shortlist on a 10-epoch training run and promotes the
winner. E4 retrains the same four candidates at the full 25-epoch budget, same
data, same protocol, seed 0.

| Candidate | Judge rank | Val @10 epochs | Val @25 epochs |
|---|---|---|---|
| `cnn_small_deep_s4` | #3 | 0.7022 | **0.8433** |
| `cnn_medium_s4` | #4 | **0.7733** | 0.8233 |
| `resnet_small_s4` | #2 | 0.7189 | 0.6944 |
| `resnet_wide_shallow_s3` | #1 | 0.5211 | 0.4972 |

- **The 10-epoch winner is not the 25-epoch winner.** The short run promoted
  `cnn_medium_s4`; at full budget `cnn_small_deep_s4` is better by 2.0 points
  (0.8433 vs 0.8233).
- Kendall tau between the two rankings is **+0.333** — the short proxy preserves
  the broad ordering (both ResNets stay at the bottom) but inverts the pair that
  decides the outcome.

### Replicated at 7,200 images — the winner flips again

Same comparison on the pilot's `experiment_b` shortlist (7,200 train images,
25 epochs, 2,160 s).

| Candidate | Judge | Val @10 ep | Val @25 ep | Δ | Params |
|---|---|---|---|---|---|
| `resnet_s2_wide_d3-3_w64-128` | #2 | 0.8806 | **0.8983** | +0.0178 | 1.047 M |
| `resnet_s3_deeper_d4-4-4_w32-64-128` | #1 | **0.8856** | 0.8761 | −0.0094 | 1.472 M |
| `resnet_s3_narrow_d2-2-2_w24-48-96` | #3 | 0.8589 | 0.8389 | −0.0200 | 0.393 M |
| `cnn_s3_wide_d2-3-3_w64-128-256` | #4 | 0.7917 | 0.8128 | +0.0211 | 1.884 M |

**The winner flips here too**: the short run promoted `resnet_s3_deeper`, but at
full budget `resnet_s2_wide` leads by 2.2 points. Kendall tau +0.667 — the short
proxy again preserves the tail and inverts the top.

This one compounds. The pilot promoted `resnet_s3_deeper_d4-4-4` and then spent
its whole refinement round producing variants of it, settling on
`d5-4-4` (1.491 M) as Experiment B's final architecture. At full budget the
better model is `resnet_s2_wide` — **smaller (1.047 M), never refined, and
2.2 points better than the parent the refinement round was built on.** The
refinement stage optimised the wrong parent.

### Correction to a claim I made from the first run alone

From `paper_run` I wrote that the proxy is "biased by family" — both CNNs
improved, both ResNets degraded. With both searches in, that is *directionally*
right but not clean: across the eight retrained candidates, **all three CNNs
improved** (+0.141, +0.050, +0.021) and **four of five ResNets got worse**
(−0.024, −0.024, −0.020, −0.009). The exception is `resnet_s2_wide` (+0.018) —
and it is precisely the full-budget winner. So a "prefer CNNs when extrapolating
short runs" heuristic would not have rescued either search. State the effect as a
tendency with a named counterexample, not a rule.

This matters beyond the search. **`cnn_medium_s4` is the architecture the paper
run's headline is built on** — the "+6.1 points test accuracy over resnet34"
result. E4 says it was not the best model in its own shortlist; the search's
cheap ranking step picked the runner-up, and nothing downstream ever revisited
that.

The limitation we previously only declared ("the shortlist is ranked on a short
run") is now measured: **it changes the winner, in both searches tested.**

---

## 6. What this adds up to

- The loop's most defensible decision — when to stop — **is not made by the
  agent** (§0), and the planner asked directly would have made it wrong 20/20
  times.
- The planner is **highly reproducible and barely diagnostic** (§1): the same
  two knobs for overfitting, health and underfitting alike, with regularization
  applied to an underfitting model in 19/20 trials.
- The judge is **stable without being valid** (§2 vs §3/§3b): tau 0.927 across
  repeated trials, tau +0.244 against reality, with its top pick ranking 8th of
  10 and its last pick 3rd.
- The ranking signal the search promotes on **disagrees with the full-budget
  measurement it proxies** (§4). Tested on two independent searches, the
  10-epoch winner was the 25-epoch winner **neither time** — and in the pilot the
  refinement round then spent its entire budget on the wrong parent.

None of this says the pipeline does not work — it produces small, accurate
architectures, and the guard keeps it honest. It says the *agentic* parts of it
carry less of the load than the trajectory logs imply, and that an evaluation
reading those logs at face value would credit the LLM for decisions made by a
threshold and by a short training run.

### Reproducing

```bash
set -a; . ./.env; set +a          # OPENAI_API_KEY, DLENS_LLM=gpt-5.2

.venv/bin/python scripts/run_planner_reliability.py \
    --out docs/agent_reliability/e1_planner.json --trials 20

.venv/bin/python scripts/run_judge_reliability.py \
    --out docs/agent_reliability/e2_judge.json --trials 20

.venv/bin/python scripts/run_search_validity.py --mode e3b \
    --source docs/paper_run.json --data-root ~/Personal/GSoC/deeplense_data/model1 \
    --train-dir train_small --val-dir val --epochs 10 \
    --out docs/agent_reliability/e3b_paper_run.json

.venv/bin/python scripts/run_search_validity.py --mode e4 \
    --source docs/paper_run.json --data-root ~/Personal/GSoC/deeplense_data/model1 \
    --train-dir train_small --val-dir val --epochs 25 \
    --out docs/agent_reliability/e4_paper_run.json
```

E3 is pure analysis of recorded runs (`e3_judge_predictiveness.json`); no
re-running required.
