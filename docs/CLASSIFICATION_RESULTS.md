# Real end-to-end classification results — DeepLense Model_I

**Date: 2026-07-20** · Runner: `scripts/run_classification.py` · Raw output: `pipeline_results.json` (with the dataset, not committed)

First fully real run of the AI-Scientist pipeline: real DeepLense data, agents reasoning
on gpt-5.2, real CNN training with torch — **no mocks anywhere in the run**.

## Dataset

The official Google Drive datasets are quota-blocked for programmatic download, so I
generated the dataset with **DeepLenseSim's own Model_I recipes** running real
**lenstronomy 1.9.2 + pyHalo** (era-pinned) in the Docker sandbox image:

- **9,000 images** — 3,000 per class (`no_sub`, `cdm`, `axion`), 150×150 px, single
  channel, ~809 MB (float32 `.npy`)
- axion masses sampled 10^U(−24, −22) per image, exactly as `sim_axion.py` does
- split **7,200 train / 1,800 val** (80/20, seeded)

## Pipeline (all four agents, live)

| Stage | Agent (gpt-5.2) | What actually ran | Time |
|---|---|---|---|
| 1 | ModelDesignAgent | recommended **resnet34**, lr 3e-4, batch 32, 30 epochs, cross-entropy | 9 s |
| 2 | TrainAgent | real training (torch, **MPS**), from-scratch ResNet-34 built from the spec | 14.7 min |
| 3 | InferAgent | real predictions on the 1,800 held-out images | 9 s |
| 4 | AnalysisAgent | deterministic metrics (tool-computed, not LLM-transcribed) | 8 s |

## Results (held-out val, 1,800 images)

| Metric | Value |
|---|---|
| **Accuracy** | **0.8367** |
| **Macro AUC (OvR)** | **0.9429** |
| Train accuracy (final epoch) | 0.9971 |

Per class:

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| no_sub | 0.956 | 0.933 | **0.944** | 600 |
| cdm | 0.755 | 0.762 | 0.758 | 600 |
| axion | 0.803 | 0.815 | 0.809 | 600 |

Confusion matrix (rows = true, cols = predicted `no_sub, cdm, axion`):

```
no_sub  [560,  39,   1]
cdm     [ 24, 457, 119]
axion   [  2, 109, 489]
```

## Reading of the results

- **Substructure detection is nearly solved** (no_sub F1 0.944); the errors concentrate
  in **distinguishing the substructure type** — cdm↔axion account for 228 of the 294
  mistakes. That is the physically hard problem, consistent with the DeepLense papers.
- The train/val gap (99.7% vs 83.7%) says the baseline overfits by epoch ~25 —
  headroom for augmentation, early stopping, or the physics-informed architectures the
  experiment planner is meant to explore. That is exactly the iteration loop this
  pipeline exists to automate.
- Numbers come from each agent's **tool-computed `last_result`** (deps-backed), never
  from LLM-transcribed text.

## Reproduce

```bash
# 1. generate data (Docker image from sandbox/, era-pinned lenstronomy/pyHalo stack)
docker run --rm -v $DATA:/work -w /work dlens-lenstronomy:latest python scripts/gen_model1_dataset.py no_sub 3000 raw   # + cdm, axion
# 2. split
uv run python scripts/prepare_deeplense_dataset.py --src $DATA/raw --out $DATA/model1 --classes no_sub,cdm,axion
# 3. run (agents on gpt-5.2; training local torch)
OPENAI_API_KEY=... uv run python scripts/run_classification.py --data-root $DATA/model1 --model gpt-5.2
```
