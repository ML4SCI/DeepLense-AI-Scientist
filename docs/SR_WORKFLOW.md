# Super Resolution AI Scientist

This workflow extends DLens with x2 grayscale super resolution. It trains and compares candidate models on paired images, checks them against an iterative autoencoder baseline trained on the same data, and writes an experiment report. The validation set chooses models; the test set stays held out.

## Install

Use Python 3.12 or later from the repository root:

```bash
uv sync
```

## Data

Supply matching `.npy` files under these directories:

```text
splits/train/LR/   splits/train/HR/
splits/val/LR/     splits/val/HR/
```

The filename of each LR image must match its HR image. Images must be single channel, and HR must be twice the LR height and width. Use the same `train` and `val` splits for both stages. Dataset arrays are not included in this source package.

## 1. Train and measure the baseline

The Kaggle workflow uses the compact iterative autoencoder from the earlier SR run. Train it for 50 epochs with width 32, batch size 4, and 200 training batches per epoch. A fixed 128-pair validation subset selects the checkpoint. The best checkpoint then generates PSNR and SSIM scores on **every** `val` pair using eight refinement steps.

```bash
python baselines/atal_gupta/iterative_train.py \
  --splits-root /path/to/splits \
  --epochs 50 \
  --out-dir /path/to/iterative_baseline

python baselines/atal_gupta/iterative_evaluate.py \
  --splits-root /path/to/splits \
  --checkpoint /path/to/iterative_baseline/best_model.pt \
  --out-dir /path/to/iterative_baseline
```

Keep the checkpoint and metrics together outside Git. Resume interrupted training with `--resume /path/to/iterative_baseline/last.pt` and the same settings. This model is adapted from the project's earlier `iterative-auto-encoder-kaggle.ipynb`.

Validation caps per-image PSNR at 150 dB so exact matches have finite values in CSV summaries and rankings. The standalone bicubic command accepts `--lr-size N` for square inputs other than 75×75. The iterative comparison script reads image sizes from the split and uses the inference steps recorded in the baseline summary.

## 2. Run the scientist

```bash
python scripts/run_sr_scientist.py \
  --splits-root /path/to/splits \
  --baseline-dir /path/to/iterative_baseline \
  --out-dir /path/to/results \
  --train-split train \
  --device auto
```

The runner verifies validation IDs, order, metric settings, and per-image bicubic scores before using the selected baseline. By default, the planner and selection are deterministic. To enable LLM proposals and judging, configure `DLENS_MODEL_PROVIDER` and `DLENS_MODEL` (see `.env.example`) and add `--use-llm`. The runner saves its state as it progresses; use `--resume` with the same output directory after an interruption. Run one round at a time with `--rounds-this-run 1` if desired.

With `--use-llm`, the generator proposes `screening_epochs` for each architecture before promotion. After screening, an epoch planner chooses additional promotion epochs for each selected architecture. It then chooses additional tuning epochs before every tuning run, including the first. Set the allowed ranges with `--min-search-epochs`/`--search-epochs`, `--min-promotion-epochs`/`--promotion-epochs`, and `--min-tune-epochs`/`--tune-epochs`. The maximum flags retain their fixed-budget behavior without an LLM. Epoch choices and rationales are saved in the run state and report. Promotion and tuning continue from checkpoints, so their epoch counts are additional training epochs.

The default LLM ranges are 1–10 screening epochs, 1–30 additional promotion epochs, and 1–30 additional tuning epochs. The special-family promotion multiplier defaults to 1, so these ranges apply to every family; `--special-promotion-multiplier` can raise the promotion ceiling when explicitly set.

For an LR size other than 75×75, add `--lr-size N` to the scientist command. The baseline loader reads the paired image dimensions directly.

Create a report from a completed run:

```bash
python scripts/report_sr_scientist.py /path/to/results/sr_scientist_run.json
```

Run checks with `uv run pytest tests`.

## Code locations

- `src/dlens/`: models, data loader, metrics, training, selection, and agents
- `scripts/`: scientist runner, bicubic baseline runner, and report generator
- `baselines/atal_gupta/`: iterative autoencoder model, trainer, and evaluator
- `tests/`: focused code checks

Training datasets, checkpoints, experiment outputs, and cached files are excluded from Git.
