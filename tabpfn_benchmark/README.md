# TabPFN benchmark

This is a reproducible binary-classification benchmark of TabPFN, LightGBM,
XGBoost, Random Forest, and Logistic Regression on the OpenML **German Credit**
dataset (data ID 31). The data has 1,000 rows, 20 mixed-type features, and a
non-trivial `good`/`bad` credit-risk target.

The frozen dependency set requires Python 3.12 or newer; `uv` selects or
downloads a suitable interpreter automatically.

## Run

```powershell
cd tabpfn_benchmark
uv sync
uv run python run_benchmark.py
```

Outputs are written to `results/`:

- `benchmark_results.json` — all fold-level metrics, model configurations,
  timing information, environment versions, and aggregate results.
- `benchmark_report.html` — standalone HTML report, with dataset details at the
  top and sections for generalization, overfitting gaps, metrics, and timing.
- `checkpoints/<signature>.json` — atomically updated after every completed
  model-fold. Rerunning the same configuration resumes successful work instead
  of starting over.

Open the report on Windows with:

```powershell
Start-Process results\benchmark_report.html
```

The data is cached under `data/` after the first OpenML download. To download it
again, add `--force-download`.

The benchmark is idempotent for an identical dataset, package set, CV setup, and
model configuration. Use `--restart` only when you deliberately want to discard
that run's checkpoint and execute every model-fold again.

## Optimized five-dataset suite

```powershell
uv run --locked python run_multidataset_benchmark.py
```

This adds Adult Income, Bank Marketing, Spambase, and Phoneme to German Credit.
Each task is deterministically stratified-sampled to 500 rows when needed.
TabPFN uses `predict_proba_batched()` to score all five equally sized validation
folds in one fused call, then all five training folds in a second fused call for
the overfitting-gap calculation. Its report is written to
`results/multidataset_benchmark_report.html`.

## Compact nested tuning for tree models

```powershell
uv run --locked python run_tuned_benchmarks.py
```

This adds `LightGBM Tuned`, `XGBoost Tuned`, and `Random Forest Tuned` without
rerunning TabPFN or the existing baseline models. Each outer validation fold is
kept untouched: Optuna compares four curated profiles on a stratified 25% inner
validation split drawn only from that outer fold's training subset. Profiles run
concurrently with single-threaded model fits, keeping the search fast and
avoiding nested CPU oversubscription. The results are merged into
`results/multidataset_tuned_benchmark_report.html` and checkpointed after each
tuned model-fold.

## Evaluation protocol

- One shared, shuffled, 5-fold stratified outer cross-validation split (seed
  `2026`) is used for every model.
- The held-out fold is never used for tuning or threshold selection, so its
  ROC-AUC estimates generalization to unseen data.
- Overfitting gap is computed per fold as **training ROC-AUC − held-out ROC-AUC**.
  The report gives its mean and standard deviation alongside training and
  validation ROC-AUC.
- Other held-out metrics are average precision, accuracy, F1, precision, recall,
  specificity, log loss, and Brier score.
- Models have fixed, reported configurations; this is a controlled benchmark,
  not a hyperparameter-search leaderboard.
- Models run serially. This makes wall-clock profiles attributable and avoids CPU
  or GPU contention; LightGBM, XGBoost, and Random Forest each use one CPU thread.

## Credentials

The script optionally loads `.env` from this directory. Copy `.env.example` if
you need to provide `TABPFN_TOKEN` on a fresh machine. Existing cached TabPFN
weights are used when available. `.env`, downloaded data, environments, and
generated results are git-ignored.
