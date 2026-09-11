"""Nested-CV Optuna tuning for boosting and random-forest benchmark variants.

This script reuses the fixed 500-row data caches and the exact outer splits from
the completed suite. For every outer fold, Optuna sees only that fold's training
rows through an inner 3-fold stratified CV. The held-out outer fold stays unseen
until a tuned configuration has been chosen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from build_multidataset_report import build_multidataset_report
from multidataset_config import DATASETS, SAMPLE_SIZE, SAMPLING_SEED
from run_benchmark import (
    DATA_DIR,
    PROJECT_DIR,
    classification_metrics,
    get_feature_groups,
    one_hot_preprocessor,
    package_versions,
    write_json,
)
from run_multidataset_benchmark import MODEL_ORDER, load_dataset, overall_summary, summarize


RESULTS_DIR = PROJECT_DIR / "results"
TUNED_MODELS = ("LightGBM Tuned", "XGBoost Tuned", "Random Forest Tuned")
INNER_FOLDS = 1
TUNING_TRIALS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trials", type=int, default=TUNING_TRIALS)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def selected_specs(argument: str) -> list[tuple[str, dict[str, Any]]]:
    keys = [key.strip() for key in argument.split(",") if key.strip()]
    unknown = sorted(set(keys).difference(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}")
    return [(key, DATASETS[key]) for key in keys]


def model_name(base: str) -> str:
    return f"{base} Tuned"


def make_pipeline(
    base_model: str,
    parameters: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
    seed: int,
) -> Pipeline:
    preprocessor = one_hot_preprocessor(numeric_columns, categorical_columns, scale_numeric=False)
    if base_model == "LightGBM":
        estimator = LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            **parameters,
        )
    elif base_model == "XGBoost":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            **parameters,
        )
    elif base_model == "Random Forest":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=1, **parameters)
    else:
        raise ValueError(f"Unknown base model: {base_model}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def suggest_parameters(base_model: str, trial: optuna.Trial) -> dict[str, Any]:
    """Small, high-impact spaces that suit 400-row outer-train samples.

    The ranges avoid enormous forests, deep trees, and vanishingly tiny learning
    rates: those choices tend to overfit or waste trials on this suite.
    """
    profile = trial.suggest_categorical("profile", ["regularized", "balanced", "expressive", "subsampled"])
    return curated_profiles(base_model)[profile]
    raise ValueError(f"Unknown base model: {base_model}")


def curated_profiles(base_model: str) -> dict[str, dict[str, Any]]:
    """Four coherent hypotheses, not an arbitrary Cartesian parameter grid."""
    if base_model == "LightGBM":
        return {
            "regularized": {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 30, "reg_lambda": 5.0, "feature_fraction": 0.85},
            "balanced": {"n_estimators": 200, "learning_rate": 0.07, "num_leaves": 15, "min_child_samples": 20, "reg_lambda": 1.0, "feature_fraction": 1.0},
            "expressive": {"n_estimators": 150, "learning_rate": 0.08, "num_leaves": 31, "min_child_samples": 15, "reg_lambda": 0.3, "feature_fraction": 0.85},
            "subsampled": {"n_estimators": 300, "learning_rate": 0.04, "num_leaves": 15, "min_child_samples": 20, "reg_lambda": 2.0, "feature_fraction": 0.7},
        }
    if base_model == "XGBoost":
        return {
            "regularized": {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5.0},
            "balanced": {"n_estimators": 200, "learning_rate": 0.07, "max_depth": 3, "min_child_weight": 3, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0},
            "expressive": {"n_estimators": 150, "learning_rate": 0.08, "max_depth": 4, "min_child_weight": 1, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 0.3},
            "subsampled": {"n_estimators": 300, "learning_rate": 0.04, "max_depth": 3, "min_child_weight": 3, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 2.0},
        }
    if base_model == "Random Forest":
        return {
            "regularized": {"n_estimators": 250, "max_depth": 6, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": None},
            "balanced": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": None},
            "expressive": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": 0.8, "class_weight": None},
            "subsampled": {"n_estimators": 250, "max_depth": 10, "min_samples_leaf": 2, "max_features": 0.5, "class_weight": "balanced"},
        }
    raise ValueError(f"Unknown base model: {base_model}")


def tune_parameters(
    *,
    base_model: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    seed: int,
    trials: int,
) -> tuple[dict[str, Any], str, float, float, int]:
    """Choose a profile using an inner validation split only."""
    started = time.perf_counter()
    inner_train, inner_validation = train_test_split(
        np.arange(len(train_features)),
        test_size=0.25,
        random_state=seed,
        stratify=train_target,
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(base_model, trial)
        pipeline = make_pipeline(base_model, parameters, numeric_columns, categorical_columns, seed)
        pipeline.fit(train_features.iloc[inner_train], train_target.iloc[inner_train])
        probabilities = pipeline.predict_proba(train_features.iloc[inner_validation])[:, 1]
        return float(roc_auc_score(train_target.iloc[inner_validation], probabilities))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler({"profile": list(curated_profiles(base_model))}, seed=seed),
    )
    study.optimize(objective, n_trials=trials, n_jobs=4, show_progress_bar=False)
    best_profile = str(study.best_params["profile"])
    return (
        dict(curated_profiles(base_model)[best_profile]),
        best_profile,
        float(study.best_value),
        time.perf_counter() - started,
        len(study.trials),
    )


def tune_outer_fold(
    *,
    base_model: str,
    dataset_key: str,
    fold: int,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    train_target: pd.Series,
    validation_target: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    seed: int,
    trials: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        parameters, selected_profile, inner_score, tuning_seconds, completed_trials = tune_parameters(
            base_model=base_model,
            train_features=train_features,
            train_target=train_target,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            seed=seed + fold,
            trials=trials,
        )
        pipeline = make_pipeline(base_model, parameters, numeric_columns, categorical_columns, seed)
        fit_started = time.perf_counter()
        pipeline.fit(train_features, train_target)
        fit_seconds = time.perf_counter() - fit_started
        train_started = time.perf_counter()
        train_probabilities = pipeline.predict_proba(train_features)[:, 1]
        train_prediction_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation_probabilities = pipeline.predict_proba(validation_features)[:, 1]
        validation_prediction_seconds = time.perf_counter() - validation_started
        train_metrics = classification_metrics(train_target, train_probabilities)
        validation_metrics = classification_metrics(validation_target, validation_probabilities)
        return {
            "status": "success", "dataset_key": dataset_key, "model": model_name(base_model), "fold": fold,
            "train_rows": int(len(train_features)), "validation_rows": int(len(validation_features)),
            "train_metrics": train_metrics, "validation_metrics": validation_metrics,
            "overfitting_gap_roc_auc": train_metrics["roc_auc"] - validation_metrics["roc_auc"],
            "tuning": {"inner_validation_fraction": 0.25, "trials_requested": trials, "trials_completed": completed_trials, "selected_profile": selected_profile, "best_inner_validation_roc_auc": inner_score, "best_parameters": parameters},
            "timing_seconds": {"tune": tuning_seconds, "fit": fit_seconds, "predict_train": train_prediction_seconds, "predict_validation": validation_prediction_seconds, "total": time.perf_counter() - started},
        }
    except Exception as error:
        return {"status": "failed", "dataset_key": dataset_key, "model": model_name(base_model), "fold": fold, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "timing_seconds": {"total": time.perf_counter() - started}}


def signature(dataset_keys: list[str], args: argparse.Namespace, packages: dict[str, str]) -> str:
    payload = {"runner": "parallel_curated_optuna_v4", "datasets": dataset_keys, "sample_size": SAMPLE_SIZE, "sampling_seed": SAMPLING_SEED, "outer_folds": args.folds, "inner_validation_fraction": 0.25, "seed": args.seed, "trials": args.trials, "packages": packages}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_combined_result(
    baseline: dict[str, Any], tuned_records: list[dict[str, Any]], tuning_metadata: dict[str, Any], run_id: str, packages: dict[str, str], invocation_seconds: float, checkpoint_path: Path, trials: int,
) -> dict[str, Any]:
    by_dataset: list[dict[str, Any]] = []
    for baseline_dataset in baseline["dataset_summaries"]:
        key = baseline_dataset["dataset_key"]
        merged_records = [record for record in baseline["fold_results"] if record["dataset_key"] == key]
        merged_records.extend(record for record in tuned_records if record["dataset_key"] == key)
        by_dataset.append({
            "dataset_key": key,
            "dataset": baseline_dataset["dataset"],
            "model_summaries": summarize_with_tuned(merged_records),
            "tabpfn_batch_profile_seconds": baseline_dataset.get("tabpfn_batch_profile_seconds", {}),
        })
    all_records = list(baseline["fold_results"]) + tuned_records
    result = dict(baseline)
    tuned_configurations = {
        model_name(base_model): {
            "selection": "Optuna GridSampler selects one of four curated profiles using only a stratified 25% inner validation split from each outer training fold.",
            "profiles": curated_profiles(base_model),
            "parallel_trials": 4,
            "estimator_threads_per_trial": 1,
        }
        for base_model in ("LightGBM", "XGBoost", "Random Forest")
    }
    result.update({
        "schema_version": 2,
        "status": "complete",
        "tuning_run_signature": run_id,
        "generated_at_utc": now_iso(),
        "dataset_summaries": by_dataset,
        "overall_model_summary": overall_summary_with_tuned(by_dataset),
        "fold_results": all_records,
        "tuning_metadata": tuning_metadata,
        "environment": {**baseline["environment"], "packages": packages},
        "model_configurations": {**baseline["model_configurations"], **tuned_configurations},
        "profiling_seconds": {**baseline["profiling_seconds"], "tuned_model_work": float(sum(record["timing_seconds"].get("total", 0.0) for record in tuned_records)), "tuned_invocation_seconds": invocation_seconds},
        "outputs": {"json": "results/multidataset_tuned_benchmark_results.json", "html_report": "results/multidataset_tuned_benchmark_report.html", "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR))},
    })
    result["evaluation_protocol"] = {
        **baseline["evaluation_protocol"],
        "tuned_model_protocol": f"Nested CV: for each outer fold, Optuna compares {trials} curated, model-specific profiles using one stratified 25% inner validation split from that outer fold's training data. The outer held-out fold is evaluated once after selection. Profile trials run concurrently with single-threaded estimators.",
    }
    return result


def summarize_with_tuned(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original_order = list(MODEL_ORDER) + list(TUNED_MODELS)
    summaries = []
    for name in original_order:
        successful = [record for record in records if record["model"] == name and record["status"] == "success"]
        failed = [record for record in records if record["model"] == name and record["status"] != "success"]
        item: dict[str, Any] = {"model": name, "successful_folds": len(successful), "failed_folds": len(failed), "validation_metrics": {}, "train_metrics": {}, "overfitting_gap_roc_auc": {}, "timing_seconds": {}}
        if successful:
            for split_key in ("validation_metrics", "train_metrics"):
                for metric in ("roc_auc", "average_precision", "accuracy", "f1", "precision", "recall", "specificity", "log_loss", "brier_score"):
                    values = np.asarray([record[split_key][metric] for record in successful])
                    item[split_key][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
            gaps = np.asarray([record["overfitting_gap_roc_auc"] for record in successful])
            item["overfitting_gap_roc_auc"] = {"mean": float(gaps.mean()), "std": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0}
            for timing in ("tune", "fit", "predict_train", "predict_validation", "total"):
                values = np.asarray([record["timing_seconds"].get(timing, 0.0) for record in successful])
                item["timing_seconds"][timing] = {"mean": float(values.mean()), "total": float(values.sum())}
        summaries.append(item)
    return sorted(summaries, key=lambda item: item["validation_metrics"].get("roc_auc", {}).get("mean", -float("inf")), reverse=True)


def overall_summary_with_tuned(dataset_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = list(MODEL_ORDER) + list(TUNED_MODELS)
    output = []
    for name in names:
        scores: list[float] = []
        gaps: list[float] = []
        ranks: list[int] = []
        total_time = 0.0
        for dataset in dataset_summaries:
            ranked = [item for item in dataset["model_summaries"] if item["validation_metrics"].get("roc_auc")]
            for rank, item in enumerate(ranked, start=1):
                if item["model"] == name:
                    scores.append(item["validation_metrics"]["roc_auc"]["mean"])
                    gaps.append(item["overfitting_gap_roc_auc"]["mean"])
                    ranks.append(rank)
                    total_time += item["timing_seconds"]["total"]["total"]
        output.append({"model": name, "datasets_completed": len(scores), "validation_roc_auc": {"mean": float(np.mean(scores)) if scores else None, "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0}, "overfitting_gap_roc_auc": {"mean": float(np.mean(gaps)) if gaps else None, "std": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0}, "mean_rank": float(np.mean(ranks)) if ranks else None, "total_time_seconds": total_time})
    return sorted(output, key=lambda item: item["mean_rank"] if item["mean_rank"] is not None else float("inf"))


def main() -> None:
    args = parse_args()
    if args.folds != 5:
        raise ValueError("The cached suite uses five fixed outer folds.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.perf_counter()
    baseline_path = RESULTS_DIR / "multidataset_benchmark_results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    selected = selected_specs(args.datasets)
    packages = package_versions()
    packages["optuna"] = "4.8.0"
    run_id = signature([key for key, _ in selected], args, packages)
    checkpoint_path = RESULTS_DIR / "tuned_checkpoints" / f"{run_id}.json"
    if args.restart and checkpoint_path.exists():
        checkpoint_path.unlink()
    prior = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    record_map: dict[tuple[str, int, str], dict[str, Any]] = {(record["dataset_key"], int(record["fold"]), record["model"]): record for record in prior.get("fold_results", [])}

    def save_checkpoint(status: str = "in_progress") -> None:
        records = sorted(record_map.values(), key=lambda item: (item["dataset_key"], item["fold"], item["model"]))
        write_json(checkpoint_path, {"schema_version": 1, "status": status, "run_signature": run_id, "updated_at_utc": now_iso(), "progress": {"planned_evaluations": len(selected) * len(TUNED_MODELS) * args.folds, "successful_evaluations": sum(item["status"] == "success" for item in records)}, "fold_results": records})

    for dataset_key, spec in selected:
        features, target, _, _ = load_dataset(dataset_key, spec)
        numeric_columns, categorical_columns = get_feature_groups(features)
        outer_cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for base_model in ("LightGBM", "XGBoost", "Random Forest"):
            tuned_name = model_name(base_model)
            for fold, (train_indices, validation_indices) in enumerate(outer_cv.split(features, target), start=1):
                key = (dataset_key, fold, tuned_name)
                if record_map.get(key, {}).get("status") == "success":
                    print(f"{spec['name']} | Fold {fold} | {tuned_name} (resumed)", flush=True)
                    continue
                print(f"{spec['name']} | Fold {fold} | {tuned_name} ({args.trials} Optuna trials)", flush=True)
                record_map[key] = tune_outer_fold(
                    base_model=base_model, dataset_key=dataset_key, fold=fold,
                    train_features=features.iloc[train_indices], validation_features=features.iloc[validation_indices],
                    train_target=target.iloc[train_indices], validation_target=target.iloc[validation_indices],
                    numeric_columns=numeric_columns, categorical_columns=categorical_columns,
                    seed=args.seed, trials=args.trials,
                )
                save_checkpoint()

    tuned_records = list(record_map.values())
    tuning_metadata = {"optimizer": "Optuna GridSampler", "trials_per_outer_fold": args.trials, "inner_validation_fraction": 0.25, "objective": "Inner validation ROC-AUC", "search_design": "Four curated, correlated profiles per model: regularized, balanced, expressive, and subsampled. Trial profiles run concurrently while each estimator remains single-threaded."}
    combined = build_combined_result(baseline, tuned_records, tuning_metadata, run_id, packages, time.perf_counter() - started, checkpoint_path, args.trials)
    save_checkpoint("complete")
    json_path = RESULTS_DIR / "multidataset_tuned_benchmark_results.json"
    html_path = RESULTS_DIR / "multidataset_tuned_benchmark_report.html"
    write_json(json_path, combined)
    build_multidataset_report(combined, html_path)
    print(f"\nTuned benchmark complete. JSON: {json_path}; report: {html_path}")


if __name__ == "__main__":
    main()
