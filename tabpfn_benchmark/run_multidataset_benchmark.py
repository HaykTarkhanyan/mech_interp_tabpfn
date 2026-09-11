"""Run a checkpointed five-dataset benchmark with fused TabPFN CV inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier
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
    tabpfn_device,
    tabpfn_preprocessor,
    write_json,
)


RESULTS_DIR = PROJECT_DIR / "results"
MODEL_ORDER = ("TabPFN", "LightGBM", "XGBoost", "Random Forest", "Logistic Regression")
METRICS = ("roc_auc", "average_precision", "accuracy", "f1", "precision", "recall", "specificity", "log_loss", "brier_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated suite keys. Defaults to all five datasets.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--tabpfn-estimators",
        type=int,
        default=1,
        help="TabPFN ensemble members. CPU default is 1 for practical five-dataset throughput; use 8 for the library default.",
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def selected_dataset_specs(argument: str) -> list[tuple[str, dict[str, Any]]]:
    keys = [key.strip() for key in argument.split(",") if key.strip()]
    unknown = sorted(set(keys).difference(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}. Choices: {sorted(DATASETS)}")
    if len(set(keys)) != len(keys):
        raise ValueError("Each dataset may be selected only once.")
    return [(key, DATASETS[key]) for key in keys]


def load_dataset(key: str, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], float]:
    """Fetch, deterministically sample, and cache one OpenML binary task."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = DATA_DIR / f"{key}_sample_{SAMPLE_SIZE}.csv"
    metadata_path = DATA_DIR / f"{key}_sample_{SAMPLE_SIZE}.json"
    started = time.perf_counter()

    if sample_path.exists() and metadata_path.exists():
        frame = pd.read_csv(sample_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["load_source"] = "local sampled CSV cache"
    else:
        bunch = fetch_openml(data_id=spec["openml_data_id"], as_frame=True, parser="auto")
        raw = bunch.data.copy()
        raw["__target__"] = bunch.target.astype(str)
        original_rows = len(raw)
        if original_rows < SAMPLE_SIZE:
            raise ValueError(f"{key} has only {original_rows} rows; need at least {SAMPLE_SIZE}.")
        if original_rows > SAMPLE_SIZE:
            sample_indices, _ = train_test_split(
                np.arange(original_rows),
                train_size=SAMPLE_SIZE,
                random_state=SAMPLING_SEED,
                stratify=raw["__target__"].astype(str),
            )
            raw = raw.iloc[np.sort(sample_indices)].reset_index(drop=True)
            sampling = f"Stratified random sample of {SAMPLE_SIZE:,} from {original_rows:,} original rows (seed {SAMPLING_SEED})."
        else:
            sampling = f"Dataset already contains exactly {SAMPLE_SIZE:,} rows; no sampling applied."
        frame = raw
        metadata = {
            "key": key,
            "name": spec["name"],
            "description": spec["description"],
            "openml_data_id": spec["openml_data_id"],
            "source_url": f"https://www.openml.org/d/{spec['openml_data_id']}",
            "original_rows": original_rows,
            "sampling": sampling,
            "load_source": "OpenML download",
        }
        frame.to_csv(sample_path, index=False)
        write_json(metadata_path, metadata)

    target_text = frame.pop("__target__").astype(str).str.strip().str.lower()
    labels = sorted(target_text.unique().tolist())
    if len(labels) != 2:
        raise ValueError(f"{key} must be binary; found {labels}")
    desired_positive = str(spec["positive_label"]).lower()
    positive_label = desired_positive if desired_positive in labels else labels[-1]
    target = (target_text == positive_label).astype(int)
    numeric_columns, categorical_columns = get_feature_groups(frame)
    metadata.update(
        {
            "cache_path": str(sample_path.relative_to(PROJECT_DIR)),
            "rows": int(len(frame)),
            "raw_feature_count": int(frame.shape[1]),
            "numeric_feature_count": len(numeric_columns),
            "categorical_feature_count": len(categorical_columns),
            "positive_class": positive_label,
            "negative_class": next(label for label in labels if label != positive_label),
            "class_counts": {str(label): int(count) for label, count in Counter(target_text).items()},
            "missing_cells": int(frame.isna().sum().sum()),
            "feature_names": frame.columns.tolist(),
        }
    )
    return frame, target, metadata, time.perf_counter() - started


def positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    return probabilities[:, list(model.classes_).index(1)]


def classical_pipeline(
    model_name: str, numeric_columns: list[str], categorical_columns: list[str], seed: int
) -> tuple[Pipeline, dict[str, Any]]:
    if model_name == "LightGBM":
        model = LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=seed, n_jobs=1, verbosity=-1,
        )
        configuration = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "n_jobs": 1}
        scale_numeric = False
    elif model_name == "XGBoost":
        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, random_state=seed, n_jobs=1, verbosity=0,
        )
        configuration = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "tree_method": "hist", "n_jobs": 1}
        scale_numeric = False
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=1)
        configuration = {"n_estimators": 500, "min_samples_leaf": 2, "n_jobs": 1}
        scale_numeric = False
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=2_000, solver="lbfgs", random_state=seed)
        configuration = {"max_iter": 2_000, "solver": "lbfgs"}
        scale_numeric = True
    else:
        raise ValueError(f"Not a classical model: {model_name}")
    return (
        Pipeline(
            [
                ("preprocess", one_hot_preprocessor(numeric_columns, categorical_columns, scale_numeric=scale_numeric)),
                ("model", model),
            ]
        ),
        {
            "preprocessing": "Median imputation for numeric columns; most-frequent imputation plus one-hot encoding for categorical columns." + (" Numeric columns are standardized." if scale_numeric else ""),
            "random_state": seed,
            **configuration,
        },
    )


def run_classical_fold(
    *, model_name: str, numeric_columns: list[str], categorical_columns: list[str], seed: int,
    fold: int, train_features: pd.DataFrame, validation_features: pd.DataFrame,
    train_target: pd.Series, validation_target: pd.Series,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        model, _ = classical_pipeline(model_name, numeric_columns, categorical_columns, seed)
        fit_started = time.perf_counter()
        model.fit(train_features, train_target)
        fit_seconds = time.perf_counter() - fit_started
        train_started = time.perf_counter()
        train_probabilities = positive_probability(model, train_features)
        train_predict_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation_probabilities = positive_probability(model, validation_features)
        validation_predict_seconds = time.perf_counter() - validation_started
        train_metrics = classification_metrics(train_target, train_probabilities)
        validation_metrics = classification_metrics(validation_target, validation_probabilities)
        return success_record(
            model_name, fold, train_target, validation_target, train_metrics, validation_metrics,
            {"fit": fit_seconds, "predict_train": train_predict_seconds, "predict_validation": validation_predict_seconds, "total": time.perf_counter() - started},
        )
    except Exception as error:
        return failed_record(model_name, fold, error, started)


def success_record(
    model_name: str, fold: int, train_target: pd.Series, validation_target: pd.Series,
    train_metrics: dict[str, float], validation_metrics: dict[str, float], timing: dict[str, float],
) -> dict[str, Any]:
    return {
        "status": "success", "model": model_name, "fold": fold,
        "train_rows": int(len(train_target)), "validation_rows": int(len(validation_target)),
        "train_metrics": train_metrics, "validation_metrics": validation_metrics,
        "overfitting_gap_roc_auc": train_metrics["roc_auc"] - validation_metrics["roc_auc"],
        "timing_seconds": timing,
    }


def failed_record(model_name: str, fold: int, error: Exception, started: float) -> dict[str, Any]:
    return {
        "status": "failed", "model": model_name, "fold": fold,
        "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
        "timing_seconds": {"total": time.perf_counter() - started},
    }


def run_tabpfn_batched(
    *, features: pd.DataFrame, target: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]],
    numeric_columns: list[str], categorical_columns: list[str], seed: int, estimators: int, device: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Score all same-shaped CV folds in two fused TabPFN calls.

    One call scores the five held-out folds; the second scores the five training
    folds needed for the explicitly requested training-vs-validation gap.
    """
    started = time.perf_counter()
    training_arrays: list[np.ndarray] = []
    validation_arrays: list[np.ndarray] = []
    training_targets: list[np.ndarray] = []
    fold_inputs: list[tuple[pd.Series, pd.Series]] = []
    preprocess_seconds: list[float] = []
    for train_indices, validation_indices in folds:
        preprocessing_started = time.perf_counter()
        preprocessor = tabpfn_preprocessor(numeric_columns, categorical_columns)
        train_features = features.iloc[train_indices]
        validation_features = features.iloc[validation_indices]
        train_target = target.iloc[train_indices]
        validation_target = target.iloc[validation_indices]
        training_arrays.append(np.asarray(preprocessor.fit_transform(train_features), dtype=np.float64))
        validation_arrays.append(np.asarray(preprocessor.transform(validation_features), dtype=np.float64))
        training_targets.append(train_target.to_numpy(dtype=int))
        fold_inputs.append((train_target, validation_target))
        preprocess_seconds.append(time.perf_counter() - preprocessing_started)

    categories = list(range(len(numeric_columns), len(numeric_columns) + len(categorical_columns)))
    classifier = TabPFNClassifier(
        device=device, n_estimators=estimators, categorical_features_indices=categories or None,
        random_state=seed, n_preprocessing_jobs=1, show_progress_bar=False,
    )
    try:
        validation_started = time.perf_counter()
        validation_probabilities = classifier.predict_proba_batched(training_arrays, training_targets, validation_arrays)
        validation_seconds = time.perf_counter() - validation_started
        training_started = time.perf_counter()
        training_probabilities = classifier.predict_proba_batched(training_arrays, training_targets, training_arrays)
        training_seconds = time.perf_counter() - training_started
        records: list[dict[str, Any]] = []
        for index, (train_target, validation_target) in enumerate(fold_inputs):
            train_metrics = classification_metrics(train_target, np.asarray(training_probabilities[index, :, 1]))
            validation_metrics = classification_metrics(validation_target, np.asarray(validation_probabilities[index, :, 1]))
            records.append(
                success_record(
                    "TabPFN", index + 1, train_target, validation_target, train_metrics, validation_metrics,
                    {
                        "preprocess": preprocess_seconds[index],
                        "fit": 0.0,
                        "predict_train": training_seconds / len(folds),
                        "predict_validation": validation_seconds / len(folds),
                        "total": preprocess_seconds[index] + (training_seconds + validation_seconds) / len(folds),
                    },
                )
            )
        return records, {
            "preprocess_total": float(sum(preprocess_seconds)),
            "fused_train_prediction": training_seconds,
            "fused_validation_prediction": validation_seconds,
            "total": time.perf_counter() - started,
        }
    except Exception as error:
        return [failed_record("TabPFN", index + 1, error, started) for index in range(len(folds))], {"total": time.perf_counter() - started}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        successful = [record for record in records if record["model"] == model_name and record["status"] == "success"]
        failed = [record for record in records if record["model"] == model_name and record["status"] != "success"]
        summary: dict[str, Any] = {"model": model_name, "successful_folds": len(successful), "failed_folds": len(failed), "validation_metrics": {}, "train_metrics": {}, "overfitting_gap_roc_auc": {}, "timing_seconds": {}}
        if successful:
            for split_key in ("validation_metrics", "train_metrics"):
                for metric_name in METRICS:
                    values = np.asarray([record[split_key][metric_name] for record in successful])
                    summary[split_key][metric_name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
            gaps = np.asarray([record["overfitting_gap_roc_auc"] for record in successful])
            summary["overfitting_gap_roc_auc"] = {"mean": float(gaps.mean()), "std": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0}
            for timing_name in ("preprocess", "fit", "predict_train", "predict_validation", "total"):
                values = np.asarray([record["timing_seconds"].get(timing_name, 0.0) for record in successful])
                summary["timing_seconds"][timing_name] = {"mean": float(values.mean()), "total": float(values.sum())}
        summaries.append(summary)
    return sorted(summaries, key=lambda item: item["validation_metrics"].get("roc_auc", {}).get("mean", -float("inf")), reverse=True)


def overall_summary(dataset_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        scores: list[float] = []
        gaps: list[float] = []
        ranks: list[int] = []
        total_time = 0.0
        for dataset_summary in dataset_summaries:
            summaries = dataset_summary["model_summaries"]
            successful = [item for item in summaries if item["validation_metrics"].get("roc_auc")]
            ranked = sorted(successful, key=lambda item: item["validation_metrics"]["roc_auc"]["mean"], reverse=True)
            for rank, item in enumerate(ranked, start=1):
                if item["model"] == model_name:
                    ranks.append(rank)
                    scores.append(item["validation_metrics"]["roc_auc"]["mean"])
                    gaps.append(item["overfitting_gap_roc_auc"]["mean"])
                    total_time += item["timing_seconds"]["total"]["total"]
        output.append({
            "model": model_name,
            "datasets_completed": len(scores),
            "validation_roc_auc": {"mean": float(np.mean(scores)) if scores else None, "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0},
            "overfitting_gap_roc_auc": {"mean": float(np.mean(gaps)) if gaps else None, "std": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0},
            "mean_rank": float(np.mean(ranks)) if ranks else None,
            "total_time_seconds": total_time,
        })
    return sorted(output, key=lambda item: item["mean_rank"] if item["mean_rank"] is not None else float("inf"))


def signature(dataset_keys: list[str], args: argparse.Namespace, device: str, packages: dict[str, str]) -> str:
    payload = {"runner_version": "adaptive-v3", "datasets": dataset_keys, "sample_size": SAMPLE_SIZE, "sampling_seed": SAMPLING_SEED, "folds": args.folds, "seed": args.seed, "tabpfn_estimators": args.tabpfn_estimators, "device": device, "packages": packages}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_DIR / ".env")
    selected = selected_dataset_specs(args.datasets)
    if args.folds != 5:
        raise ValueError("This suite requires exactly 5 folds so every sampled task has equal-sized train and validation splits.")
    started = time.perf_counter()
    device = tabpfn_device()
    tabpfn_execution = (
        "Five same-shaped folds are fused with predict_proba_batched: one validation batch and one training-prediction batch."
        if device == "cuda"
        else "CPU scores folds independently because fusing or concatenating contexts is memory-heavy and slower on this backend. The CPU default uses a one-member ensemble; pass --tabpfn-estimators 8 for the library default."
    )
    packages = package_versions()
    run_id = signature([key for key, _ in selected], args, device, packages)
    checkpoint_path = RESULTS_DIR / "multidataset_checkpoints" / f"{run_id}.json"
    if args.restart and checkpoint_path.exists():
        checkpoint_path.unlink()
    previous = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    record_map: dict[tuple[str, int, str], dict[str, Any]] = {
        (record["dataset_key"], int(record["fold"]), record["model"]): record
        for record in previous.get("fold_results", [])
    }
    dataset_metadata: dict[str, dict[str, Any]] = previous.get("datasets", {})
    batch_profiles: dict[str, dict[str, float]] = previous.get("tabpfn_batch_profiles", {})

    def save_checkpoint(status: str) -> None:
        records = sorted(record_map.values(), key=lambda item: (item["dataset_key"], item["fold"], MODEL_ORDER.index(item["model"])))
        completed = sum(item["status"] == "success" for item in records)
        write_json(checkpoint_path, {
            "schema_version": 1, "status": status, "run_signature": run_id, "updated_at_utc": now_iso(),
            "datasets": dataset_metadata, "tabpfn_batch_profiles": batch_profiles,
            "evaluation_protocol": {"strategy": "Five-fold stratified outer cross-validation per dataset", "generalization_metric": "Mean held-out-fold ROC-AUC", "overfitting_gap": "Training ROC-AUC minus held-out-fold ROC-AUC", "tabpfn_execution": tabpfn_execution},
            "progress": {"planned_evaluations": len(selected) * len(MODEL_ORDER) * args.folds, "successful_evaluations": completed},
            "fold_results": records,
        })

    for dataset_key, spec in selected:
        features, target, metadata, load_seconds = load_dataset(dataset_key, spec)
        metadata["load_seconds_this_invocation"] = load_seconds
        dataset_metadata[dataset_key] = metadata
        numeric_columns, categorical_columns = get_feature_groups(features)
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(features, target))
        tabpfn_keys = [(dataset_key, fold_number, "TabPFN") for fold_number in range(1, args.folds + 1)]
        if not all(record_map.get(key, {}).get("status") == "success" for key in tabpfn_keys):
            print(f"{metadata['name']} | TabPFN {'fused batch' if device == 'cuda' else 'CPU per-fold'}", flush=True)
            if device == "cuda":
                tabpfn_records, profile = run_tabpfn_batched(
                    features=features, target=target, folds=folds, numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns, seed=args.seed, estimators=args.tabpfn_estimators, device=device,
                )
            else:
                tabpfn_records = []
                cpu_started = time.perf_counter()
                for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
                    preprocessor = tabpfn_preprocessor(numeric_columns, categorical_columns)
                    train_features = features.iloc[train_indices]
                    validation_features = features.iloc[validation_indices]
                    train_target = target.iloc[train_indices]
                    validation_target = target.iloc[validation_indices]
                    preprocessing_started = time.perf_counter()
                    train_array = np.asarray(preprocessor.fit_transform(train_features), dtype=np.float64)
                    validation_array = np.asarray(preprocessor.transform(validation_features), dtype=np.float64)
                    preprocessing_seconds = time.perf_counter() - preprocessing_started
                    classifier = TabPFNClassifier(
                        device=device, n_estimators=args.tabpfn_estimators,
                        categorical_features_indices=list(range(len(numeric_columns), len(numeric_columns) + len(categorical_columns))) or None,
                        random_state=args.seed, n_preprocessing_jobs=1, show_progress_bar=False,
                    )
                    fold_started = time.perf_counter()
                    try:
                        fit_started = time.perf_counter()
                        classifier.fit(train_array, train_target.to_numpy(dtype=int))
                        fit_seconds = time.perf_counter() - fit_started
                        train_prediction_started = time.perf_counter()
                        train_probabilities = classifier.predict_proba(train_array)[:, 1]
                        train_prediction_seconds = time.perf_counter() - train_prediction_started
                        validation_prediction_started = time.perf_counter()
                        validation_probabilities = classifier.predict_proba(validation_array)[:, 1]
                        validation_prediction_seconds = time.perf_counter() - validation_prediction_started
                        tabpfn_records.append(success_record(
                            "TabPFN", fold_number, train_target, validation_target,
                            classification_metrics(train_target, train_probabilities), classification_metrics(validation_target, validation_probabilities),
                            {"preprocess": preprocessing_seconds, "fit": fit_seconds, "predict_train": train_prediction_seconds, "predict_validation": validation_prediction_seconds, "total": time.perf_counter() - fold_started},
                        ))
                    except Exception as error:
                        tabpfn_records.append(failed_record("TabPFN", fold_number, error, fold_started))
                    # Persist every CPU TabPFN fold rather than waiting for all five.
                    record = tabpfn_records[-1]
                    record["dataset_key"] = dataset_key
                    record_map[(dataset_key, record["fold"], "TabPFN")] = record
                    save_checkpoint("in_progress")
                profile = {"mode": "cpu_per_fold", "total": time.perf_counter() - cpu_started}
            batch_profiles[dataset_key] = profile
            for record in tabpfn_records:
                record["dataset_key"] = dataset_key
                record_map[(dataset_key, record["fold"], "TabPFN")] = record
                save_checkpoint("in_progress")
        for model_name in MODEL_ORDER[1:]:
            for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
                key = (dataset_key, fold_number, model_name)
                if record_map.get(key, {}).get("status") == "success":
                    print(f"{metadata['name']} | Fold {fold_number} | {model_name} (resumed)", flush=True)
                    continue
                print(f"{metadata['name']} | Fold {fold_number} | {model_name}", flush=True)
                record = run_classical_fold(
                    model_name=model_name, numeric_columns=numeric_columns, categorical_columns=categorical_columns,
                    seed=args.seed, fold=fold_number, train_features=features.iloc[train_indices], validation_features=features.iloc[validation_indices],
                    train_target=target.iloc[train_indices], validation_target=target.iloc[validation_indices],
                )
                record["dataset_key"] = dataset_key
                record_map[key] = record
                save_checkpoint("in_progress")

    records = sorted(record_map.values(), key=lambda item: (item["dataset_key"], item["fold"], MODEL_ORDER.index(item["model"])))
    dataset_summaries = []
    for dataset_key, spec in selected:
        records_for_dataset = [record for record in records if record["dataset_key"] == dataset_key]
        dataset_summaries.append({"dataset_key": dataset_key, "dataset": dataset_metadata[dataset_key], "model_summaries": summarize(records_for_dataset), "tabpfn_batch_profile_seconds": batch_profiles.get(dataset_key, {})})
    save_checkpoint("complete")
    result_path = RESULTS_DIR / "multidataset_benchmark_results.json"
    report_path = RESULTS_DIR / "multidataset_benchmark_report.html"
    result: dict[str, Any] = {
        "schema_version": 1, "status": "complete", "run_signature": run_id, "generated_at_utc": now_iso(),
        "datasets": dataset_metadata, "dataset_summaries": dataset_summaries, "overall_model_summary": overall_summary(dataset_summaries),
        "evaluation_protocol": {"strategy": "Five-fold stratified outer cross-validation per dataset", "sampling": f"Each task is deterministically stratified-sampled to {SAMPLE_SIZE:,} rows when larger (seed {SAMPLING_SEED}).", "generalization_metric": "Mean held-out-fold ROC-AUC; overall score is an unweighted mean across datasets.", "overfitting_gap": "Training ROC-AUC minus held-out-fold ROC-AUC.", "tabpfn_execution": tabpfn_execution, "classical_execution": "Classical models are intentionally serial and single-threaded so timing is attributable."},
        "model_configurations": {"TabPFN": {"n_estimators": args.tabpfn_estimators, "device": device, "execution": "predict_proba_batched on CUDA; independent per-fold queries on CPU", "preprocessing": "Per-fold median imputation plus ordinal categorical encoding; categorical indices provided to TabPFN."}, "LightGBM": classical_pipeline("LightGBM", [], [], args.seed)[1], "XGBoost": classical_pipeline("XGBoost", [], [], args.seed)[1], "Random Forest": classical_pipeline("Random Forest", [], [], args.seed)[1], "Logistic Regression": classical_pipeline("Logistic Regression", [], [], args.seed)[1]},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "tabpfn_device": device, "packages": packages},
        "profiling_seconds": {"end_to_end_total": time.perf_counter() - started, "all_model_work": float(sum(record["timing_seconds"].get("total", 0.0) for record in records))},
        "fold_results": records,
        "outputs": {"json": str(result_path.relative_to(PROJECT_DIR)), "html_report": str(report_path.relative_to(PROJECT_DIR)), "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR))},
    }
    write_json(result_path, result)
    report_started = time.perf_counter()
    build_multidataset_report(result, report_path)
    result["profiling_seconds"]["html_report_build"] = time.perf_counter() - report_started
    write_json(result_path, result)
    build_multidataset_report(result, report_path)
    print(f"\nCompleted {len(selected)} datasets. JSON: {result_path}; report: {report_path}")


if __name__ == "__main__":
    main()
