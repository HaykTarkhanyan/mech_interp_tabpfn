"""Benchmark TabPFN and classical classifiers on the OpenML German Credit data.

The benchmark uses one shared, stratified outer cross-validation protocol.
Each held-out fold is never used for tuning, so its metrics estimate
generalization. The training-vs-validation ROC-AUC difference is the reported
overfitting gap.
"""

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
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from build_html_report import build_html_report


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
DATASET_ID = 31
DATASET_NAME = "German Credit (credit-g)"
TARGET_COLUMN = "credit_risk"
METRIC_KEYS = (
    "roc_auc",
    "average_precision",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "specificity",
    "log_loss",
    "brier_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5, help="Stratified outer CV folds.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for all splits/models.")
    parser.add_argument(
        "--tabpfn-estimators",
        type=int,
        default=8,
        help="TabPFN ensemble members; 8 is the library default.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore the cached CSV and fetch OpenML data again.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard only the checkpoint for this exact configuration and rerun every model-fold.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def load_german_credit(force_download: bool) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], float]:
    """Download OpenML data once, then use an on-disk CSV cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "credit_g.csv"
    metadata_path = DATA_DIR / "credit_g_metadata.json"
    started = time.perf_counter()

    if csv_path.exists() and metadata_path.exists() and not force_download:
        frame = pd.read_csv(csv_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["load_source"] = "local CSV cache"
    else:
        bunch = fetch_openml(data_id=DATASET_ID, as_frame=True, parser="auto")
        frame = bunch.data.copy()
        frame[TARGET_COLUMN] = bunch.target.astype(str)
        metadata = {
            "name": DATASET_NAME,
            "openml_data_id": DATASET_ID,
            "source_url": f"https://www.openml.org/d/{DATASET_ID}",
            "description": "German Credit (Statlog): binary credit-risk classification with mixed numeric and categorical attributes.",
            "load_source": "OpenML download",
        }
        frame.to_csv(csv_path, index=False)
        write_json(metadata_path, metadata)

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Cached data must contain {TARGET_COLUMN!r}.")

    target_text = frame.pop(TARGET_COLUMN).astype(str).str.strip().str.lower()
    labels = sorted(target_text.unique().tolist())
    if len(labels) != 2:
        raise ValueError(f"Expected a binary target, received labels: {labels}")
    positive_label = "good" if "good" in labels else labels[-1]
    target = (target_text == positive_label).astype(int)
    metadata.update(
        {
            "cache_path": str(csv_path.relative_to(PROJECT_DIR)),
            "rows": int(len(frame)),
            "raw_feature_count": int(frame.shape[1]),
            "positive_class": positive_label,
            "negative_class": next(label for label in labels if label != positive_label),
            "class_counts": {str(key): int(value) for key, value in Counter(target).items()},
            "missing_cells": int(frame.isna().sum().sum()),
        }
    )
    return frame, target, metadata, time.perf_counter() - started


def get_feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = [
        column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def one_hot_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str], *, scale_numeric: bool
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def tabpfn_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    """Keep categories compact and identify their transformed indices to TabPFN."""
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def tabpfn_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def model_specifications(
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    seed: int,
    tabpfn_estimators: int,
    device: str,
) -> dict[str, dict[str, Any]]:
    """Return factories to ensure every fold starts with a fresh estimator."""
    tabpfn_categories = list(
        range(len(numeric_columns), len(numeric_columns) + len(categorical_columns))
    )

    def tabpfn_factory() -> Pipeline:
        return Pipeline(
            [
                ("preprocess", tabpfn_preprocessor(numeric_columns, categorical_columns)),
                (
                    "model",
                    TabPFNClassifier(
                        device=device,
                        n_estimators=tabpfn_estimators,
                        categorical_features_indices=tabpfn_categories or None,
                        random_state=seed,
                        n_preprocessing_jobs=1,
                        show_progress_bar=False,
                    ),
                ),
            ]
        )

    def lightgbm_factory() -> Pipeline:
        return Pipeline(
            [
                ("preprocess", one_hot_preprocessor(numeric_columns, categorical_columns, scale_numeric=False)),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=seed,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    def xgboost_factory() -> Pipeline:
        return Pipeline(
            [
                ("preprocess", one_hot_preprocessor(numeric_columns, categorical_columns, scale_numeric=False)),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=seed,
                        n_jobs=1,
                        verbosity=0,
                    ),
                ),
            ]
        )

    def random_forest_factory() -> Pipeline:
        return Pipeline(
            [
                ("preprocess", one_hot_preprocessor(numeric_columns, categorical_columns, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def logistic_regression_factory() -> Pipeline:
        return Pipeline(
            [
                ("preprocess", one_hot_preprocessor(numeric_columns, categorical_columns, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(max_iter=2_000, solver="lbfgs", random_state=seed),
                ),
            ]
        )

    return {
        "TabPFN": {
            "build": tabpfn_factory,
            "configuration": {
                "preprocessing": "Median imputation for numeric columns; most-frequent imputation plus ordinal encoding for categorical columns; categorical indices passed to TabPFN.",
                "device": device,
                "n_estimators": tabpfn_estimators,
                "random_state": seed,
                "n_preprocessing_jobs": 1,
            },
        },
        "LightGBM": {
            "build": lightgbm_factory,
            "configuration": {
                "preprocessing": "Median imputation for numeric columns; most-frequent imputation plus one-hot encoding for categorical columns.",
                "objective": "binary",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": seed,
                "n_jobs": 1,
            },
        },
        "XGBoost": {
            "build": xgboost_factory,
            "configuration": {
                "preprocessing": "Median imputation for numeric columns; most-frequent imputation plus one-hot encoding for categorical columns.",
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": seed,
                "n_jobs": 1,
            },
        },
        "Random Forest": {
            "build": random_forest_factory,
            "configuration": {
                "preprocessing": "Median imputation for numeric columns; most-frequent imputation plus one-hot encoding for categorical columns.",
                "n_estimators": 500,
                "min_samples_leaf": 2,
                "random_state": seed,
                "n_jobs": 1,
            },
        },
        "Logistic Regression": {
            "build": logistic_regression_factory,
            "configuration": {
                "preprocessing": "Median imputation and standard scaling for numeric columns; most-frequent imputation plus one-hot encoding for categorical columns.",
                "max_iter": 2_000,
                "solver": "lbfgs",
                "random_state": seed,
            },
        },
    }


def positive_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    return probabilities[:, list(model.classes_).index(1)]


def classification_metrics(target: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    predictions = (probabilities >= 0.5).astype(int)
    true_negative = int(((target == 0) & (predictions == 0)).sum())
    false_positive = int(((target == 0) & (predictions == 1)).sum())
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probabilities)),
    }


def benchmark_one_model(
    *,
    model_name: str,
    specification: dict[str, Any],
    fold: int,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    train_target: pd.Series,
    validation_target: pd.Series,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        initialization_started = time.perf_counter()
        model = specification["build"]()
        initialization_seconds = time.perf_counter() - initialization_started

        fit_started = time.perf_counter()
        model.fit(train_features, train_target)
        fit_seconds = time.perf_counter() - fit_started

        train_prediction_started = time.perf_counter()
        train_probabilities = positive_probabilities(model, train_features)
        train_prediction_seconds = time.perf_counter() - train_prediction_started

        validation_prediction_started = time.perf_counter()
        validation_probabilities = positive_probabilities(model, validation_features)
        validation_prediction_seconds = time.perf_counter() - validation_prediction_started

        train_metrics = classification_metrics(train_target, train_probabilities)
        validation_metrics = classification_metrics(validation_target, validation_probabilities)
        return {
            "status": "success",
            "model": model_name,
            "fold": fold,
            "train_rows": int(len(train_features)),
            "validation_rows": int(len(validation_features)),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "overfitting_gap_roc_auc": train_metrics["roc_auc"] - validation_metrics["roc_auc"],
            "timing_seconds": {
                "initialization": initialization_seconds,
                "fit": fit_seconds,
                "predict_train": train_prediction_seconds,
                "predict_validation": validation_prediction_seconds,
                "total": time.perf_counter() - started,
            },
        }
    except Exception as error:
        return {
            "status": "failed",
            "model": model_name,
            "fold": fold,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "timing_seconds": {"total": time.perf_counter() - started},
        }


def summarize_records(records: list[dict[str, Any]], model_order: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model_name in model_order:
        successful = [
            record for record in records if record["model"] == model_name and record["status"] == "success"
        ]
        failed = [
            record for record in records if record["model"] == model_name and record["status"] != "success"
        ]
        summary: dict[str, Any] = {
            "model": model_name,
            "successful_folds": len(successful),
            "failed_folds": len(failed),
            "validation_metrics": {},
            "train_metrics": {},
            "overfitting_gap_roc_auc": {},
            "timing_seconds": {},
        }
        if successful:
            for split_key in ("validation_metrics", "train_metrics"):
                for metric_name in METRIC_KEYS:
                    values = np.asarray([record[split_key][metric_name] for record in successful])
                    summary[split_key][metric_name] = {
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
            gaps = np.asarray([record["overfitting_gap_roc_auc"] for record in successful])
            summary["overfitting_gap_roc_auc"] = {
                "mean": float(gaps.mean()),
                "std": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
                "min": float(gaps.min()),
                "max": float(gaps.max()),
            }
            for timing_name in ("initialization", "fit", "predict_train", "predict_validation", "total"):
                values = np.asarray(
                    [record["timing_seconds"].get(timing_name, 0.0) for record in successful]
                )
                summary["timing_seconds"][timing_name] = {
                    "mean": float(values.mean()),
                    "total": float(values.sum()),
                }
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda item: item["validation_metrics"].get("roc_auc", {}).get("mean", float("-inf")),
        reverse=True,
    )


def package_versions() -> dict[str, str]:
    packages = ("tabpfn", "lightgbm", "xgboost", "scikit-learn", "pandas", "python-dotenv")
    return {package: version(package) for package in packages}


def run_signature(
    *,
    folds: int,
    seed: int,
    tabpfn_estimators: int,
    device: str,
    model_configurations: dict[str, Any],
    packages: dict[str, str],
) -> str:
    """Create a stable ID so only identical runs resume from one another."""
    signature_input = {
        "dataset_id": DATASET_ID,
        "folds": folds,
        "seed": seed,
        "tabpfn_estimators": tabpfn_estimators,
        "device": device,
        "model_configurations": model_configurations,
        "packages": packages,
    }
    encoded = json.dumps(signature_input, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_checkpoint(path: Path, signature: str) -> dict[tuple[int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("run_signature") != signature:
        return {}
    return {
        (int(record["fold"]), str(record["model"])): record
        for record in checkpoint.get("fold_results", [])
    }


def ordered_records(
    record_map: dict[tuple[int, str], dict[str, Any]], model_order: list[str]
) -> list[dict[str, Any]]:
    model_positions = {name: index for index, name in enumerate(model_order)}
    return sorted(record_map.values(), key=lambda item: (int(item["fold"]), model_positions[item["model"]]))


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_DIR / ".env")
    started = time.perf_counter()
    features, target, dataset, dataset_load_seconds = load_german_credit(args.force_download)
    numeric_columns, categorical_columns = get_feature_groups(features)
    dataset.update(
        {
            "numeric_feature_count": len(numeric_columns),
            "categorical_feature_count": len(categorical_columns),
            "feature_names": features.columns.tolist(),
        }
    )

    device = tabpfn_device()
    split_started = time.perf_counter()
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(features, target))
    split_setup_seconds = time.perf_counter() - split_started
    specifications = model_specifications(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        seed=args.seed,
        tabpfn_estimators=args.tabpfn_estimators,
        device=device,
    )
    model_configurations = {
        name: specification["configuration"] for name, specification in specifications.items()
    }
    packages = package_versions()
    signature = run_signature(
        folds=args.folds,
        seed=args.seed,
        tabpfn_estimators=args.tabpfn_estimators,
        device=device,
        model_configurations=model_configurations,
        packages=packages,
    )
    checkpoint_path = RESULTS_DIR / "checkpoints" / f"{signature}.json"
    if args.restart and checkpoint_path.exists():
        checkpoint_path.unlink()
    record_map = load_checkpoint(checkpoint_path, signature)
    resumed_successes = sum(record["status"] == "success" for record in record_map.values())
    model_order = list(specifications)
    planned_evaluations = args.folds * len(model_order)

    def save_checkpoint(status: str = "in_progress") -> None:
        ordered = ordered_records(record_map, model_order)
        completed_successes = sum(record["status"] == "success" for record in ordered)
        checkpoint: dict[str, Any] = {
            "schema_version": 1,
            "status": status,
            "run_signature": signature,
            "updated_at_utc": now_iso(),
            "dataset": dataset,
            "evaluation_protocol": {
                "strategy": f"{args.folds}-fold stratified outer cross-validation",
                "random_seed": args.seed,
                "validation_use": "Each held-out fold is used only for evaluation; no hyperparameter search or threshold tuning is performed.",
                "generalization_metric": "Mean held-out-fold ROC-AUC with standard deviation across folds.",
                "overfitting_gap": "Per-fold training ROC-AUC minus held-out-fold ROC-AUC; a larger positive gap indicates more apparent overfitting.",
                "execution_policy": "Models run serially and classical models use one CPU thread, making timing attributable and avoiding CPU/GPU contention.",
            },
            "model_configurations": model_configurations,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "tabpfn_device": device,
                "packages": packages,
            },
            "profiling_seconds": {
                "dataset_load": dataset_load_seconds,
                "cross_validation_setup": split_setup_seconds,
                "completed_model_work": float(
                    sum(record["timing_seconds"].get("total", 0.0) for record in ordered)
                ),
            },
            "progress": {
                "planned_evaluations": planned_evaluations,
                "successful_evaluations": completed_successes,
                "remaining_evaluations": planned_evaluations - completed_successes,
            },
            "fold_results": ordered,
        }
        write_json(checkpoint_path, checkpoint)

    benchmark_started = time.perf_counter()
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        train_features = features.iloc[train_indices]
        validation_features = features.iloc[validation_indices]
        train_target = target.iloc[train_indices]
        validation_target = target.iloc[validation_indices]
        for model_name, specification in specifications.items():
            key = (fold_number, model_name)
            if record_map.get(key, {}).get("status") == "success":
                print(f"Fold {fold_number}/{args.folds} | {model_name} (resumed)", flush=True)
                continue
            print(f"Fold {fold_number}/{args.folds} | {model_name}", flush=True)
            record_map[key] = benchmark_one_model(
                model_name=model_name,
                specification=specification,
                fold=fold_number,
                train_features=train_features,
                validation_features=validation_features,
                train_target=train_target,
                validation_target=validation_target,
            )
            save_checkpoint()
    invocation_benchmark_seconds = time.perf_counter() - benchmark_started

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "benchmark_results.json"
    report_path = RESULTS_DIR / "benchmark_report.html"
    records = ordered_records(record_map, model_order)
    save_checkpoint(status="complete")
    all_model_work_seconds = float(sum(record["timing_seconds"].get("total", 0.0) for record in records))
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "run_signature": signature,
        "generated_at_utc": now_iso(),
        "dataset": dataset,
        "evaluation_protocol": {
            "strategy": f"{args.folds}-fold stratified outer cross-validation",
            "random_seed": args.seed,
            "validation_use": "Each held-out fold is used only for evaluation; no hyperparameter search or threshold tuning is performed.",
            "generalization_metric": "Mean held-out-fold ROC-AUC with standard deviation across folds.",
            "overfitting_gap": "Per-fold training ROC-AUC minus held-out-fold ROC-AUC; a larger positive gap indicates more apparent overfitting.",
            "execution_policy": "Models run serially and classical models use one CPU thread, making timing attributable and avoiding CPU/GPU contention.",
        },
        "model_configurations": model_configurations,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tabpfn_device": device,
            "packages": packages,
        },
        "profiling_seconds": {
            "dataset_load": dataset_load_seconds,
            "cross_validation_setup": split_setup_seconds,
            "model_benchmark_total": all_model_work_seconds,
            "this_invocation_model_work": invocation_benchmark_seconds,
            "resumed_successful_evaluations": resumed_successes,
            "end_to_end_before_report": time.perf_counter() - started,
        },
        "fold_results": records,
        "model_summaries": summarize_records(records, list(specifications)),
        "outputs": {
            "json": str(result_path.relative_to(PROJECT_DIR)),
            "html_report": str(report_path.relative_to(PROJECT_DIR)),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        },
    }
    write_json(result_path, result)

    report_started = time.perf_counter()
    build_html_report(result, report_path)
    result["profiling_seconds"]["html_report_build"] = time.perf_counter() - report_started
    result["profiling_seconds"]["end_to_end_total"] = time.perf_counter() - started
    write_json(result_path, result)
    build_html_report(result, report_path)

    print("\nBenchmark complete.")
    print(f"JSON:   {result_path}")
    print(f"Report: {report_path}")
    for summary in result["model_summaries"]:
        score = summary["validation_metrics"].get("roc_auc", {}).get("mean")
        if score is not None:
            print(f"{summary['model']}: validation ROC-AUC {score:.3f}")


if __name__ == "__main__":
    main()
