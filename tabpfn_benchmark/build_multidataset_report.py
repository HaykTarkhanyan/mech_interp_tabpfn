"""Build a detailed, self-contained HTML report from benchmark result JSON."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Avg. precision",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall / TPR",
    "specificity": "Specificity / TNR",
    "false_positive_rate": "False-positive rate",
    "youden_j": "Youden's J",
    "log_loss": "Log loss",
    "brier_score": "Brier score",
}
BASE_MODELS = ("TabPFN", "LightGBM", "XGBoost", "Random Forest", "Logistic Regression")


def metric(value: float | None, digits: int = 3) -> str:
    return "&mdash;" if value is None else f"{value:.{digits}f}"


def plus_minus(stats: dict[str, float] | None, digits: int = 3) -> str:
    if not stats or stats.get("mean") is None:
        return "&mdash;"
    return f"{metric(stats['mean'], digits)} <span class=\"pm\">&plusmn; {metric(stats.get('std'), digits)}</span>"


def seconds(value: float | None) -> str:
    if value is None:
        return "&mdash;"
    if value < 1:
        return f"{value * 1_000:.0f} ms"
    if value < 60:
        return f"{value:.2f} s"
    return f"{value / 60:.2f} min"


def cell(display: str, value: float | str | None = None, classes: str = "") -> tuple[str, float | str | None, str]:
    return display, value, classes


def metric_tone(name: str, value: float | None) -> str:
    if value is None:
        return ""
    if name in {"roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity", "youden_j"}:
        if value >= 0.90:
            return "tone-excellent"
        if value >= 0.80:
            return "tone-good"
        if value >= 0.70:
            return "tone-mid"
        return "tone-low"
    if name == "false_positive_rate":
        return "tone-good" if value <= 0.15 else "tone-mid" if value <= 0.30 else "tone-low"
    if name == "log_loss":
        return "tone-good" if value <= 0.35 else "tone-mid" if value <= 0.60 else "tone-low"
    if name == "brier_score":
        return "tone-good" if value <= 0.12 else "tone-mid" if value <= 0.20 else "tone-low"
    if name.startswith("gap"):
        return "tone-good" if value <= 0.06 else "tone-mid" if value <= 0.13 else "tone-low"
    return ""


def delta_tone(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) < 0.002:
        return "tone-neutral"
    return "tone-positive" if value > 0 else "tone-negative"


def column_colour_direction(label: str, override: str = "") -> str | None:
    """Return the comparison direction for report columns worth heat-mapping."""
    if override:
        return override
    normalized = label.casefold()
    if normalized in {"fpr", "false-positive rate", "log loss", "brier", "brier score", "auc gap"}:
        return "low"
    if "delta" in normalized or normalized == "outer - inner":
        return "delta"
    if normalized in {
        "roc-auc", "avg. precision", "accuracy", "balanced acc.", "balanced accuracy",
        "f1", "precision", "recall", "recall / tpr", "specificity", "specificity / tnr",
        "youden's j", "train auc", "train roc-auc", "held-out auc", "held-out roc-auc",
        "inner auc", "outer auc",
    }:
        return "high"
    return None


def apply_column_colours(
    headers: list[tuple[str, bool, str]],
    rows: list[list[tuple[str, float | str | None, str]]],
) -> list[list[tuple[str, float | str | None, str]]]:
    """Apply a relative heatmap independently to every meaningful metric column."""
    coloured = [[cell_item for cell_item in row] for row in rows]
    for index, (label, is_numeric, override) in enumerate(headers):
        direction = column_colour_direction(label, override)
        if not is_numeric or direction is None:
            continue

        values: list[float] = []
        for row in coloured:
            display, raw_value, _ = row[index]
            if "Failed" in display or raw_value is None:
                continue
            try:
                values.append(float(raw_value))
            except (TypeError, ValueError):
                continue
        if not values:
            continue

        minimum, maximum = min(values), max(values)
        for row in coloured:
            display, raw_value, classes = row[index]
            if "Failed" in display or raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue

            # Remove the former absolute-threshold class: table colours are now
            # meaningful only relative to peer entries in this same column.
            preserved = " ".join(token for token in classes.split() if not token.startswith("tone-"))
            if direction == "delta":
                colour = "col-positive" if value > 0.002 else "col-negative" if value < -0.002 else "col-neutral"
            elif maximum == minimum:
                colour = "col-neutral"
            else:
                score = (value - minimum) / (maximum - minimum)
                if direction == "low":
                    score = 1 - score
                colour = (
                    "col-best" if score >= 0.75 else "col-good" if score >= 0.50
                    else "col-mid" if score >= 0.25 else "col-low"
                )
            row[index] = cell(display, raw_value, f"{preserved} {colour}".strip())
    return coloured


def sortable_table(
    table_id: str,
    headers: list[tuple[str, bool, str]],
    rows: list[list[tuple[str, float | str | None, str]]],
    caption: str = "",
) -> str:
    rows = apply_column_colours(headers, rows)
    head = "".join(
        f'<th scope="col" data-col="{index}" data-numeric="{str(is_numeric).lower()}"><button type="button" aria-label="Sort by {html.escape(label)}">{html.escape(label)}<span class="sort-mark">&#8597;</span></button></th>'
        for index, (label, is_numeric, _) in enumerate(headers)
    )
    body = ""
    for row in rows:
        columns = []
        for display, value, classes in row:
            data_value = "" if value is None else html.escape(str(value), quote=True)
            columns.append(f'<td data-sort="{data_value}" class="{classes}">{display}</td>')
        body += "<tr>" + "".join(columns) + "</tr>"
    caption_html = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="table-wrap"><table id="{table_id}" class="sort-table">{caption_html}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def summary_value(summary: dict[str, Any], split: str, metric_name: str) -> dict[str, float] | None:
    return summary.get(split, {}).get(metric_name)


def derived_summary_metrics(summary: dict[str, Any], split: str = "validation_metrics") -> dict[str, dict[str, float]]:
    source = summary.get(split, {})
    derived_names = ("balanced_accuracy", "false_positive_rate", "youden_j")
    if all(name in source for name in derived_names):
        return {name: source[name] for name in derived_names}
    recall = source.get("recall", {})
    specificity = source.get("specificity", {})
    if not recall or not specificity:
        return {}
    derived: dict[str, dict[str, float]] = {}
    for name, fn in {
        "balanced_accuracy": lambda r, s: (r + s) / 2,
        "false_positive_rate": lambda r, s: 1 - s,
        "youden_j": lambda r, s: r + s - 1,
    }.items():
        derived[name] = {
            "mean": fn(recall["mean"], specificity["mean"]),
            "std": (recall.get("std", 0.0) ** 2 + specificity.get("std", 0.0) ** 2) ** 0.5 / 2
            if name == "balanced_accuracy"
            else 0.0,
        }
    return derived


def enrich_summaries_with_fold_metrics(
    summaries: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach exact derived means/stds computed from individual outer folds."""
    enriched: list[dict[str, Any]] = []
    for summary in summaries:
        copied = json.loads(json.dumps(summary))
        matching = [
            record for record in records
            if record["model"] == summary["model"] and record["status"] == "success"
        ]
        if matching:
            for split in ("validation_metrics", "train_metrics"):
                for name in ("balanced_accuracy", "false_positive_rate", "youden_j"):
                    values = [fold_derived_metrics(record, split)[name] for record in matching]
                    copied[split][name] = {
                        "mean": mean(values),
                        "std": stdev(values) if len(values) > 1 else 0.0,
                    }
        enriched.append(copied)
    return enriched


def fold_derived_metrics(record: dict[str, Any], split: str = "validation_metrics") -> dict[str, float]:
    metrics = record[split]
    recall = metrics["recall"]
    specificity = metrics["specificity"]
    return {
        "balanced_accuracy": (recall + specificity) / 2,
        "false_positive_rate": 1 - specificity,
        "youden_j": recall + specificity - 1,
    }


def base_model_name(model: str) -> str | None:
    return model.removesuffix(" Tuned") if model.endswith(" Tuned") else None


def model_sort_key(model: str) -> tuple[int, str]:
    try:
        return BASE_MODELS.index(model), model
    except ValueError:
        base = base_model_name(model)
        return (BASE_MODELS.index(base) + len(BASE_MODELS) if base in BASE_MODELS else 999, model)


def aggregate_overall(dataset_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create unweighted across-dataset metrics for every displayed model."""
    summaries_by_model: dict[str, list[dict[str, Any]]] = {}
    for dataset_item in dataset_summaries:
        for summary in dataset_item["model_summaries"]:
            if summary["validation_metrics"].get("roc_auc"):
                summaries_by_model.setdefault(summary["model"], []).append(summary)

    output: list[dict[str, Any]] = []
    for model, summaries in summaries_by_model.items():
        values: dict[str, list[float]] = {name: [] for name in METRIC_LABELS}
        train_auc: list[float] = []
        gaps: list[float] = []
        for summary in summaries:
            validation = summary["validation_metrics"]
            derived = derived_summary_metrics(summary)
            for name in ("roc_auc", "average_precision", "accuracy", "f1", "precision", "recall", "specificity", "log_loss", "brier_score"):
                values[name].append(validation[name]["mean"])
            for name, stats in derived.items():
                values[name].append(stats["mean"])
            train_auc.append(summary["train_metrics"]["roc_auc"]["mean"])
            gaps.append(summary["overfitting_gap_roc_auc"]["mean"])
        output.append({
            "model": model,
            "datasets_completed": len(summaries),
            "validation_metrics": {
                name: {"mean": mean(items), "std": stdev(items) if len(items) > 1 else 0.0}
                for name, items in values.items() if items
            },
            "train_auc": {"mean": mean(train_auc), "std": stdev(train_auc) if len(train_auc) > 1 else 0.0},
            "overfitting_gap_roc_auc": {"mean": mean(gaps), "std": stdev(gaps) if len(gaps) > 1 else 0.0},
            "timing_seconds": {
                key: sum(summary["timing_seconds"].get(key, {}).get("total", 0.0) for summary in summaries)
                for key in ("tune", "fit", "predict_train", "predict_validation", "total")
            },
        })

    for item in output:
        base = base_model_name(item["model"])
        if base:
            baseline = next((candidate for candidate in output if candidate["model"] == base), None)
            item["auc_delta_vs_baseline"] = (
                item["validation_metrics"]["roc_auc"]["mean"] - baseline["validation_metrics"]["roc_auc"]["mean"]
                if baseline else None
            )
        else:
            item["auc_delta_vs_baseline"] = None
    return sorted(output, key=lambda item: item["validation_metrics"]["roc_auc"]["mean"], reverse=True)


def metric_cell(stats: dict[str, float] | None, name: str, *, with_std: bool = True) -> tuple[str, float | None, str]:
    value = stats.get("mean") if stats else None
    display = plus_minus(stats) if with_std else metric(value)
    return cell(display, value, metric_tone(name, value))


def overall_rows(overall: list[dict[str, Any]]) -> list[list[tuple[str, float | str | None, str]]]:
    rows = []
    for item in overall:
        metrics = item["validation_metrics"]
        gap = item["overfitting_gap_roc_auc"]
        delta = item.get("auc_delta_vs_baseline")
        delta_display = "Baseline" if delta is None else f"{delta:+.3f}"
        rows.append([
            cell(html.escape(item["model"]), item["model"], "model-name"),
            cell(str(item["datasets_completed"]), item["datasets_completed"]),
            metric_cell(metrics["roc_auc"], "roc_auc"),
            metric_cell(metrics["average_precision"], "average_precision"),
            metric_cell(metrics["balanced_accuracy"], "balanced_accuracy"),
            metric_cell(metrics["f1"], "f1"),
            metric_cell(metrics["precision"], "precision"),
            metric_cell(metrics["recall"], "recall"),
            metric_cell(metrics["specificity"], "specificity"),
            metric_cell(metrics["false_positive_rate"], "false_positive_rate"),
            metric_cell(metrics["log_loss"], "log_loss"),
            metric_cell(metrics["brier_score"], "brier_score"),
            metric_cell(item["train_auc"], "roc_auc"),
            metric_cell(gap, "gap_auc"),
            cell(delta_display, delta, delta_tone(delta)),
            cell(seconds(item["timing_seconds"]["tune"]), item["timing_seconds"]["tune"], "time-cell"),
            cell(seconds(item["timing_seconds"]["total"]), item["timing_seconds"]["total"], "time-cell"),
        ])
    return rows


def quick_overview_rows(overall: list[dict[str, Any]]) -> list[list[tuple[str, float | str | None, str]]]:
    """Compact macro-average comparison for the first screen of the report."""
    rows = []
    for item in overall:
        metrics = item["validation_metrics"]
        delta = item.get("auc_delta_vs_baseline")
        rows.append([
            cell(html.escape(item["model"]), item["model"], "model-name"),
            metric_cell(metrics["roc_auc"], "roc_auc"),
            metric_cell(metrics["average_precision"], "average_precision"),
            metric_cell(metrics["balanced_accuracy"], "balanced_accuracy"),
            metric_cell(metrics["f1"], "f1"),
            metric_cell(item["overfitting_gap_roc_auc"], "gap_auc"),
            metric_cell(metrics["log_loss"], "log_loss"),
            metric_cell(metrics["brier_score"], "brier_score"),
            cell("Baseline" if delta is None else f"{delta:+.3f}", delta, delta_tone(delta)),
            cell(seconds(item["timing_seconds"]["total"]), item["timing_seconds"]["total"], "time-cell"),
        ])
    return rows


def dataset_score_rows(summaries: list[dict[str, Any]]) -> list[list[tuple[str, float | str | None, str]]]:
    rows = []
    summary_map = {summary["model"]: summary for summary in summaries}
    ordered = sorted(summaries, key=lambda item: item["validation_metrics"].get("roc_auc", {}).get("mean", -1), reverse=True)
    for rank, summary in enumerate(ordered, start=1):
        validation = summary["validation_metrics"]
        derived = derived_summary_metrics(summary)
        base = base_model_name(summary["model"])
        delta = None
        if base and base in summary_map:
            delta = validation["roc_auc"]["mean"] - summary_map[base]["validation_metrics"]["roc_auc"]["mean"]
        rows.append([
            cell(str(rank), rank),
            cell(html.escape(summary["model"]), summary["model"], "model-name"),
            metric_cell(validation["roc_auc"], "roc_auc"),
            metric_cell(validation["average_precision"], "average_precision"),
            metric_cell(validation["accuracy"], "accuracy"),
            metric_cell(derived["balanced_accuracy"], "balanced_accuracy"),
            metric_cell(validation["f1"], "f1"),
            metric_cell(validation["precision"], "precision"),
            metric_cell(validation["recall"], "recall"),
            metric_cell(validation["specificity"], "specificity"),
            metric_cell(derived["false_positive_rate"], "false_positive_rate"),
            metric_cell(derived["youden_j"], "youden_j"),
            cell("Baseline" if delta is None else f"{delta:+.3f}", delta, delta_tone(delta)),
        ])
    return rows


def dataset_reliability_rows(summaries: list[dict[str, Any]]) -> list[list[tuple[str, float | str | None, str]]]:
    rows = []
    ordered = sorted(summaries, key=lambda item: item["validation_metrics"].get("roc_auc", {}).get("mean", -1), reverse=True)
    for summary in ordered:
        validation = summary["validation_metrics"]
        train = summary["train_metrics"]
        timing = summary["timing_seconds"]
        rows.append([
            cell(html.escape(summary["model"]), summary["model"], "model-name"),
            metric_cell(train["roc_auc"], "roc_auc"),
            metric_cell(validation["roc_auc"], "roc_auc"),
            metric_cell(summary["overfitting_gap_roc_auc"], "gap_auc"),
            metric_cell(validation["log_loss"], "log_loss"),
            metric_cell(validation["brier_score"], "brier_score"),
            cell(seconds(timing.get("tune", {}).get("total", 0.0)), timing.get("tune", {}).get("total", 0.0), "time-cell"),
            cell(seconds(timing.get("fit", {}).get("total", 0.0)), timing.get("fit", {}).get("total", 0.0), "time-cell"),
            cell(seconds(timing.get("predict_train", {}).get("total", 0.0)), timing.get("predict_train", {}).get("total", 0.0), "time-cell"),
            cell(seconds(timing.get("predict_validation", {}).get("total", 0.0)), timing.get("predict_validation", {}).get("total", 0.0), "time-cell"),
            cell(seconds(timing.get("total", {}).get("total", 0.0)), timing.get("total", {}).get("total", 0.0), "time-cell"),
        ])
    return rows


def fold_rows(records: list[dict[str, Any]]) -> list[list[tuple[str, float | str | None, str]]]:
    rows = []
    for record in sorted(records, key=lambda item: (model_sort_key(item["model"]), item["fold"])):
        if record["status"] != "success":
            rows.append([
                cell(html.escape(record["model"]), record["model"], "model-name"), cell(str(record["fold"]), record["fold"]),
                cell("Failed", -1, "tone-low"), cell(html.escape(record.get("error", "Unknown error"))),
            ])
            continue
        validation = record["validation_metrics"]
        train = record["train_metrics"]
        derived = fold_derived_metrics(record)
        rows.append([
            cell(html.escape(record["model"]), record["model"], "model-name"),
            cell(str(record["fold"]), record["fold"]),
            cell(metric(train["roc_auc"]), train["roc_auc"], metric_tone("roc_auc", train["roc_auc"])),
            cell(metric(validation["roc_auc"]), validation["roc_auc"], metric_tone("roc_auc", validation["roc_auc"])),
            cell(metric(record["overfitting_gap_roc_auc"]), record["overfitting_gap_roc_auc"], metric_tone("gap_auc", record["overfitting_gap_roc_auc"])),
            cell(metric(validation["average_precision"]), validation["average_precision"], metric_tone("average_precision", validation["average_precision"])),
            cell(metric(validation["accuracy"]), validation["accuracy"], metric_tone("accuracy", validation["accuracy"])),
            cell(metric(derived["balanced_accuracy"]), derived["balanced_accuracy"], metric_tone("balanced_accuracy", derived["balanced_accuracy"])),
            cell(metric(validation["f1"]), validation["f1"], metric_tone("f1", validation["f1"])),
            cell(metric(validation["recall"]), validation["recall"], metric_tone("recall", validation["recall"])),
            cell(metric(validation["specificity"]), validation["specificity"], metric_tone("specificity", validation["specificity"])),
            cell(metric(derived["false_positive_rate"]), derived["false_positive_rate"], metric_tone("false_positive_rate", derived["false_positive_rate"])),
            cell(metric(validation["log_loss"]), validation["log_loss"], metric_tone("log_loss", validation["log_loss"])),
            cell(metric(validation["brier_score"]), validation["brier_score"], metric_tone("brier_score", validation["brier_score"])),
            cell(seconds(record["timing_seconds"].get("total", 0.0)), record["timing_seconds"].get("total", 0.0), "time-cell"),
        ])
    return rows


def tuning_rows(records: list[dict[str, Any]]) -> list[list[tuple[str, float | str | None, str]]]:
    rows = []
    for record in sorted((item for item in records if item.get("tuning")), key=lambda item: (model_sort_key(item["model"]), item["fold"])):
        tuning = record["tuning"]
        parameters = html.escape(json.dumps(tuning.get("best_parameters", {}), separators=(", ", ": ")))
        inner_score = tuning.get("best_inner_validation_roc_auc", tuning.get("best_inner_cv_roc_auc"))
        outer_score = record["validation_metrics"]["roc_auc"]
        rows.append([
            cell(html.escape(record["model"]), record["model"], "model-name"),
            cell(str(record["fold"]), record["fold"]),
            cell(html.escape(str(tuning.get("selected_profile", "legacy search"))), str(tuning.get("selected_profile", "legacy search")), "profile-cell"),
            cell(metric(inner_score), inner_score, metric_tone("roc_auc", inner_score)),
            cell(metric(outer_score), outer_score, metric_tone("roc_auc", outer_score)),
            cell(f"{outer_score - inner_score:+.3f}" if inner_score is not None else "&mdash;", outer_score - inner_score if inner_score is not None else None, delta_tone(outer_score - inner_score if inner_score is not None else None)),
            cell(str(tuning.get("trials_completed", "&mdash;")), tuning.get("trials_completed")),
            cell(seconds(record["timing_seconds"].get("tune", 0.0)), record["timing_seconds"].get("tune", 0.0), "time-cell"),
            cell(f"<code>{parameters}</code>"),
        ])
    return rows


def profile_counts(records: list[dict[str, Any]]) -> list[list[tuple[str, float | str | None, str]]]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        if record.get("tuning"):
            counts[(record["model"], str(record["tuning"].get("selected_profile", "legacy search")))] += 1
    return [
        [cell(html.escape(model), model, "model-name"), cell(html.escape(profile), profile, "profile-cell"), cell(str(count), count)]
        for (model, profile), count in sorted(counts.items(), key=lambda item: (model_sort_key(item[0][0]), item[0][1]))
    ]


def dataset_card(item: dict[str, Any], all_records: list[dict[str, Any]], index: int) -> str:
    dataset = item["dataset"]
    key = item["dataset_key"]
    records = [record for record in all_records if record["dataset_key"] == key]
    summaries = enrich_summaries_with_fold_metrics(item["model_summaries"], records)
    tabpfn_profile = item.get("tabpfn_batch_profile_seconds", {})
    profile_note = ""
    if tabpfn_profile:
        mode = "CPU per-fold inference" if tabpfn_profile.get("mode") == "cpu_per_fold" else "fused batch inference"
        profile_note = f'<span class="meta-chip">TabPFN: {html.escape(mode)} ({seconds(tabpfn_profile.get("total"))})</span>'
    class_distribution = ", ".join(
        f"{html.escape(str(label))}: {count}" for label, count in dataset["class_counts"].items()
    )
    detail_id = f"dataset-{index}"
    score_table = sortable_table(
        f"scores-{index}",
        [("Rank", True, ""), ("Model", False, ""), ("ROC-AUC", True, ""), ("Avg. precision", True, ""), ("Accuracy", True, ""), ("Balanced acc.", True, ""), ("F1", True, ""), ("Precision", True, ""), ("Recall / TPR", True, ""), ("Specificity / TNR", True, ""), ("FPR", True, ""), ("Youden's J", True, ""), ("Tuned AUC delta", True, "")],
        dataset_score_rows(summaries),
        "Held-out outer-fold metrics. Values are mean +/- sample standard deviation across five folds.",
    )
    reliability_table = sortable_table(
        f"reliability-{index}",
        [("Model", False, ""), ("Train ROC-AUC", True, ""), ("Held-out ROC-AUC", True, ""), ("AUC gap", True, ""), ("Log loss", True, ""), ("Brier", True, ""), ("Tune", True, ""), ("Fit", True, ""), ("Train predict", True, ""), ("Held-out predict", True, ""), ("Total", True, "")],
        dataset_reliability_rows(summaries),
        "Calibration losses are lower-is-better. AUC gap is train ROC-AUC minus held-out ROC-AUC; smaller is generally preferable.",
    )
    fold_table = sortable_table(
        f"folds-{index}",
        [("Model", False, ""), ("Fold", True, ""), ("Train AUC", True, ""), ("Held-out AUC", True, ""), ("AUC gap", True, ""), ("Avg. precision", True, ""), ("Accuracy", True, ""), ("Balanced acc.", True, ""), ("F1", True, ""), ("Recall", True, ""), ("Specificity", True, ""), ("FPR", True, ""), ("Log loss", True, ""), ("Brier", True, ""), ("Total time", True, "")],
        fold_rows(records),
        "Fold-level held-out scores. Click any heading to sort.",
    )
    tuning_table = ""
    rows = tuning_rows(records)
    if rows:
        tuning_table = f'<details><summary>Tuning audit: selected profile, inner validation, and exact parameters</summary>{sortable_table(f"tuning-{index}", [("Model", False, ""), ("Outer fold", True, ""), ("Profile", False, ""), ("Inner AUC", True, ""), ("Outer AUC", True, ""), ("Outer - inner", True, ""), ("Trials", True, ""), ("Tune time", True, ""), ("Selected parameters", False, "")], rows, "Each tuned model selected a profile using only the outer training subset.")}{sortable_table(f"profiles-{index}", [("Model", False, ""), ("Profile", False, ""), ("Selected folds", True, "")], profile_counts(records), "Profile selection frequency across outer folds.")}</details>'
    return f"""<section id="{detail_id}" class="dataset-card">
  <div class="dataset-heading">
    <div><p class="eyebrow">Dataset {index + 1} / 5</p><h2>{html.escape(dataset['name'])}</h2><p class="lede">{html.escape(dataset['description'])}</p></div>
    <a class="source-link" href="{html.escape(dataset['source_url'])}">OpenML {dataset['openml_data_id']} &nearr;</a>
  </div>
  <div class="dataset-meta">
    <span class="meta-chip">{dataset['rows']:,} rows</span><span class="meta-chip">{dataset['raw_feature_count']} features: {dataset['numeric_feature_count']} numeric / {dataset['categorical_feature_count']} categorical</span><span class="meta-chip">Positive class: <code>{html.escape(dataset['positive_class'])}</code></span><span class="meta-chip">Class counts: {class_distribution}</span>{profile_note}
  </div>
  <p class="note">{html.escape(dataset['sampling'])} Missing cells: {dataset['missing_cells']:,}.</p>
  <div class="card table-card"><h3>Generalization and threshold metrics</h3>{score_table}</div>
  <div class="card table-card"><h3>Overfitting, calibration, and timing</h3>{reliability_table}</div>
  <details><summary>Fold-level results (all {len(records)} model-fold evaluations)</summary>{fold_table}</details>
  {tuning_table}
</section>"""


def render_html(results: dict[str, Any]) -> str:
    dataset_summaries = []
    for item in results["dataset_summaries"]:
        records = [record for record in results["fold_results"] if record["dataset_key"] == item["dataset_key"]]
        copied = dict(item)
        copied["model_summaries"] = enrich_summaries_with_fold_metrics(item["model_summaries"], records)
        dataset_summaries.append(copied)
    overall = aggregate_overall(dataset_summaries)
    best = overall[0]
    tuned = [item for item in overall if item["model"].endswith(" Tuned")]
    total_tuned_time = sum(item["timing_seconds"]["total"] for item in tuned)
    dataset_rows = []
    for index, item in enumerate(dataset_summaries, start=1):
        dataset = item["dataset"]
        class_counts = ", ".join(f"{html.escape(str(label))}: {count}" for label, count in dataset["class_counts"].items())
        dataset_rows.append([
            cell(str(index), index),
            cell(f'<a href="#dataset-{index}">{html.escape(dataset["name"])}</a>', dataset["name"], "model-name"),
            cell(f'<a href="{html.escape(dataset["source_url"])}">OpenML {dataset["openml_data_id"]}</a>'),
            cell(f'{dataset["rows"]:,}', dataset["rows"]),
            cell(f'{dataset["numeric_feature_count"]} numeric / {dataset["categorical_feature_count"]} categorical', dataset["raw_feature_count"]),
            cell(html.escape(dataset["positive_class"])),
            cell(class_counts),
            cell(html.escape(dataset["sampling"])),
        ])
    overall_table = sortable_table(
        "overall-metrics",
        [("Model", False, ""), ("Tasks", True, ""), ("ROC-AUC", True, ""), ("Avg. precision", True, ""), ("Balanced acc.", True, ""), ("F1", True, ""), ("Precision", True, ""), ("Recall", True, ""), ("Specificity", True, ""), ("FPR", True, ""), ("Log loss", True, ""), ("Brier", True, ""), ("Train AUC", True, ""), ("AUC gap", True, ""), ("Tuned AUC delta", True, ""), ("Tune time", True, ""), ("Total time", True, "")],
        overall_rows(overall),
        "Unweighted means across five datasets. Standard deviations here quantify variation between datasets, not between folds.",
    )
    quick_overview_table = sortable_table(
        "quick-overview-metrics",
        [
            ("Model", False, ""), ("ROC-AUC", True, ""), ("Avg. precision", True, ""),
            ("Balanced acc.", True, ""), ("F1", True, ""), ("AUC gap", True, ""),
            ("Log loss", True, ""), ("Brier", True, ""), ("Tuned AUC delta", True, ""),
            ("Total time", True, ""),
        ],
        quick_overview_rows(overall),
        "Quick macro-average view across all five datasets. Values are mean +/- between-dataset standard deviation; timing is informational and is not heat-mapped.",
    )
    datasets_table = sortable_table(
        "dataset-index",
        [("#", True, ""), ("Dataset", False, ""), ("Source", False, ""), ("Rows", True, ""), ("Feature mix", True, ""), ("Positive class", False, ""), ("Class counts", False, ""), ("Sampling", False, "")],
        dataset_rows,
        "Five reproducibly sampled OpenML binary-classification tasks.",
    )
    cards = "".join(dataset_card(item, results["fold_results"], index) for index, item in enumerate(dataset_summaries))
    config = html.escape(json.dumps(results["model_configurations"], indent=2))
    env = html.escape(json.dumps(results["environment"], indent=2))
    protocol = results["evaluation_protocol"]
    profile = results["profiling_seconds"]
    tuned_protocol = protocol.get("tuned_model_protocol", "No tuned variants are included in this report.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detailed tabular benchmark report</title>
<style>
:root {{ --ink:#18233a; --muted:#5e6c84; --line:#dce3ee; --soft:#f5f7fb; --panel:#ffffff; --navy:#14213d; --blue:#2764d8; --green:#087443; --green-bg:#dff7eb; --mid:#9a6700; --mid-bg:#fff3cf; --red:#bb2733; --red-bg:#ffe6e8; --purple:#6246ea; }}
* {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }} body {{ margin:0; font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:var(--soft); }}
a {{ color:var(--blue); text-decoration:none; }} a:hover {{ text-decoration:underline; }} code {{ font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }}
header {{ color:#fff; background:linear-gradient(120deg,#14213d,#1f4c8d 58%,#2764d8); padding:44px max(24px,calc((100vw - 1440px)/2)); }} header .content {{ max-width:1392px; margin:auto; }} .badge {{ display:inline-block; padding:4px 10px; font-size:12px; font-weight:700; border:1px solid #a8c8ff77; border-radius:999px; background:#ffffff19; }} h1 {{ margin:12px 0 6px; font-size:clamp(28px,4vw,42px); line-height:1.1; letter-spacing:-.035em; }} h2 {{ margin:0; font-size:26px; letter-spacing:-.025em; }} h3 {{ margin:0 0 12px; font-size:17px; }} .subtitle {{ max-width:920px; margin:0; color:#dbe9ff; font-size:16px; }}
.layout {{ display:grid; grid-template-columns:205px minmax(0,1fr); gap:28px; max-width:1392px; padding:30px 24px 64px; margin:auto; }} .sidebar {{ position:sticky; top:16px; align-self:start; }} .sidebar strong {{ display:block; margin-bottom:8px; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }} .sidebar a {{ display:block; padding:6px 0; color:var(--muted); font-size:13px; }} .sidebar a:hover {{ color:var(--blue); }} main {{ min-width:0; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:26px; }} .kpi {{ padding:17px; border:1px solid var(--line); border-radius:12px; background:var(--panel); box-shadow:0 1px 2px #14213d0a; }} .kpi-label {{ display:block; color:var(--muted); font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.045em; }} .kpi-value {{ display:block; margin-top:6px; font-size:23px; font-weight:750; letter-spacing:-.02em; }} .kpi-note {{ display:block; margin-top:3px; color:var(--muted); font-size:12px; }}
.section {{ margin-top:30px; scroll-margin-top:16px; }} .section > p {{ color:var(--muted); margin:8px 0 14px; }} .card {{ border:1px solid var(--line); border-radius:12px; background:var(--panel); box-shadow:0 1px 2px #14213d0a; }} .table-card {{ margin-top:14px; padding:18px; }} .dataset-card {{ padding:25px 0 4px; margin-top:44px; border-top:2px solid var(--line); scroll-margin-top:16px; }} .dataset-heading {{ display:flex; gap:20px; justify-content:space-between; align-items:flex-start; }} .dataset-heading h2 {{ margin:1px 0 4px; }} .eyebrow {{ margin:0; color:var(--blue); font-size:11px; font-weight:750; letter-spacing:.09em; text-transform:uppercase; }} .lede {{ margin:0; color:var(--muted); }} .source-link {{ padding:7px 10px; white-space:nowrap; border:1px solid #c9d8f5; border-radius:8px; background:#f5f8ff; font-size:12px; }} .dataset-meta {{ display:flex; flex-wrap:wrap; gap:7px; margin-top:14px; }} .meta-chip {{ padding:4px 8px; border-radius:999px; background:#eef2f8; color:#3e4b60; font-size:12px; }} .note {{ color:var(--muted); font-size:12px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px; padding:12px 14px; border:1px solid var(--line); border-radius:10px; background:#fbfcff; color:var(--muted); font-size:12px; }} .legend span {{ padding:2px 7px; border-radius:5px; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:9px; }} table {{ width:100%; border-collapse:collapse; font-size:12px; white-space:nowrap; }} caption {{ padding:10px 12px; color:var(--muted); text-align:left; white-space:normal; font-size:12px; background:#fbfcff; border-bottom:1px solid var(--line); }} th {{ position:sticky; top:0; z-index:1; padding:0; color:#526177; background:#f1f4f9; text-align:left; font-size:10px; letter-spacing:.045em; text-transform:uppercase; }} th button {{ width:100%; padding:9px 10px; border:0; color:inherit; background:transparent; cursor:pointer; font:inherit; text-align:left; }} th button:hover {{ color:var(--blue); background:#e9f0ff; }} .sort-mark {{ float:right; color:#97a6bc; font-size:12px; }} td {{ padding:8px 10px; border-top:1px solid var(--line); vertical-align:top; }} tbody tr:hover td {{ background:#f7faff; }} .model-name {{ font-weight:700; }} .pm {{ color:var(--muted); font-weight:500; }} .time-cell {{ color:#56657b; font-variant-numeric:tabular-nums; }} .profile-cell {{ color:var(--purple); font-weight:650; }}
.col-best {{ color:#075d38; background:#c9f1dc; }} .col-good {{ color:#117045; background:#e2f7e9; }} .col-mid {{ color:#8a5c00; background:#fff3cf; }} .col-low {{ color:#a92330; background:#ffe4e7; }} .col-positive {{ color:#087443; background:#dff7eb; font-weight:700; }} .col-negative {{ color:#bb2733; background:#ffe6e8; font-weight:700; }} .col-neutral {{ color:var(--muted); background:#eef2f8; }}
details {{ margin-top:14px; border:1px solid var(--line); border-radius:10px; background:var(--panel); padding:0 16px 16px; }} summary {{ padding:14px 0; cursor:pointer; color:var(--blue); font-weight:700; }} pre {{ overflow:auto; margin:0; padding:14px; border-radius:8px; color:#dde9ff; background:#111b30; font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }}
footer {{ padding-top:34px; color:var(--muted); font-size:12px; }} @media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} .sidebar {{ display:none; }} .kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }} @media (max-width:520px) {{ header {{ padding:30px 20px; }} .layout {{ padding:20px 14px 40px; }} .kpis {{ grid-template-columns:1fr; }} .dataset-heading {{ display:block; }} .source-link {{ display:inline-block; margin-top:12px; }} }}
</style></head><body>
<header><div class="content"><span class="badge">Completed &bull; Baselines + tuned tree variants</span><h1>Detailed tabular benchmark</h1><p class="subtitle">Five OpenML binary-classification tasks, shared 5-fold outer cross-validation, calibration and threshold metrics, train-vs-held-out gaps, profile-level tuning audit, and reproducible timing.</p></div></header>
<div class="layout"><nav class="sidebar"><strong>On this page</strong><a href="#overview">Overview</a><a href="#quick-overview">Quick overview</a><a href="#datasets">Datasets</a><a href="#overall">Overall metrics</a>{''.join(f'<a href="#dataset-{index}">{html.escape(item["dataset"]["name"])}</a>' for index, item in enumerate(dataset_summaries))}<a href="#method">Method & reproducibility</a></nav><main>
<section id="overview" class="kpis"><div class="kpi"><span class="kpi-label">Best overall model</span><span class="kpi-value">{html.escape(best['model'])}</span><span class="kpi-note">Mean ROC-AUC {metric(best['validation_metrics']['roc_auc']['mean'])} across five tasks</span></div><div class="kpi"><span class="kpi-label">Completed evaluations</span><span class="kpi-value">{sum(1 for record in results['fold_results'] if record['status'] == 'success')}/{len(results['fold_results'])}</span><span class="kpi-note">All baseline and tuned model-fold records</span></div><div class="kpi"><span class="kpi-label">Evaluation protocol</span><span class="kpi-value">5 x 5-fold CV</span><span class="kpi-note">Five datasets; held-out outer folds never tune parameters</span></div><div class="kpi"><span class="kpi-label">Tuned model work</span><span class="kpi-value">{seconds(results['profiling_seconds'].get('tuned_model_work', 0.0))}</span><span class="kpi-note">{len(tuned)} tuned model families, compact inner selection</span></div></section>
<section id="quick-overview" class="section"><h2>Quick aggregate overview</h2><p>Macro averages give each dataset equal weight. Colours compare models <strong>only within the same column</strong>; green is better, including lower-is-better loss and overfitting-gap columns.</p><div class="legend"><strong>Per-column colour:</strong><span class="col-best">best relative result</span><span class="col-good">above peers</span><span class="col-mid">middle</span><span class="col-low">weakest relative result</span><span class="col-positive">tuned AUC improvement</span><span class="col-negative">tuned AUC regression</span></div><div class="card table-card">{quick_overview_table}</div></section>
<section id="datasets" class="section"><h2>Dataset inventory</h2><p>Each task was deterministically stratified-sampled to 500 rows when larger. The task-level tables below retain their own metrics rather than pooling rows across unlike datasets.</p>{datasets_table}</section>
<section id="overall" class="section"><h2>Detailed overall comparison</h2><p>Scores are unweighted means across datasets; sort any column to change the view. Cell colours are relative to the values visible in that individual metric column. ROC-AUC and average precision are threshold-independent; accuracy, F1, precision, recall, specificity, FPR, and Youden's J use the fixed 0.5 probability threshold.</p><div class="card table-card">{overall_table}</div></section>
<section id="per-dataset" class="section"><h2>Per-dataset analysis</h2><p>Every section includes discrimination, fixed-threshold behaviour, calibration, overfitting, timing, raw outer-fold measurements, and the tuning choices behind each <em>Tuned</em> model.</p>{cards}</section>
<section id="method" class="section"><h2>Method and reproducibility</h2><div class="card table-card"><h3>What is measured</h3><ul><li><strong>Generalization:</strong> held-out outer-fold ROC-AUC, with other held-out metrics alongside it.</li><li><strong>Overfitting gap:</strong> training ROC-AUC minus held-out ROC-AUC. A lower gap is generally better, conditional on a competitive held-out AUC.</li><li><strong>Balanced accuracy:</strong> (recall + specificity) / 2. <strong>FPR:</strong> 1 - specificity. <strong>Youden's J:</strong> recall + specificity - 1.</li><li><strong>Calibration:</strong> log loss and Brier score; lower is better.</li><li><strong>Timing:</strong> serial wall-clock time; tuned models include profile-selection time.</li></ul><p class="note">{html.escape(protocol['strategy'])} {html.escape(protocol.get('sampling', ''))}</p><p class="note">{html.escape(protocol.get('tabpfn_execution', ''))}</p><p class="note">{html.escape(tuned_protocol)}</p><p><strong>Baseline model work:</strong> {seconds(profile.get('all_model_work'))}. <strong>Tuned model work:</strong> {seconds(profile.get('tuned_model_work', 0.0))}. <span class="note">Final-invocation wall time is {seconds(profile.get('end_to_end_total'))}; checkpointed prior work is excluded from that one invocation.</span></p></div><details><summary>Model configurations</summary><pre>{config}</pre></details><details><summary>Environment</summary><pre>{env}</pre></details></section>
<footer>Generated {html.escape(results['generated_at_utc'])}. This report is standalone: sorting and colour coding run locally in the browser.</footer>
</main></div>
<script>
document.querySelectorAll('.sort-table th[data-col]').forEach((header) => {{
  header.addEventListener('click', () => {{
    const table = header.closest('table'); const index = Number(header.dataset.col); const numeric = header.dataset.numeric === 'true';
    const ascending = header.dataset.order !== 'asc';
    table.querySelectorAll('th[data-col]').forEach((item) => {{ item.dataset.order = ''; item.querySelector('.sort-mark').textContent = '↕'; }});
    header.dataset.order = ascending ? 'asc' : 'desc'; header.querySelector('.sort-mark').textContent = ascending ? '↑' : '↓';
    const rows = [...table.tBodies[0].rows];
    rows.sort((a, b) => {{
      const av = a.cells[index]?.dataset.sort ?? ''; const bv = b.cells[index]?.dataset.sort ?? '';
      if (numeric) {{ const an = Number(av); const bn = Number(bv); return (Number.isNaN(an) ? -Infinity : an) - (Number.isNaN(bn) ? -Infinity : bn); }}
      return av.localeCompare(bv, undefined, {{ numeric: true, sensitivity: 'base' }});
    }});
    if (!ascending) rows.reverse(); rows.forEach((row) => table.tBodies[0].appendChild(row));
  }});
}});
</script></body></html>"""


def build_multidataset_report(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(results), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/multidataset_benchmark_report.html"))
    args = parser.parse_args()
    build_multidataset_report(json.loads(args.results_json.read_text(encoding="utf-8")), args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
