"""Build a standalone, browser-viewable HTML report from benchmark JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def metric(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1:
        return f"{value * 1_000:.0f} ms"
    if value < 60:
        return f"{value:.2f} s"
    return f"{value / 60:.2f} min"


def table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<div class=\"table-wrap\"><table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>"


def score_bar(value: float) -> str:
    width = min(100, max(0, value * 100))
    return f'<div class="bar"><span style="width:{width:.1f}%"></span></div>'


def render_html(results: dict[str, Any]) -> str:
    dataset = results["dataset"]
    summaries = results["model_summaries"]
    profiling = results["profiling_seconds"]
    class_counts = ", ".join(
        f"{html.escape(str(label))}: {count}" for label, count in sorted(dataset["class_counts"].items())
    )
    dataset_rows = [
        ["Dataset", html.escape(dataset["name"])],
        ["Source", f'<a href="{html.escape(dataset["source_url"])}">OpenML data ID {dataset["openml_data_id"]}</a>'],
        ["Task", "Binary classification: predict good credit risk"],
        ["Rows / raw features", f'{dataset["rows"]:,} / {dataset["raw_feature_count"]}'],
        ["Feature types", f'{dataset["numeric_feature_count"]} numeric, {dataset["categorical_feature_count"]} categorical'],
        ["Class distribution", class_counts],
        ["Missing cells", f'{dataset["missing_cells"]:,}'],
        ["Data access", html.escape(dataset["load_source"])],
    ]

    summary_rows: list[list[str]] = []
    for rank, summary in enumerate(summaries, start=1):
        validation = summary["validation_metrics"]
        train = summary["train_metrics"]
        auc = validation.get("roc_auc", {})
        auc_value = auc.get("mean")
        gap = summary["overfitting_gap_roc_auc"].get("mean")
        summary_rows.append(
            [
                str(rank),
                html.escape(summary["model"]),
                f'<strong>{metric(auc_value)} ± {metric(auc.get("std"))}</strong>',
                score_bar(auc_value) if auc_value is not None else "—",
                f'{metric(train.get("roc_auc", {}).get("mean"))} ± {metric(train.get("roc_auc", {}).get("std"))}',
                f'{metric(gap)} ± {metric(summary["overfitting_gap_roc_auc"].get("std"))}',
                f'{metric(validation.get("average_precision", {}).get("mean"))} ± {metric(validation.get("average_precision", {}).get("std"))}',
                f'{metric(validation.get("accuracy", {}).get("mean"))} ± {metric(validation.get("accuracy", {}).get("std"))}',
                f'{metric(validation.get("f1", {}).get("mean"))} ± {metric(validation.get("f1", {}).get("std"))}',
            ]
        )

    timing_rows = []
    for summary in summaries:
        timing = summary["timing_seconds"]
        timing_rows.append(
            [
                html.escape(summary["model"]),
                seconds(timing.get("fit", {}).get("mean")),
                seconds(timing.get("predict_validation", {}).get("mean")),
                seconds(timing.get("total", {}).get("mean")),
                seconds(timing.get("total", {}).get("total")),
            ]
        )

    fold_rows = []
    for record in results["fold_results"]:
        if record["status"] == "success":
            validation = record["validation_metrics"]
            fold_rows.append(
                [
                    html.escape(record["model"]),
                    str(record["fold"]),
                    metric(record["train_metrics"]["roc_auc"]),
                    metric(validation["roc_auc"]),
                    metric(record["overfitting_gap_roc_auc"]),
                    metric(validation["accuracy"]),
                    metric(validation["f1"]),
                    metric(validation["log_loss"]),
                    metric(validation["brier_score"]),
                    seconds(record["timing_seconds"].get("fit")),
                    seconds(record["timing_seconds"].get("predict_validation")),
                ]
            )
        else:
            fold_rows.append(
                [
                    html.escape(record["model"]),
                    str(record["fold"]),
                    "FAILED",
                    html.escape(record.get("error", "Unknown error")),
                    "", "", "", "", "", "", "",
                ]
            )

    profile_rows = [
        ["Dataset load/download", seconds(profiling.get("dataset_load"))],
        ["CV split construction", seconds(profiling.get("cross_validation_setup"))],
        ["All model fits and predictions", seconds(profiling.get("model_benchmark_total"))],
        ["HTML report build", seconds(profiling.get("html_report_build"))],
        ["End-to-end benchmark", seconds(profiling.get("end_to_end_total"))],
    ]
    config_json = html.escape(json.dumps(results["model_configurations"], indent=2))
    environment_json = html.escape(json.dumps(results["environment"], indent=2))
    feature_names = ", ".join(html.escape(str(name)) for name in dataset["feature_names"])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TabPFN benchmark — {html.escape(dataset['name'])}</title>
  <style>
    :root {{ --ink:#172033; --muted:#5b667a; --line:#dde3ec; --panel:#fff; --ground:#f5f7fb; --accent:#2251cc; --good:#15734c; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font:15px/1.5 Inter,Segoe UI,Arial,sans-serif; color:var(--ink); background:var(--ground); }}
    main {{ max-width:1320px; margin:0 auto; padding:32px 24px 56px; }} h1 {{ margin:0; font-size:30px; letter-spacing:-.03em; }} h2 {{ margin:34px 0 12px; font-size:21px; }}
    .lede {{ margin:8px 0 0; color:var(--muted); }} .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:20px; box-shadow:0 1px 2px #18223a08; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:16px; }} .note {{ color:var(--muted); font-size:13px; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }} th {{ text-align:left; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; background:#f8faff; }} th,td {{ border-bottom:1px solid var(--line); padding:10px; vertical-align:top; }} tr:last-child td {{ border-bottom:0; }} .table-wrap {{ overflow-x:auto; }}
    .bar {{ width:130px; height:8px; background:#e6ebf5; border-radius:999px; overflow:hidden; }} .bar span {{ display:block; height:100%; background:var(--accent); border-radius:999px; }}
    a {{ color:var(--accent); }} code,pre {{ font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }} pre {{ overflow:auto; background:#101827; color:#e6edf7; padding:14px; border-radius:8px; }}
    details {{ margin-top:14px; }} summary {{ cursor:pointer; color:var(--accent); }} .pill {{ display:inline-block; color:var(--good); background:#e9f8f0; border-radius:999px; padding:3px 9px; font-size:12px; font-weight:600; }}
  </style>
</head>
<body><main>
  <header><span class="pill">Completed benchmark</span><h1>Tabular classification benchmark</h1><p class="lede">Generated {html.escape(results['generated_at_utc'])}. Ranking uses mean held-out-fold ROC-AUC.</p></header>
  <h2>Dataset</h2><section class="card">{table(["Field", "Value"], dataset_rows)}</section>
  <h2>Generalization and overfitting</h2><section class="card"><p class="note">Validation metrics are means ± sample standard deviations across shared outer folds. Overfitting gap = train ROC-AUC − held-out ROC-AUC; smaller is generally better, provided held-out AUC remains strong.</p>{table(["Rank", "Model", "Validation ROC-AUC", "AUC", "Train ROC-AUC", "Overfit gap", "Avg. precision", "Accuracy", "F1"], summary_rows)}</section>
  <h2>Time profile</h2><section class="grid"><div class="card">{table(["Stage", "Elapsed"], profile_rows)}</div><div class="card"><p class="note">Per-model times are serial wall-clock measurements. Classical models use one CPU thread to avoid resource contention.</p>{table(["Model", "Mean fit", "Mean validation predict", "Mean fold total", "All folds"], timing_rows)}</div></section>
  <h2>Per-fold results</h2><section class="card">{table(["Model", "Fold", "Train AUC", "Validation AUC", "AUC gap", "Accuracy", "F1", "Log loss", "Brier", "Fit", "Validation predict"], fold_rows)}</section>
  <h2>Protocol and reproducibility</h2><section class="card"><p>{html.escape(results['evaluation_protocol']['strategy'])}; seed {results['evaluation_protocol']['random_seed']}.</p><p class="note">{html.escape(results['evaluation_protocol']['validation_use'])}</p><p class="note">{html.escape(results['evaluation_protocol']['execution_policy'])}</p><details><summary>Model configurations</summary><pre>{config_json}</pre></details><details><summary>Environment</summary><pre>{environment_json}</pre></details><details><summary>Feature names ({len(dataset['feature_names'])})</summary><p>{feature_names}</p></details></section>
</main></body></html>"""


def build_html_report(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(results), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_report.html"))
    args = parser.parse_args()
    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    build_html_report(results, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
