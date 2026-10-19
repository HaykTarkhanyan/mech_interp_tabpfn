#!/usr/bin/env python3
"""Build the transfer-pilot report from a results JSON written by transfer_pilot.py (no recomputation).

For each builder world (feature generator x target model) and each replicate sample D, compare the world's
explainer ranking with reality's ranking (mean error over reality's task replicates):
  Kendall tau      rank agreement, 1 = same order, 0 = unrelated (random choice has expectation 0)
  top-1 match      does the world pick the same best explainer as reality?
  regret           how much worse, in reality, the world's pick is than reality's best (relative)
The oracle rows (reality with a fresh sample) show the agreement that task noise alone allows.
Writes results/transfer_pilot/<tag>_summary.json and <tag>_report.html, and opens nothing.

Usage (from synthetic_data/, ~5 s):
    .venv/Scripts/python.exe scripts/transfer_report.py --tag smoke
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "transfer_pilot"
RED, BLUE, ORANGE, GRAY = "#D90012", "#0033A0", "#F2A800", "#8a8f99"


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "transfer_report.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])


def summarize(res: dict, level: str) -> dict:
    """Agreement of every world with reality, for errors at `level` ('err_data' or 'err_model')."""
    ex = res["explainers"]
    real = np.array([[t[level][e] for e in ex] for t in res["tasks"] if t["role"] == "reality"])
    real_mean = real.mean(0)
    best = real_mean.min()
    rows = {}
    for t in res["tasks"]:
        if t["role"] == "reality":
            continue
        err = np.array([t[level][e] for e in ex])
        tau = kendalltau(err, real_mean).statistic
        pick = int(err.argmin())
        rows.setdefault(t["world"] if t["role"] == "builder" else "oracle (fresh reality sample)", []).append(
            {"tau": float(tau), "top1": bool(pick == int(real_mean.argmin())),
             "regret": float((real_mean[pick] - best) / best), "pick": ex[pick]})
    worlds = {}
    for w, rs in rows.items():
        taus = np.array([r["tau"] for r in rs])
        worlds[w] = {"n": len(rs), "tau_mean": float(taus.mean()), "tau_sd": float(taus.std(ddof=1)) if len(rs) > 1 else None,
                     "top1_rate": float(np.mean([r["top1"] for r in rs])),
                     "regret_mean": float(np.mean([r["regret"] for r in rs])), "picks": [r["pick"] for r in rs]}
    return {
        "reality_mean_error": dict(zip(ex, map(float, real_mean))),
        "reality_error_sd": dict(zip(ex, map(float, real.std(0, ddof=1)))),
        "reality_ranking": [ex[i] for i in np.argsort(real_mean)],
        "random_regret": float(np.mean((real_mean - best) / best)),
        "worlds": worlds,
    }


def bar(x, y, title, ytitle, colors, ref=None, ref_label=None, fmt=".2f", err=None):
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors, text=[f"{v:{fmt}}" for v in y], textposition="outside",
                           error_y=dict(type="data", array=err, visible=err is not None)))
    if ref is not None:
        fig.add_hline(y=ref, line_dash="dash", line_color=GRAY, annotation_text=ref_label, annotation_position="top right")
    fig.update_layout(title=title, yaxis_title=ytitle, template="plotly_white", height=430,
                      margin=dict(l=60, r=30, t=60, b=140), xaxis_tickangle=-35)
    return fig


def section(sid: str, title: str, body: str) -> str:
    return f'<details open id="{sid}"><summary><h2>{title}</h2></summary>{body}</details>'


def build_html(res: dict, summ: dict, tag: str) -> str:
    data = summ["data_level"]
    worlds = sorted(data["worlds"], key=lambda w: (w.startswith("oracle"), w))
    colors = [GRAY if w.startswith("oracle") else (BLUE if w.startswith("bootstrap") else RED if w.startswith("independent") else ORANGE)
              for w in worlds]
    figs = [
        bar(res["explainers"], [data["reality_mean_error"][e] for e in res["explainers"]],
            "Explainer error in reality (data level), mean over reality's task replicates", "relative error",
            [BLUE] * len(res["explainers"]), err=[data["reality_error_sd"][e] for e in res["explainers"]]),
        bar(worlds, [data["worlds"][w]["tau_mean"] for w in worlds],
            "Rank agreement with reality, by builder world (data level)", "mean Kendall tau", colors,
            ref=0, ref_label="random = 0", err=[data["worlds"][w]["tau_sd"] or 0 for w in worlds]),
        bar(worlds, [data["worlds"][w]["regret_mean"] for w in worlds],
            "Regret in reality of each world's chosen explainer (data level)", "relative regret", colors,
            ref=data["random_regret"], ref_label=f"random pick = {data['random_regret']:.2f}"),
    ]
    divs = [f.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False)) for i, f in enumerate(figs)]
    model = summ["model_level"]
    rows = "".join(
        f"<tr><td>{w}</td><td>{data['worlds'][w]['tau_mean']:.2f}</td><td>{model['worlds'][w]['tau_mean']:.2f}</td>"
        f"<td>{data['worlds'][w]['top1_rate']:.2f}</td><td>{data['worlds'][w]['regret_mean']:.2f}</td></tr>" for w in worlds)
    table = ('<table><colgroup><col style="width:40%"><col style="width:15%"><col style="width:15%"><col style="width:15%">'
             '<col style="width:15%"></colgroup><thead><tr><th>World</th><th>tau, data level</th><th>tau, model level</th>'
             f'<th>top-1 match</th><th>regret</th></tr></thead><tbody>{rows}</tbody></table>')
    cfg = res["config"]
    design = (f"<p>Reality: {cfg['dataset']} feature rows, a random forest fitted to the real target as the true function, "
              f"noise sd {res['reality_noise_sd']:.3f}. Builders see one sample of {cfg['n_D']} rows and build each "
              f"generator x target world from it. In every world a gradient-boosting model is trained on {cfg['n_train']} rows "
              f"and {cfg['n_explain']} points are explained by {len(res['explainers'])} explainers, scored against "
              f"interventional SHAP of the world's true function (data level) and of the trained model (model level). "
              f"{res['reps']} replicate samples; reality's ranking averages {cfg['reality_task_reps']} task replicates. "
              f"Run {res['created']}, {res['runtime_s']:.0f} s, seed {res['seed']}.</p>"
              f"<p>Reality's ranking (best first): {', '.join(data['reality_ranking'])}.</p>")
    oracle = data["worlds"].get("oracle (fresh reality sample)", {})
    caveat = (f"<p><b>Read every bar against the gray oracle bar.</b> The oracle is reality itself with a fresh "
              f"sample, so its agreement (mean tau {oracle.get('tau_mean', float('nan')):.2f}) is what task noise alone "
              f"allows. A builder above it is not evidence of transfer unless the replicates are enough to separate "
              f"them; with {res['reps']} replicate(s) and one task per builder world, they are not.</p>")
    secs = [("design", "1. Design", design), ("reality", "2. Reality's ranking", divs[0]),
            ("agreement", "3. Do builder worlds reproduce it?", caveat + divs[1] + table),
            ("regret", "4. What a wrong pick costs", divs[2])]
    toc = "".join(f'<li><a href="#{s}">{t}</a></li>' for s, t, _ in secs)
    css = ("body{font:16px/1.55 -apple-system,'Segoe UI',Roboto,Arial,sans-serif;max-width:1000px;margin:0 auto;"
           "padding:0 16px 60px;background:#fbfaf7;color:#1d1f24}details{background:#fff;border:1px solid #dcdfe5;"
           "border-radius:10px;margin:16px 0;padding:4px 16px 12px}summary h2{display:inline;font-size:1.2rem}"
           "table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:.92rem}"
           "th,td{border-bottom:1px solid #dcdfe5;padding:5px 6px;text-align:left}nav{background:#fff;"
           "border:1px solid #dcdfe5;border-radius:10px;padding:8px 16px}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width, initial-scale=1'><title>Transfer pilot {tag}</title><style>{css}</style></head>"
            f"<body><h1>Do data-fitted benchmarks reproduce explainer rankings? Pilot: {tag}</h1>"
            f"<nav><b>Contents</b><ol>{toc}</ol></nav>"
            + "".join(section(s, t, b) for s, t, b in secs) + "</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    setup_logging()
    start = time.time()
    res = json.loads((OUT_DIR / f"{args.tag}.json").read_text(encoding="utf-8"))
    summ = {"source": f"{args.tag}.json", "data_level": summarize(res, "err_data"), "model_level": summarize(res, "err_model")}
    (OUT_DIR / f"{args.tag}_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    html = build_html(res, summ, args.tag)
    (OUT_DIR / f"{args.tag}_report.html").write_text(html, encoding="utf-8")
    for w, s in sorted(summ["data_level"]["worlds"].items()):
        logging.info(f"{w:32s} tau={s['tau_mean']:+.2f} top1={s['top1_rate']:.2f} regret={s['regret_mean']:.2f}")
    logging.info(f"reality ranking: {summ['data_level']['reality_ranking']}")
    logging.info(f"Wrote {args.tag}_summary.json and {args.tag}_report.html in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
