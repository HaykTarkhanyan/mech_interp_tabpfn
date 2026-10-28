#!/usr/bin/env python3
"""Summary JSON and HTML report for a transfer_pilot_v2.py run (no recomputation of the experiment).

Primary metric (fixed before the full run): at model level, the share of reality's *resolved* explainer pairs
(pairs whose paired 95% interval over reality's tasks excludes zero) that a world orders the same way, using the
world's mean error over its K tasks. Secondary: Kendall tau_b against reality's mean ranking, regret of the
world's chosen explainer in reality, and top-1 match when reality's best explainer is itself resolved.
Intervals are percentile bootstraps over replicate samples D. The oracle (fresh reality tasks) is the ceiling
set by task noise; the null (a generic world that never sees D) is what a data-agnostic benchmark achieves.

Usage (from synthetic_data/, a few seconds):
    .venv/Scripts/python.exe scripts/transfer_report_v2.py --tag v2_rf
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.stats import kendalltau, t as t_dist

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "transfer_pilot"
SEED = 509
RED, BLUE, ORANGE, GRAY = "#D90012", "#0033A0", "#F2A800", "#8a8f99"
N_BOOT = 2000


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "transfer_report_v2.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])


def resolved_pairs(err: np.ndarray) -> dict:
    """Pairs (a, b) whose paired 95% t-interval over tasks excludes zero -> +1 if a has lower error, else -1."""
    k, j = err.shape
    if k < 3:
        raise ValueError(f"need at least 3 reality tasks to resolve pairs, got {k}")
    q = t_dist.ppf(0.975, k - 1)
    out = {}
    for a in range(j):
        for b in range(a + 1, j):
            d = err[:, a] - err[:, b]
            half = q * d.std(ddof=1) / np.sqrt(k)
            if abs(d.mean()) > half:
                out[(a, b)] = 1 if d.mean() < 0 else -1
    return out


def world_scores(mean_err: np.ndarray, real_mean: np.ndarray, pairs: dict, best_resolved: bool) -> dict:
    agree = [np.sign(mean_err[b] - mean_err[a]) == s for (a, b), s in pairs.items()]
    pick = int(mean_err.argmin())
    return {
        "pair_agreement": float(np.mean(agree)) if agree else float("nan"),
        "tau_b": float(kendalltau(mean_err, real_mean).statistic),
        "regret": float((real_mean[pick] - real_mean.min()) / real_mean.min()),
        "top1": float(pick == int(real_mean.argmin())) if best_resolved else float("nan"),
    }


def boot_ci(values: list, rng: np.random.Generator) -> list:
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or np.isnan(v).all():
        return [float("nan"), float("nan")]
    means = [np.nanmean(rng.choice(v, len(v))) for _ in range(N_BOOT)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(res: dict, level: str) -> dict:
    ex = res["explainers"]
    rng = np.random.default_rng(SEED)
    real = np.array([[t[level][e] for e in ex] for t in res["tasks"] if t["role"] == "reality"])
    real_mean = real.mean(0)
    pairs = resolved_pairs(real)
    best = int(real_mean.argmin())
    best_resolved = all(((min(best, o), max(best, o)) in pairs) for o in range(len(ex)) if o != best)
    groups: dict = {}
    for t in res["tasks"]:
        if t["role"] == "reality":
            continue
        name = {"oracle": "oracle (fresh reality tasks)", "null": "null (generic, ignores D)"}.get(t["role"], t["world"])
        groups.setdefault(name, {}).setdefault(t["rep"], []).append([t[level][e] for e in ex])
    worlds = {}
    for name, reps in groups.items():
        per_rep = [world_scores(np.mean(v, axis=0), real_mean, pairs, best_resolved) for v in reps.values()]
        worlds[name] = {m: {"mean": float(np.nanmean([s[m] for s in per_rep])), "ci95": boot_ci([s[m] for s in per_rep], rng)}
                        for m in ("pair_agreement", "tau_b", "regret", "top1")}
        worlds[name]["n_reps"] = len(per_rep)
        worlds[name]["k_tasks"] = int(np.mean([len(v) for v in reps.values()]))
    return {
        "reality_mean_error": dict(zip(ex, map(float, real_mean))),
        "reality_error_se": dict(zip(ex, map(float, real.std(0, ddof=1) / np.sqrt(len(real))))),
        "reality_ranking": [ex[i] for i in np.argsort(real_mean)],
        "n_pairs": len(ex) * (len(ex) - 1) // 2, "n_resolved_pairs": len(pairs), "best_resolved": best_resolved,
        "random_regret": float(np.mean((real_mean - real_mean.min()) / real_mean.min())),
        "worlds": worlds,
    }


def estimand_sensitivity(res: dict) -> dict | None:
    """v3 runs: reality tasks scored against both conditional (err_model) and interventional SHAP of the model."""
    real = [t for t in res["tasks"] if t["role"] == "reality" and "err_model_interventional" in t]
    if not real:
        return None
    ex = res["explainers"]
    cond = np.array([[t["err_model"][e] for e in ex] for t in real])
    intv = np.array([[t["err_model_interventional"][e] for e in ex] for t in real])
    p_cond, p_int = resolved_pairs(cond), resolved_pairs(intv)
    both = set(p_cond) & set(p_int)
    flips = sorted(f"{ex[a]} vs {ex[b]}" for a, b in both if p_cond[(a, b)] != p_int[(a, b)])
    return {
        "n_reality_tasks": len(real),
        "mean_error_conditional": dict(zip(ex, map(float, cond.mean(0)))),
        "mean_error_interventional": dict(zip(ex, map(float, intv.mean(0)))),
        "ranking_conditional": [ex[i] for i in np.argsort(cond.mean(0))],
        "ranking_interventional": [ex[i] for i in np.argsort(intv.mean(0))],
        "kendall_tau_between_rankings": float(kendalltau(cond.mean(0), intv.mean(0)).statistic),
        "pairs_resolved_under_both": len(both), "pairs_flipped": len(flips), "flipped_pairs": flips,
        "estimand_gap_mean": float(np.mean([t["estimand_gap"] for t in real])),
        "truth_mc_noise_mean": float(np.mean([t["truth_mc_noise"] for t in real])) if "truth_mc_noise" in real[0] else None,
    }


def timing(res: dict) -> dict:
    out = {}
    for t in res["tasks"]:
        key = t["role"] if t["role"] != "builder" else t["world"]
        out.setdefault(key, []).append(t["seconds"]["total"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def bar_ci(names, metric, data, title, ytitle, ref=None, ref_label=None):
    vals = [data["worlds"][w][metric]["mean"] for w in names]
    lo = [data["worlds"][w][metric]["mean"] - data["worlds"][w][metric]["ci95"][0] for w in names]
    hi = [data["worlds"][w][metric]["ci95"][1] - data["worlds"][w][metric]["mean"] for w in names]
    colors = [GRAY if w.startswith("oracle") else ORANGE if w.startswith("null") else
              {"independent": RED, "gaussian": "#7a5cc4", "bootstrap": BLUE, "arf": "#1f8a4c", "mixture3": "#2a9d8f",
               "mixture8": "#1f8a4c", "copula": "#8d6e63"}.get(w.split("|")[0], BLUE) for w in names]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors, text=[f"{v:.2f}" for v in vals], textposition="outside",
                           error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo)))
    if ref is not None:
        fig.add_hline(y=ref, line_dash="dash", line_color=GRAY, annotation_text=ref_label, annotation_position="top right")
    fig.update_layout(title=title, yaxis_title=ytitle, template="plotly_white", height=460,
                      margin=dict(l=60, r=30, t=60, b=150), xaxis_tickangle=-40)
    return fig


def build_html(res: dict, summ: dict, tag: str) -> str:
    model = summ["model_level"]
    names = sorted(model["worlds"], key=lambda w: (not w.startswith("oracle"), not w.startswith("null"), w))
    ex = res["explainers"]
    fig_real = go.Figure(go.Bar(x=ex, y=[model["reality_mean_error"][e] for e in ex], marker_color=BLUE,
                                error_y=dict(type="data", array=[1.96 * model["reality_error_se"][e] for e in ex]),
                                text=[f"{model['reality_mean_error'][e]:.3f}" for e in ex], textposition="outside"))
    fig_real.update_layout(title="Explainer error in reality, model level (mean over reality tasks, 95% interval)",
                           yaxis_title="relative error", template="plotly_white", height=430,
                           margin=dict(l=60, r=30, t=60, b=140), xaxis_tickangle=-35)
    figs = [fig_real,
            bar_ci(names, "pair_agreement", model, "Primary: share of reality's resolved explainer pairs each world orders correctly",
                   "pair agreement", ref=0.5, ref_label="coin flip = 0.5"),
            bar_ci(names, "regret", model, "Regret in reality of each world's chosen explainer", "relative regret",
                   ref=model["random_regret"], ref_label=f"random pick = {model['random_regret']:.2f}")]
    divs = [f.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False)) for i, f in enumerate(figs)]
    data = summ["data_level"]
    rows = "".join(
        f"<tr><td>{w}</td><td>{model['worlds'][w]['pair_agreement']['mean']:.2f}</td>"
        f"<td>{model['worlds'][w]['tau_b']['mean']:.2f}</td><td>{model['worlds'][w]['regret']['mean']:.2f}</td>"
        f"<td>{data['worlds'][w]['pair_agreement']['mean']:.2f}</td><td>{model['worlds'][w]['n_reps']} x {model['worlds'][w]['k_tasks']}</td></tr>"
        for w in names)
    table = ('<table><colgroup><col style="width:34%"><col style="width:14%"><col style="width:12%"><col style="width:12%">'
             '<col style="width:14%"><col style="width:14%"></colgroup><thead><tr><th>World</th><th>pair agreement (model)</th>'
             '<th>tau_b (model)</th><th>regret (model)</th><th>pair agreement (data)</th><th>reps x tasks</th></tr></thead>'
             f'<tbody>{rows}</tbody></table>')
    cfg = res["config"]
    design = (f"<p>Reality: {cfg['dataset']} rows with a {res['reality']} true function (signal R2 {res['reality_signal_r2']:.2f}). "
              f"Builders see {cfg['n_D']} rows; every world's ranking is the mean over {res['k']} tasks; reality's over "
              f"{res['k_reality']}. {len(ex)} explainers, scored at model level (primary) and data level. "
              f"{res['reps']} replicate samples. Run {res['created']}, {res['runtime_s']:.0f} s.</p>"
              f"<p>Reality resolves {model['n_resolved_pairs']} of {model['n_pairs']} explainer pairs at model level; "
              f"its best explainer is {'resolved' if model['best_resolved'] else 'not resolved'} "
              f"({model['reality_ranking'][0]}). Pair agreement is computed on resolved pairs only.</p>"
              + ("<p><b>Timing run (one task per world): no conclusions can be drawn from these numbers.</b></p>"
                 if res.get("timing_mode") else ""))
    secs = [("design", "1. Design", design), ("reality", "2. Reality's ranking", divs[0]),
            ("agreement", "3. Primary result: do worlds reproduce reality's resolved pairs?", divs[1] + table),
            ("regret", "4. What a wrong pick costs", divs[2])]
    est = summ.get("estimand_sensitivity")
    if est:
        rows = "".join(
            f"<tr><td>{e}</td><td>{est['mean_error_conditional'][e]:.3f}</td>"
            f"<td>{est['ranking_conditional'].index(e) + 1}</td><td>{est['mean_error_interventional'][e]:.3f}</td>"
            f"<td>{est['ranking_interventional'].index(e) + 1}</td></tr>" for e in est["ranking_conditional"])
        noise = (f" Two independent Monte Carlo estimates of the conditional truth differ by "
                 f"{est['truth_mc_noise_mean']:.3f} on average (the truth's noise floor)." if est["truth_mc_noise_mean"] else "")
        body = (f"<p>Reality's {est['n_reality_tasks']} tasks, each explainer scored against conditional and against "
                f"interventional SHAP of the same model. The two truths differ by {est['estimand_gap_mean']:.2f} "
                f"(relative error).{noise} Kendall tau between the two rankings: {est['kendall_tau_between_rankings']:+.2f}; "
                f"of the {est['pairs_resolved_under_both']} explainer pairs resolved under both, "
                f"{est['pairs_flipped']} flip order.</p>"
                '<table><colgroup><col style="width:40%"><col style="width:15%"><col style="width:15%"><col style="width:15%">'
                '<col style="width:15%"></colgroup><thead><tr><th>explainer</th><th>error vs conditional</th><th>rank</th>'
                f'<th>error vs interventional</th><th>rank</th></tr></thead><tbody>{rows}</tbody></table>')
        secs.append(("estimand", "5. Estimand sensitivity: the same explainers against two ground truths", body))
    toc = "".join(f'<li><a href="#{s}">{t}</a></li>' for s, t, _ in secs)
    css = ("body{font:16px/1.55 -apple-system,'Segoe UI',Roboto,Arial,sans-serif;max-width:1000px;margin:0 auto;padding:0 16px 60px;"
           "background:#fbfaf7;color:#1d1f24}details{background:#fff;border:1px solid #dcdfe5;border-radius:10px;margin:16px 0;"
           "padding:4px 16px 12px}summary h2{display:inline;font-size:1.2rem}table{width:100%;border-collapse:collapse;"
           "table-layout:fixed;font-size:.9rem}th,td{border-bottom:1px solid #dcdfe5;padding:5px 6px;text-align:left;"
           "overflow-wrap:anywhere}nav{background:#fff;border:1px solid #dcdfe5;border-radius:10px;padding:8px 16px}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, "
            f"initial-scale=1'><title>Transfer pilot v2 {tag}</title><style>{css}</style></head><body>"
            f"<h1>Do data-fitted benchmarks reproduce explainer rankings? "
            f"{'Pilot v3, conditional estimand' if res.get('estimand') == 'conditional' else 'Pilot v2'}: {tag}</h1>"
            f"<nav><b>Contents</b><ol>{toc}</ol></nav>"
            + "".join(f'<details open id="{s}"><summary><h2>{t}</h2></summary>{b}</details>' for s, t, b in secs)
            + "</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    setup_logging()
    start = time.time()
    res = json.loads((OUT_DIR / f"{args.tag}.json").read_text(encoding="utf-8"))
    summ = {"source": f"{args.tag}.json", "model_level": summarize(res, "err_model"),
            "data_level": summarize(res, "err_data"), "seconds_per_task": timing(res)}
    est = estimand_sensitivity(res)
    if est:
        summ["estimand_sensitivity"] = est
    (OUT_DIR / f"{args.tag}_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    (OUT_DIR / f"{args.tag}_report.html").write_text(build_html(res, summ, args.tag), encoding="utf-8")
    m = summ["model_level"]
    logging.info(f"reality resolves {m['n_resolved_pairs']}/{m['n_pairs']} pairs; ranking {m['reality_ranking']}")
    for w, s in sorted(m["worlds"].items()):
        logging.info(f"{w:32s} pairs={s['pair_agreement']['mean']:.2f} {s['pair_agreement']['ci95']} tau_b={s['tau_b']['mean']:+.2f}")
    logging.info(f"Wrote {args.tag}_summary.json and _report.html in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
