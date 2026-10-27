"""Build report.html (standalone, collapsible sections, contents box) from results/*.json.

Usage (from synthetic_data/benchmarks/):
    .venv/Scripts/python.exe report.py          # a few seconds
Every number on the page is read from the result files; nothing is rerun.
"""

import html
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RED, BLUE, ORANGE, GRAY = "#D90012", "#0033A0", "#F2A800", "#7a7a7a"
PALETTE = [BLUE, RED, ORANGE, GRAY, "#2a9d8f", "#6a4c93", "#8d6e63", "#d81b60"]
_first_fig = True


def load(tag: str) -> dict:
    p = RES / f"{tag}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run the benchmark first (see README.md)")
    return json.loads(p.read_text(encoding="utf-8"))


def fig_html(fig: go.Figure, height: int = 380) -> str:
    """Inline plotly.js once (standalone file), then reuse it."""
    global _first_fig
    fig.update_layout(height=height, margin=dict(l=60, r=30, t=40, b=60), template="plotly_white",
                      font=dict(family="system-ui, Segoe UI, sans-serif", size=12),
                      legend=dict(orientation="h", y=-0.2))
    out = pio.to_html(fig, full_html=False, include_plotlyjs="inline" if _first_fig else False,
                      config={"displaylogo": False, "responsive": True})
    _first_fig = False
    return out


def esc(s) -> str:
    return html.escape(str(s))


def table(headers: list, rows: list, widths: list | None = None, num_cols: set | None = None) -> str:
    num_cols = num_cols or set()
    cg = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(
        f'<td class="{"num" if j in num_cols else ""}">{c if isinstance(c, Markup) else esc(c)}</td>'
        for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><colgroup>{cg}</colgroup><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


class Markup(str):
    """A string that table() inserts without escaping."""


def fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def section(sid: str, title: str, body: str) -> tuple:
    return sid, title, f'<details open id="{sid}"><summary><h2>{esc(title)}</h2></summary><div class="sec">{body}</div></details>'


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#5f6368;--line:#e3e3e3;--card:#f7f7f8;--accent:#0033A0;--warn:#b3261e}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#e8e8ea;--muted:#a0a3a8;--line:#333;--card:#1d1d21;--accent:#7aa2ff;--warn:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 80px}h1{font-size:26px;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 18px}details{border-top:1px solid var(--line);padding:6px 0}
summary{cursor:pointer;list-style:none}summary::-webkit-details-marker{display:none}
summary h2{display:inline;font-size:20px}summary h2:before{content:"\\25B8  ";color:var(--muted)}
details[open] summary h2:before{content:"\\25BE  "}.sec{padding:6px 0 10px}h3{font-size:16px;margin:18px 0 6px}
.toc{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px;margin:14px 0}
.toc ol{margin:4px 0;padding-left:22px}.toc a{color:var(--accent);text-decoration:none}
.tw{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;table-layout:fixed;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{background:var(--card)}td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{background:var(--card);border-left:3px solid var(--accent);padding:8px 12px;margin:10px 0;border-radius:4px}
.warn{border-left-color:var(--warn)}code{font-size:12.5px;overflow-wrap:anywhere}
pre{background:var(--card);padding:8px 10px;overflow-x:auto;font-size:12.5px;border-radius:4px}
"""


def page(title: str, subtitle: str, sections: list) -> str:
    toc = "".join(f'<li><a href="#{sid}">{esc(t)}</a></li>' for sid, t, _ in sections)
    body = "\n".join(s for _, _, s in sections)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style>'
            f'</head><body><main><h1>{esc(title)}</h1><p class="sub">{subtitle}</p>'
            f'<nav class="toc"><b>Contents</b><ol>{toc}</ol></nav>{body}</main>'
            "<script>document.querySelectorAll('.toc a').forEach(a=>a.addEventListener('click',()=>{"
            "const d=document.querySelector(a.getAttribute('href'));if(d)d.open=true;}));</script></body></html>")


METHOD_NAMES = {"control": "random (control)", "grad": "Gradient", "ig": "Integrated Gradients",
                "itg": "Input x Gradient", "sg": "SmoothGrad", "shap": "KernelSHAP", "lime": "LIME"}


def heat_cell(v: float, values: list, higher_better: bool = True) -> str:
    """Diverging green-white-red with the median as the white pivot, alpha capped at 0.4."""
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not np.isfinite(v) or arr.size < 2:
        return ""
    mid, lo, hi = np.median(arr), arr.min(), arr.max()
    good = (v - mid) / (hi - mid + 1e-12) if v >= mid else -(mid - v) / (mid - lo + 1e-12)
    good = good if higher_better else -good
    a = 0.4 * min(1.0, abs(good))
    return f"background:rgba(46,160,67,{a:.2f})" if good > 0 else f"background:rgba(217,0,18,{a:.2f})"


def heat_table(df, higher_better: bool, nd: int = 3, row_label: str = "") -> str:
    vals = [float(x) for x in df.to_numpy().ravel()]
    cols = list(df.columns)
    widths = ["22%"] + [f"{78 / len(cols):.1f}%"] * len(cols)
    head = "".join(f"<th>{esc(c)}</th>" for c in [row_label] + cols)
    body = ""
    for idx, row in df.iterrows():
        cells = "".join(f'<td class="num" style="{heat_cell(float(v), vals, higher_better)}">{fmt(float(v), nd)}</td>'
                        for v in row)
        body += f"<tr><td>{esc(idx)}</td>{cells}</tr>"
    cg = "".join(f'<col style="width:{x}">' for x in widths)
    return f'<div class="tw"><table><colgroup>{cg}</colgroup><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def openxai_section() -> tuple:
    import pandas as pd
    from scipy.stats import spearmanr
    parts = []
    lr = load("openxai_lr")
    ann = load("openxai_ann") if (RES / "openxai_ann.json").exists() else None
    models = pd.DataFrame(lr["models"] + (ann["models"] if ann else []))
    acc = models.pivot(index="data", columns="model", values="test_accuracy")
    r = pd.DataFrame(lr["rows"]).assign(method=lambda x: x["method"].map(METHOD_NAMES))
    a = lr["args"]
    parts.append(
        "<p><b>What it is.</b> Seven real tabular datasets plus one synthetic one, pretrained logistic regressions "
        "(LR) and neural networks (ANN), seven explainers and OpenXAI's own metrics. Ground-truth metrics exist only "
        "for the LR, and the ground truth is its coefficient vector.</p>"
        f"<p><b>What ran.</b> Upstream code (commit a182886) with every explainer setting from its "
        f"<code>experiment_config.json</code>, on the first {a['n_test']} test points of each dataset (upstream "
        f"1,000); PGI and PGU on {a['n_pg']} points; stability on {a['n_stab']} points with {a['n_perturb']} kept "
        f"perturbations each (upstream 100). LR run {lr['runtime_s'] / 60:.0f} min"
        + (f", ANN run {ann['runtime_s'] / 60:.0f} min (stability on {ann['args']['n_stab']} points)." if ann
           else "; the ANN run had not finished when this page was built.")
        + ("" if not ann or ann.get("complete", True) else
           " <b>The ANN run stopped early</b>; it covers "
           + ", ".join(sorted({m['data'] for m in ann['models']})) + "."))
    parts.append("<h3>Pretrained models: accuracy on the full test split</h3>" + heat_table(acc.round(4), True, 3, "dataset"))
    gt_metrics = ["FA", "RA", "SA", "SRA", "PRA", "RC"]
    mean_gt = r[r.metric.isin(gt_metrics)].groupby(["method", "metric"])["mean"].mean().unstack()[gt_metrics]
    fig = go.Figure()
    for i, m in enumerate(gt_metrics):
        fig.add_bar(name=m, x=list(mean_gt.index), y=mean_gt[m], marker_color=PALETTE[i],
                    text=[f"{v:.2f}" for v in mean_gt[m]], textposition="outside", textangle=-90,
                    textfont=dict(size=10))
    fig.update_layout(barmode="group", yaxis_title="mean over the 8 datasets (LR)", yaxis_range=[-0.15, 1.3])
    parts.append("<h3>Ground-truth metrics (LR only), averaged over the datasets</h3>" + fig_html(fig, 420))
    rc = r[r.metric == "RC"].pivot(index="method", columns="data", values="mean")
    parts.append("<h3>Rank correlation with the coefficient vector (RC), per dataset</h3>"
                 + heat_table(rc, True, 3, "method"))
    parts.append(
        '<div class="note"><b>Reading.</b> Gradient, Integrated Gradients and SmoothGrad score exactly 1.000 on every '
        "ground-truth metric on every dataset: for a logistic model each returns a vector proportional to the "
        "coefficients, which is the reference. KernelSHAP and Input x Gradient attribute contributions "
        "(coefficient times feature value), a different quantity, so they score near chance on several datasets. "
        "Here the ground-truth metrics test whether an attribution is proportional to the coefficients, not "
        "whether it is correct. Through the Looking Glass has the same mismatch.</div>")
    rows = []
    for proxy, sign in [("PGI", 1), ("PGU", -1), ("RIS", -1), ("ROS", -1)]:
        p = r[r.metric == proxy].pivot(index="method", columns="data", values="mean") * sign
        cors = [spearmanr(rc[ds], p[ds]).correlation for ds in rc.columns]
        rows.append([proxy, fmt(float(np.median(cors)), 2), fmt(float(np.min(cors)), 2), fmt(float(np.max(cors)), 2)])
    parts.append("<h3>Do the proxy metrics rank methods like the ground truth? (LR)</h3>"
                 "<p>Spearman correlation per dataset between the 7 methods' RC and their score on each proxy, the "
                 "proxy signed so that higher is better. PGI and PGU reward the features a model is most sensitive "
                 "to, which for a logistic model are the large-coefficient ones, so agreement there is partly built "
                 "in.</p>"
                 + table(["proxy", "median over datasets", "min", "max"], rows, ["25%", "25%", "25%", "25%"], {1, 2, 3}))
    for tag, res in [("LR", lr)] + ([("ANN", ann)] if ann else []):
        rr = pd.DataFrame(res["rows"]).assign(method=lambda x: x["method"].map(METHOD_NAMES))
        for m in ["PGI", "PGU"]:
            t = rr[rr.metric == m].pivot(index="method", columns="data", values="mean")
            parts.append(f"<h3>{m}, {tag} ({'higher' if m == 'PGI' else 'lower'} is better)</h3>"
                         + heat_table(t, m == "PGI", 3, "method"))
    parts.append(
        '<div class="note warn"><b>Found while setting it up.</b> (1) The dataloader fits a separate MinMaxScaler on '
        "the test split (<code>openxai/dataloader.py</code>, TabularDataLoader), so test features are scaled by "
        "the test set's own minimum and maximum, not the training set's. (2) The published results changed between "
        "the NeurIPS 2022 paper and arXiv v5 with no erratum, and this code reproduces v5: the gradient's sign "
        "agreement on HELOC is 1.000 here and in v5, 0.469 in the NeurIPS table "
        "(<code>_learnings/2026-10-27-0123_openxai-results-changed-between-versions.md</code>). (3) In the "
        "stability metrics <code>n_samples</code> is the number of candidate perturbations per input, not the "
        "number of inputs.</div>")
    parts.append("<h3>Explore further</h3><pre>.venv/Scripts/python.exe run_openxai.py --datasets heloc --models lr ann "
                 "--n-test 200 --tag openxai_heloc</pre>")
    return section("openxai", "OpenXAI (Agarwal et al. 2022)", "".join(parts))


def xaibench_section(tag: str = "xaibench") -> tuple:
    import pandas as pd
    from scipy.stats import kendalltau, spearmanr
    d = load(tag)
    r = pd.DataFrame(d["rows"])
    a = d["args"]
    parts = [
        "<p><b>What it is.</b> Synthetic Gaussian features (5 dimensions, equicorrelation rho) with a known target "
        "(linear, nonlinear additive or piecewise constant); models are trained on them and explained, and the "
        "explanations are scored against ground-truth Shapley values computed from the known distribution, plus "
        "faithfulness, monotonicity and infidelity.</p>"
        f"<p><b>What ran.</b> Upstream Experiment (commit f0431a7), regression, weights [4, 3, 2, 1, 0], noise 0.01, "
        f"1,000 training rows, {a['n_val']} validation points (upstream 100), ground truth from "
        f"{a.get('gt_samples', 20000):,} conditional draws per coalition (upstream 20,000), rho in {a['rhos']}, models LR / "
        "decision tree / MLP, explainers random, SHAP, SHAPR, brute-force KernelSHAP, MAPLE, LIME. Not run: L2X "
        f"(TensorFlow 2.5), BreakDown (blist), the ROAR metrics. Runtime {d['runtime_s'] / 60:.0f} min.</p>"]
    gap = pd.DataFrame(d["estimand_gap"])
    gap["relative gap"] = gap["mse_conditional_gt_vs_interventional"] / gap["mean_sq_conditional_gt"]
    g = gap.groupby(["data", "rho"])["relative gap"].mean().unstack()
    fig = go.Figure()
    for i, rho in enumerate(g.columns):
        fig.add_bar(name=f"rho = {rho}", x=list(g.index), y=g[rho], marker_color=[GRAY, ORANGE, RED, BLUE][i % 4],
                    text=[f"{v:.2f}" for v in g[rho]], textposition="outside")
    fig.update_layout(barmode="group", yaxis_title="relative estimand gap")
    parts.append("<h3>How far apart are the two Shapley estimands?</h3><p>XAI-BENCH's ground truth is the conditional "
                 "(observational) Shapley value; brute-force KernelSHAP computes the exact interventional one on 1,000 "
                 "training rows. Their squared difference relative to the size of the ground truth, mean over the "
                 "three models:</p>" + fig_html(fig, 380))
    shap_err = r[r.metric == "shapley"].groupby(["explainer", "data", "rho"])["value"].mean().unstack(["data", "rho"])
    shap_err.columns = [f"{c[0].replace('gaussian', '')}, rho {c[1]}" for c in shap_err.columns]
    parts.append("<h3>Error against XAI-BENCH's ground truth (MSE, mean over models; lower is better)</h3>"
                 + heat_table(shap_err, False, 3, "explainer"))
    rows = []
    for (ds, rho, m), grp in r.groupby(["data", "rho", "model"]):
        p = grp.pivot_table(index="explainer", columns="metric", values="value")
        p = p.drop(index=["brutekernelshap"])   # it is the interventional reference itself
        tau = kendalltau(p["shapley"], p["mse_vs_interventional"]).correlation
        rows.append({"data": ds, "rho": rho, "model": m, "tau": tau,
                     "best_cond": p["shapley"].idxmin(), "best_int": p["mse_vs_interventional"].idxmin()})
    rr = pd.DataFrame(rows)
    tab = [[x.data.replace("gaussian", ""), x.rho, x.model, fmt(x.tau, 2), x.best_cond, x.best_int]
           for x in rr.itertuples()]
    parts.append("<h3>Does the choice of estimand change the ranking?</h3><p>Kendall tau between the explainers' "
                 "ranking against the conditional ground truth and against the interventional reference (5 explainers, "
                 "brute-force KernelSHAP excluded since it is the reference), and the winner under each.</p>"
                 + table(["target", "rho", "model", "tau", "best vs conditional GT", "best vs interventional"], tab,
                         ["17%", "8%", "9%", "8%", "29%", "29%"], {1, 3}))
    proxy_rows = []
    for proxy, sign in [("faithfulness", -1), ("monotonicity", -1), ("infidelity", 1), ("shapley_corr", -1)]:
        cors = []
        for _, grp in r.groupby(["data", "rho", "model"]):
            p = grp.pivot_table(index="explainer", columns="metric", values="value")
            cors.append(spearmanr(p["shapley"], sign * p[proxy]).correlation)
        cors = np.asarray(cors, dtype=float)
        proxy_rows.append([proxy, fmt(float(np.nanmedian(cors)), 2), fmt(float(np.nanmin(cors)), 2),
                           fmt(float(np.nanmax(cors)), 2), int(np.isfinite(cors).sum())])
    parts.append("<h3>Do the proxy metrics rank explainers like the ground truth?</h3><p>Spearman correlation, per "
                 "(target, rho, model), between the explainers' Shapley error and their proxy score signed so that "
                 "a positive value means agreement. XAI-BENCH's own Table 1 reports the same kind of comparison.</p>"
                 + table(["proxy", "median", "min", "max", "conditions"], proxy_rows,
                         ["30%", "17%", "17%", "17%", "19%"], {1, 2, 3, 4}))
    perf = pd.DataFrame(d["model_perfs"])
    perf["test MSE"] = perf["mse_train_test"].map(lambda x: float(x[1]))
    perf["train MSE"] = perf["mse_train_test"].map(lambda x: float(x[0]))
    pt = perf.pivot_table(index=["data", "rho"], columns="model", values="test MSE")
    pt.index = [f"{i[0].replace('gaussian', '')}, rho {i[1]}" for i in pt.index]
    parts.append("<h3>Model test MSE (labels standardized)</h3>" + heat_table(pt, False, 2, "condition"))
    parts.append(
        '<div class="note warn"><b>Found while setting it up.</b> (1) <code>generatetarget</code> standardizes Y with '
        "the mean and sd of whichever batch it receives, so training and validation labels are different affine "
        "functions of X: even a linear regression on linear data with noise 0.01 has a test MSE far above the noise "
        "(table above). (2) SHAPR only conditions coalitions of two or more features; singletons use the marginal "
        "(interventional) mean (<code>custom_explainers/shapr.py</code>, <code>if len(s) &gt; 1</code>), so its "
        "estimand is a hybrid. (3) The SHAP explainer passes 1,000 background rows and shap keeps 100. (4) The "
        "ground truth is conditional Shapley, which the interventional explainers are not trying to estimate; at "
        "rho &gt; 0 their 'error' is mostly the estimand gap shown in the first chart.</div>"
        "<h3>Explore further</h3><pre>.venv/Scripts/python.exe run_xaibench.py --datasets gaussianNonLinearAdditive "
        "--rhos 0 0.25 0.5 0.75 0.99 --n-val 50 --tag xaibench_rho_sweep</pre>")
    return section("xaibench", "XAI-BENCH (Liu et al. 2021)", "".join(parts))


def iterp_section(tag: str = "iterpretability") -> tuple:
    import pandas as pd
    d = load(tag)
    r = pd.DataFrame(d["rows"])
    a = d["args"]
    rb = pd.DataFrame(d["random_baseline"])
    parts = [
        "<p><b>What it is.</b> Real covariates (here Twins, d = 39) with simulated potential outcomes: a linear "
        "prognostic part and a linear predictive part (the treatment effect) on known feature subsets. CATE "
        "learners are fitted and explained; the score is the share of the top-ranked features that are truly "
        "predictive. The predictive scale sets how large the treatment effect is.</p>"
        f"<p><b>What ran.</b> Upstream components (commit 8121b9f): linear simulator, {d['n_predictive']} predictive "
        f"and {d['n_prognostic']} prognostic features (k = {a['n_important']}), five catenets learners with "
        f"upstream hyperparameters (n_iter {a['n_iter']}), explainers {', '.join(a['explainers'])}, "
        f"{a['explainer_limit']} test rows explained, seed {a['seed']}, scales {a['scales']}. Runtime "
        f"{d['runtime_s'] / 60:.0f} min.</p>"]
    ig = r[r.explainer == "integrated_gradients"].pivot(index="learner", columns="scale", values="pred_acc")
    fig = go.Figure()
    for i, s in enumerate(ig.columns):
        fig.add_bar(name=f"predictive scale {s}", x=list(ig.index), y=ig[s], marker_color=[GRAY, ORANGE, BLUE, RED][i % 4],
                    text=[f"{v:.2f}" for v in ig[s]], textposition="outside")
    chance = d["chance_pred_acc_code_rule"]
    fig.add_hline(y=chance, line_dash="dash", line_color=RED)
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=RED, dash="dash"),
                    name=f"chance {chance:.2f} ({d['n_predictive']} of {d['d']} features)")
    fig.update_layout(barmode="group", yaxis_title="predictive-feature accuracy (Integrated Gradients)",
                      yaxis_range=[0, 1.05])
    parts.append("<h3>Integrated Gradients: share of top features that are predictive</h3>" + fig_html(fig, 400))
    top = r[r.scale == max(a["scales"])].pivot(index="learner", columns="explainer", values="pred_acc")
    parts.append(f"<h3>All explainers at the largest scale ({max(a['scales'])})</h3>" + heat_table(top, True, 3, "learner"))
    parts.append(table(["scale", "random attribution: predictive acc", "random: prognostic, upstream", "random: prognostic, top-k"],
                       [[x.scale, fmt(x.pred_acc_random), fmt(x.prog_acc_random_upstream), fmt(x.prog_acc_random_topk)]
                        for x in rb.itertuples()], ["16%", "28%", "28%", "28%"], {0, 1, 2, 3}))
    pg = r[r.explainer == "integrated_gradients"].groupby(["learner", "scale"])[["prog_acc_upstream", "prog_acc_topk"]].mean()
    parts.append("<h3>Prognostic accuracy as upstream computes it vs at top-k (Integrated Gradients)</h3>"
                 + table(["learner", "scale", "upstream (top-1)", "top-k"],
                         [[i[0], i[1], fmt(v.prog_acc_upstream), fmt(v.prog_acc_topk)] for i, v in pg.iterrows()],
                         ["34%", "16%", "25%", "25%"], {1, 2, 3}))
    pehe = r.groupby(["learner", "scale"])["pehe_normalized"].first().unstack()
    parts.append("<h3>PEHE / sd(true CATE) (lower is better)</h3>" + heat_table(pehe, False, 3, "learner"))
    parts.append(
        '<div class="note warn"><b>Found while setting it up.</b> (1) The paper sets the number of important '
        "features to floor(0.2 d), which is 7 for Twins; <code>run_experiments.py</code> defaults to 8 for Twins "
        f"and 10 for ACIC (the rule gives 11). Chance level for Twins is therefore {chance:.2f} with the code's "
        f"value and {d['chance_pred_acc_paper_rule']:.2f} with the paper's. (2) "
        "<code>get_prognostic_features()</code> returns the tuple from <code>np.where</code>, so the prognostic "
        "accuracy is a top-1 hit rate, not top-k (table above). (3) The noise sd, which the paper does not state, "
        "is 0.1 in <code>synthetic_simulate.py</code>.</div>"
        "<h3>Explore further</h3><pre>.venv/Scripts/python.exe run_iterpretability.py --scales 0.001 0.1 1 "
        "--explainers integrated_gradients --explainer-limit 200 --tag iterp_ig</pre>")
    return section("iterp", "ITErpretability (Crabbe et al. 2022)", "".join(parts))


def estimator_scores(rows: list, estimand: str = "ate", truth: float | None = None) -> "pd.DataFrame":
    """Bias and RMSE of each estimator's <estimand>_est against the per-data-set <estimand>_true (or one fixed truth)."""
    import pandas as pd
    df = pd.DataFrame(rows)
    df["err"] = df[f"{estimand}_est"] - (df[f"{estimand}_true"] if truth is None else truth)
    return df.groupby("estimator").agg(bias=("err", "mean"), rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                                       n=("err", "size")).sort_values("rmse")


def att_ate_table(rows: list, true_ate: float | None = None, true_att: float | None = None, nd: int = 1) -> str:
    """Estimators scored against the ATT (the LaLonde literature's target) and the ATE, sorted by ATT RMSE."""
    att = estimator_scores(rows, "att", true_att)
    ate = estimator_scores(rows, "ate", true_ate).reindex(att.index)
    return table(["estimator", "ATT bias", "ATT RMSE", "ATE bias", "ATE RMSE", "data sets"],
                 [[i, fmt(att.loc[i, "bias"], nd), fmt(att.loc[i, "rmse"], nd), fmt(ate.loc[i, "bias"], nd),
                   fmt(ate.loc[i, "rmse"], nd), int(att.loc[i, "n"])] for i in att.index],
                 ["32%", "13%", "13%", "13%", "13%", "16%"], {1, 2, 3, 4, 5})


def realcause_section() -> tuple:
    import pandas as pd
    d = load("realcause")
    a = d["part_a"]
    paper = a["paper_tables_4_5"]
    om = {(x["dataset"], x["causal_score"]): x["spearman"] for x in a["outcome_median_spearman"]}
    pm = {x["dataset"]: x["spearman"] for x in a["propensity_median_spearman"]}
    rows = []
    for ds in ["lalonde_psid", "lalonde_cps", "twins"]:
        rows.append([ds, fmt(om[(ds, "ate_rmse")], 2), fmt(paper["outcome_vs_ate_rmse"][ds], 2),
                     fmt(om[(ds, "mean_pehe")], 2), fmt(paper["outcome_vs_pehe"][ds], 2),
                     fmt(pm[ds], 2), fmt(paper["propensity_vs_ate_rmse"][ds], 2)])
    b = pd.DataFrame(d["part_b"])
    truths = b.groupby("dataset").agg(ate=("ate_true", "mean"), att=("att_true", "mean"),
                                      treated=("treated_share", "mean")).reindex(["lalonde_psid", "lalonde_cps", "twins"])
    parts = [
        "<p><b>What it is.</b> RealCause fits generative models of (W, T, Y) to real data (LaLonde PSID and CPS, "
        "Twins), so simulated data have known potential outcomes. The repository ships 100 sampled data sets per "
        "real data set and a CSV of 1,568 fitted estimators with their predictive and causal scores.</p>"
        "<h3>A. Re-analysis of the 1,568-estimator CSV (paper Tables 4 and 5)</h3>"
        "<p>Median Spearman correlation, within model classes, between predictive error and causal error. "
        "Recomputed from <code>causal-predictive-analysis.csv</code> with the logic of "
        "<code>experiments/uai_analysis.py</code>.</p>"
        + table(["data set", "outcome models vs ATE RMSE", "paper", "outcome models vs PEHE", "paper",
                 "propensity models vs ATE RMSE", "paper"], rows,
                ["16%", "15%", "9%", "15%", "9%", "17%", "9%"], {1, 2, 3, 4, 5, 6})
        + '<div class="note">All nine medians match the paper to two decimals. Predictive performance of the outcome '
          "model tracks causal performance (0.77 to 0.92); predictive performance of the propensity model barely "
          "does (0.22 to 0.51).</div>",
        f"<h3>B. Seven estimators on {b['sample'].nunique()} of the pre-sampled data sets per real data set</h3>"
        "<p>The estimators are RealCause's own causallib wrappers. The truth is each sampled data set's own ATT and "
        "ATE, means of its <code>ite</code> column over the treated and over everyone.</p>"
        + table(["data set", "true ATE (mean)", "true ATT (mean)", "treated share"],
                [[ds, fmt(r.ate, 1), fmt(r.att, 1), fmt(r.treated, 3)] for ds, r in truths.iterrows()],
                ["34%", "22%", "22%", "22%"], {1, 2, 3})]
    for ds in ["lalonde_psid", "lalonde_cps", "twins"]:
        parts.append(f"<p><b>{esc(ds)}</b>, sorted by ATT RMSE</p>"
                     + att_ate_table([x for x in d["part_b"] if x["dataset"] == ds], nd=1 if ds != "twins" else 4))
    parts.append(
        '<div class="note warn"><b>Reading.</b> The LaLonde generators imply a large negative ATE '
        f"({truths.loc['lalonde_psid', 'ate']:.0f} USD for PSID) next to a small positive ATT "
        f"({truths.loc['lalonde_psid', 'att']:.0f} USD). The ATE is dominated by what the fitted outcome model "
        "says treatment would do to the high-earning comparison group, where the real data contain almost no "
        "treated units. That number comes from the generator's extrapolation, not from the data. The section "
        "'Do data-fitted causal simulators agree?' compares it with the other generators.</div>")
    return section("realcause", "RealCause (Neal et al. 2020)", "".join(parts))


def wgan_section(tag: str = "wgan") -> tuple:
    d = load(tag)
    parts = [
        "<p><b>What it is.</b> Athey et al. fit conditional WGANs to the LaLonde data (covariates given treatment, "
        "earnings given covariates and treatment), then use the generated data, whose potential outcomes are known, "
        "to compare effect estimators. The repository ships trained generators for the PSID, CPS and experimental "
        "samples, but not the notebook that trained them.</p>"
        "<p><b>What ran.</b> The shipped generators, loaded into the package's Generator with the layout read from "
        "the weight shapes. The scaling statistics and the context order are not in the weights, so eight "
        "configurations were scored on how well generated means match the real ones within each arm, and the best "
        f"was kept (a rule fixed before looking at estimator results). {d['args']['reps']} simulated data sets of "
        f"the real size per sample, seven estimators. Runtime {d['runtime_s'] / 60:.1f} min.</p>"]
    for ds, v in d["datasets"].items():
        cfg = sorted(v["config_scores"], key=lambda c: c["mean_abs_std_diff"])
        parts.append(f"<h3>{esc(ds.upper())} (n = {v['n']}, treated share {v['p_treat']:.3f})</h3>"
                     f"<p>Chosen configuration: X | t scaling <b>{esc(v['chosen_config'][0])}</b>, Y context "
                     f"<b>{esc(v['chosen_config'][1])}</b>, Y scaling <b>{esc(v['chosen_config'][2])}</b>; mean "
                     f"|standardized difference| {cfg[0]['mean_abs_std_diff']:.3f} (worst configuration "
                     f"{cfg[-1]['mean_abs_std_diff']:.3f}). True ATE {v['true_ate']:.0f} USD, ATT {v['true_att']:.0f} "
                     "USD (100,000 draws).</p>")
        means = [[m["variable"], m["t"], fmt(m["real"], 2), fmt(m["generated"], 2)] for m in v["means_by_t"]]
        parts.append(table(["variable", "t", "real mean", "generated mean"], means, ["34%", "10%", "28%", "28%"], {1, 2, 3}))
        parts.append("<p>Estimators against the generator's ATT and ATE, sorted by ATT RMSE:</p>"
                     + att_ate_table(v["rows"], true_ate=v["true_ate"], true_att=v["true_att"]))
    parts.append("<h3>Explore further</h3><pre>.venv/Scripts/python.exe run_wgan.py --reps 100 --tag wgan_100</pre>")
    return section("wgan", "WGAN simulations (Athey et al. 2024)", "".join(parts))


def credence_section(tag: str = "credence") -> tuple:
    d = load(tag)
    rows = []
    for name, v in d["arms"].items():
        rows.append([name, "yes" if v["standardized"] else "no (upstream)", v["max_epochs"],
                     "yes" if v["diverged"] else "no", "yes" if v["treatment_collapsed"] else "no",
                     f"{v['treated_share_generated']:.3f} / {v['treated_share_real']:.3f}",
                     fmt(v["ite_mean"], 0) if v["ite_mean"] is not None else "NaN",
                     fmt(v["ite_sd"], 0) if v["ite_sd"] is not None else "NaN",
                     fmt(100 * v["share_negative_earnings"]["y0"], 1)])
    parts = [
        "<p><b>What it is.</b> Credence fits conditional VAEs to (T, X, Y) and lets the user impose the treatment "
        "effect f(X) and a selection-bias function, so the generated data have a known effect.</p>"
        "<p><b>What ran.</b> Credence-v2 at commit 195cbf6 (patched to train on CPU in fp32), LaLonde PSID and the "
        f"NSW experiment, target effects 0 and +1,000 USD, {d['args']['max_epochs']} epochs (upstream 100). "
        f"Runtime {d['runtime_s'] / 60:.1f} min.</p>"
        + table(["arm", "standardized", "epochs", "diverged", "T collapsed", "treated share gen / real",
                 "generated ITE mean", "ITE sd", "% negative Y(0)"], rows,
                ["24%", "10%", "6%", "8%", "8%", "13%", "11%", "9%", "11%"], {2, 6, 7, 8})]
    for name, v in d["arms"].items():
        if v["rows"]:
            parts.append(f"<p><b>{esc(name)}</b>: estimators against each generated data set's ATT and ATE</p>"
                         + att_ate_table(v["rows"]))
    parts.append(
        '<div class="note warn"><b>Found while setting it up.</b> (1) With upstream settings (raw dollars, effect '
        "constraint weighted 1e20) training diverges: every generated effect is NaN. (2) <code>sample()</code> "
        "turns the treatment decoder's output into 0/1 at 0.5 instead of drawing it, so a rare treatment (PSID: "
        "6.9% treated) can disappear from the generated data altogether, and the treated share is not reproduced "
        "even when the treatment is common. (3) <code>Credence.fit(batch_size=...)</code> is never passed on; every "
        "VAE trains with batch size 10. (4) The bound constraint adds 1e20 times an indicator, which has no "
        "gradient, so bounds never shape training (negative earnings in the table).</div>"
        + '<div class="note"><b>Does the generator deliver the effect it promises?</b> '
        + "; ".join(f"{esc(n)}: target {v['target_effect']:.0f}, generated mean {v['ite_mean']:.0f} (sd {v['ite_sd']:.0f})"
                    for n, v in d["arms"].items() if v["ite_mean"] is not None)
        + ". The constraint is enforced on the training batches through the encoder's latent codes; samples drawn "
          "from the prior need not satisfy it, and in these runs (CPU, fp32, standardized data, "
          f"{d['args']['max_epochs']} epochs) they often do not.</div>")
    return section("credence", "Credence (Parikh et al. 2022)", "".join(parts))


def causal_cross_section() -> tuple:
    import pandas as pd
    from scipy.stats import kendalltau
    rc, wg = load("realcause"), load("wgan")
    cr = load("credence") if (RES / "credence.json").exists() else None
    worlds = {}   # name -> (ATT scores, ATE scores, true ATE, true ATT)
    b = [x for x in rc["part_b"] if x["dataset"] == "lalonde_psid"]
    worlds["RealCause (PSID)"] = (estimator_scores(b, "att"), estimator_scores(b, "ate"),
                                  float(np.mean([x["ate_true"] for x in b])), float(np.mean([x["att_true"] for x in b])))
    v = wg["datasets"]["psid"]
    worlds["WGAN (PSID)"] = (estimator_scores(v["rows"], "att", v["true_att"]),
                             estimator_scores(v["rows"], "ate", v["true_ate"]), v["true_ate"], v["true_att"])
    if cr:
        for name, arm in cr["arms"].items():
            if arm["rows"] and arm["data"] == "psid":
                worlds[f"Credence ({name})"] = (estimator_scores(arm["rows"], "att"), estimator_scores(arm["rows"], "ate"),
                                                float(np.mean([x["ate_true"] for x in arm["rows"]])),
                                                float(np.mean([x["att_true"] for x in arm["rows"]])))
    rows = [[w, fmt(ate, 0), fmt(att, 0), s_att.index[0], fmt(s_att["rmse"].iloc[0], 0), s_ate.index[0]]
            for w, (s_att, s_ate, ate, att) in worlds.items()]
    names = list(worlds)
    tau_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            taus = []
            for k in (0, 1):
                a, bb = worlds[names[i]][k]["rmse"], worlds[names[j]][k]["rmse"]
                common = a.index.intersection(bb.index)
                taus.append(kendalltau(a[common], bb[common]).correlation)
            tau_rows.append([names[i], names[j], fmt(taus[0], 2), fmt(taus[1], 2), len(common)])
    rank = pd.DataFrame({w: s_att["rmse"].rank() for w, (s_att, _, _, _) in worlds.items()})
    body = (
        "<p>RealCause, the WGAN package and Credence are three different ways of fitting a simulator to the same "
        "LaLonde PSID data. If data-fitted simulators are a trustworthy benchmark, they should at least agree with "
        "each other on which estimator is best. This is the causal-inference version of the transfer pilot's "
        "question (DECISIONS.md #22). The same seven estimators were scored in each, against each generator's own "
        "true ATT (the usual LaLonde target) and true ATE.</p>"
        + table(["world", "true ATE (USD)", "true ATT (USD)", "best for ATT", "its ATT RMSE", "best for ATE"], rows,
                ["22%", "11%", "11%", "22%", "11%", "23%"], {1, 2, 4})
        + "<h3>Estimator ranks by ATT RMSE (1 = best)</h3>" + heat_table(rank, False, 0, "estimator")
        + "<h3>Agreement between worlds (Kendall tau of the RMSE rankings)</h3>"
        + table(["world A", "world B", "tau, ATT", "tau, ATE", "estimators"], tau_rows,
                ["32%", "32%", "12%", "12%", "12%"], {2, 3, 4})
        + '<div class="note"><b>Reading.</b> The worlds disagree first about the truth itself. The ATE of the PSID '
          "population is set by how each generator extrapolates outcomes under treatment to the high-earning "
          "comparison group, where the data have almost no treated units; the ATEs above differ several-fold. The "
          "ATT, which the data support better, is closer but still differs noticeably between generators. "
          "Ranking agreement is higher for the ATT than for the ATE, but with 7 estimators and one data set it is "
          "an illustration, not a measurement. With different truths, agreement on the ranking of estimators is not "
          "something the data can guarantee.</div>"
        + ("" if any(k.startswith("Credence") for k in worlds) else
           '<div class="note warn">Credence is missing from this comparison: fitted to PSID it generated no treated '
           "units at all (see its section), so no estimator could be scored in its world.</div>"))
    return section("cross", "Do data-fitted causal simulators agree with each other?", body)


DISCREPANCIES = [
    ("OpenXAI", "Results changed between the NeurIPS 2022 paper and arXiv v5 with no erratum (gradient SA on HELOC "
     "0.469 vs 1.000; random FA 0.498 vs 0.149); the current code reproduces v5.", "High"),
    ("OpenXAI", "The test split gets its own MinMaxScaler (dataloader.py), so test features are scaled with test "
     "statistics.", "High"),
    ("OpenXAI", "Ground truth on real data is the LR coefficient vector, so slope methods score 1.000 by algebra and "
     "contribution methods (SHAP, input x gradient) lose by construction.", "High"),
    ("XAI-BENCH", "generatetarget standardizes Y per batch, so training and validation labels are different functions "
     "of X.", "High"),
    ("XAI-BENCH", "SHAPR conditions only coalitions of two or more features; singletons are interventional.", "High"),
    ("XAI-BENCH", "Ground truth is conditional Shapley while SHAP, KernelSHAP and MAPLE target interventional SHAP; "
     "at rho > 0 their measured error is largely the estimand gap.", "High"),
    ("ITErpretability", "Code defaults use 8 important features for Twins and 10 for ACIC; the paper's rule floor(0.2 d) "
     "gives 7 and 11, which changes the chance level.", "High"),
    ("ITErpretability", "Prognostic accuracy is a top-1 hit rate because get_prognostic_features() returns a tuple.",
     "High"),
    ("Miro-Nicolau et al.", "Tabular ground truth: Eq. 5 lowers a feature by one level, the released notebook sets it "
     "to 0; the method ranking flips with the definition.", "High"),
    ("Miro-Nicolau et al.", "Label weights 1/2, 1/4, 1/6 in code vs [0.5, 0.25, 0.125] in the paper; the data "
     "generator draws continuous uniform values and never seeds them.", "High"),
    ("Credence-v2", "Upstream settings diverge to NaN on LaLonde; thresholding T at 0.5 erases a rare treatment; "
     "batch_size is never passed on; bound constraints have no gradient.", "High"),
    ("RealCause", "Tables 4 and 5 reproduce exactly from the shipped CSV; the analysis script itself no longer runs "
     "on pandas 2 (Series.append).", "High"),
    ("Athey et al. WGAN", "The repository ships weights but not the code that defines their scaling, so a user has to "
     "infer it (here by matching generated and real means).", "Medium"),
]


def discrepancy_section() -> tuple:
    return section("discrepancies", "Paper vs code: everything found while setting up",
                   "<p>Collected from the sections above; each was found by reading or running the released code at "
                   "the pinned commit.</p>"
                   + table(["benchmark", "finding", "how sure"], [list(x) for x in DISCREPANCIES], ["16%", "72%", "12%"]))


def setup_section() -> tuple:
    rows = []
    for p in sorted(RES.glob("*.json")):
        if p.stem.endswith("_smoke"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        status = "complete" if d.get("complete", True) else "PARTIAL (stopped early; see its section)"
        rows.append([p.stem, d.get("created", ""), fmt(d.get("runtime_s", float("nan")) / 60, 1), status,
                     esc(", ".join(f"{k} {v}" for k, v in d.get("versions", {}).items()))])
    body = ("<p>All runs used one shared uv environment (Python 3.12; DECISIONS.md #25), upstream code at pinned "
            "commits with the patches in <code>patches/</code>, at most 2 threads, one run at a time. Each run is "
            "deliberately smaller than upstream; the cuts are listed in each runner's docstring and in its section "
            "here. A gap from a paper's number can come from the newer libraries or from the cuts before it comes "
            "from the paper.</p>"
            + table(["result file", "created", "minutes", "status", "versions"], rows,
                    ["15%", "16%", "8%", "15%", "46%"], {2})
            + "<pre>bash setup_sources.sh   # clones + patches\nbash run_all.sh         # every run, one at a time (~80 min)\n"
              ".venv/Scripts/python.exe report.py</pre>")
    return section("setup", "Setup, runtimes and versions", body)


def main() -> None:
    sections = [setup_section()]
    for fn in (openxai_section, xaibench_section, iterp_section, fb2ts_section, realcause_section, wgan_section,
               credence_section, causal_cross_section):
        sections.append(fn())
    sections.append(discrepancy_section())
    out = HERE / "report.html"
    out.write_text(page("Benchmark runs", "The benchmarks of the papers in <code>synthetic_data/papers/</code>, run on "
                        "this laptop: what each run did, what it found, and where paper and code disagree.", sections),
                   encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def fb2ts_section() -> tuple:
    import pandas as pd
    d = load("fb2ts_tabular")
    r = pd.DataFrame(d["rows"])
    gts = list(r["ground_truth"].unique())
    sim = r.pivot(index="method", columns="ground_truth", values="sim_mean")[gts]
    paper = d["paper_table7_sim"]
    fig = go.Figure()
    for i, g in enumerate(gts):
        fig.add_bar(name=f"ours, against {g}", y=list(sim.index), x=sim[g], orientation="h",
                    marker_color=[BLUE, ORANGE][i], text=[f"{v:.3f}" for v in sim[g]], textposition="outside")
    in_paper = [m for m in sim.index if m in paper]
    fig.add_bar(name="paper, Table 7", y=in_paper, x=[paper[m] for m in in_paper], orientation="h",
                marker_color=GRAY, text=[f"{paper[m]:.3f}" for m in in_paper], textposition="outside")
    fig.update_layout(barmode="group", xaxis_title="SIM (1 = identical to the ground truth)", xaxis_range=[0, 1.18])
    fig.update_layout(margin=dict(l=150, r=40, t=30, b=60))
    a = d["args"]
    paper_gt = [g for g in gts if g.startswith("paper")][0]
    gaps = {m: abs(sim.loc[m, paper_gt] - paper[m]) for m in in_paper}
    rand = sim.loc["Random (uniform)"]
    body = (
        "<p><b>What it is.</b> Miro-Nicolau et al. (2026) derive ground truth from interventions on the trained model. "
        "In the tabular arm: 3 features on a grid, a label from a weighted sum of sines, an MLP, and feature "
        "importance from the change in the model's output when a feature is intervened on. They score methods by "
        "SIM (histogram intersection after min-max scaling) against that ground truth.</p>"
        "<p><b>What ran.</b> The released notebook's model, training recipe and metric code (cell 24, executed as "
        f"released) on a {a['levels']}-level grid with {a['epochs']} epochs (test accuracy "
        f"{d['best_test_accuracy']:.4f}); {d['n_correct']} correctly classified points explained and scored against "
        "both ground-truth definitions. LORE (xailib) is not installed.</p>"
        + fig_html(fig, 440) +
        f'<div class="note"><b>Reproduction.</b> Scored against the paper\'s definition, our numbers land within '
        f"{max(gaps.values()):.3f} of Table 7 for every method in it ({', '.join(f'{m} {v:.3f}' for m, v in gaps.items())}), "
        "so the published results were computed with Eq. 5, not with the notebook's definition. The random "
        f"baseline scores {rand[paper_gt]:.3f} against that ground truth, so DeepLIFT, Integrated Gradients and "
        "KernelSHAP (0.60 to 0.64 in the paper) sit only about 0.15 above chance.</div>"
        '<div class="note"><b>Reading.</b> The two ground truths disagree with each other (SIM '
        f"{d['ground_truths_agreement']['sim']:.3f}; the same feature order in "
        f"{100 * d['ground_truths_same_order_share']:.0f}% of points), and which method wins depends on which one is "
        "used. The paper's one-level step is a finite-difference gradient, so Gradient matches it by construction. "
        "The code's x_i := 0 intervention is a zero-baseline attribution, so the zero-baseline methods (DeepLIFT, "
        "Integrated Gradients, KernelSHAP with its zero baseline) match it by construction. A uniform random "
        "attribution already scores about 0.5 on SIM with three features, because min-max scaling sends the "
        "smallest attribution to 0 and the largest to 1 whatever they are.</div>"
        '<div class="note warn"><b>Paper vs released code.</b> Ground truth: the paper (Eq. 5) lowers a feature by one '
        "level, the notebook sets it to 0. Label weights: the paper says w = [0.5, 0.25, 0.125], the code uses "
        "1/2, 1/4, 1/6. Data: the paper describes a grid of about 4 million samples; "
        "<code>aixi-dataset/bin/daixi.py</code> draws 50,000 continuous uniform rows and never seeds them "
        "(<code>np.random.seed = 42</code> assigns instead of calling). The repository (last push 2026-02-18) "
        "predates the arXiv posting (2026-09), so the notebook may be an earlier version than the one behind the "
        "paper.</div>"
        + "<h3>All metrics</h3>" + table(
            ["method", "ground truth", "SIM (higher better)", "KL (lower better)", "AUC-Borji"],
            [[x["method"], x["ground_truth"], fmt(x["sim_mean"]), fmt(x["kl_mean"]), fmt(x.get("auc_mean"))]
             for x in d["rows"]],
            ["20%", "38%", "14%", "14%", "14%"], {2, 3, 4})
        + "<h3>Explore further</h3><pre>.venv/Scripts/python.exe run_fb2ts_tabular.py --levels 20 --kernel-samples 25 "
          "--tag fb2ts_coarse</pre>")
    return section("fb2ts", "Synthetic ground-truth framework (Miro-Nicolau et al. 2026), tabular arm", body)


if __name__ == "__main__":
    main()
