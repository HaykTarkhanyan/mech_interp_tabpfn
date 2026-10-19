"""Build the standalone HTML report from the saved JSON results.

Run:  uv run python report.py                  (~5 s; reads results/*.json, no model calls)
      uv run python report.py --pilot results/pilot_smoke.json --out results/smoke_report.html

Inputs: results/pilot.json (run_pilot.py), results/check_model.json (check_model.py),
results/validate_ale.json (validate_ale.py). Output: results/pilot_report.html, one file with
plotly.js inlined, nothing loaded from the network. The H1/H2 verdicts are computed here by the
pre-registered rules in delta_b.py, not written by hand.
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ale import cell_average
from common import RESULTS_DIR, setup
from delta_b import BEHAVIORS, h1_verdict, h2_verdict, score_sets
from model import SITE_LABEL

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
SITE_COLOR = {"feat": RED, "item": BLUE, "mlp": ORANGE}
BEH_LABEL = {"ale_x1": "ALE X1 (main)", "ale_x2": "ALE X2 (main, non-target)", "ale_12": "ALE X1*X2 (interaction, target)"}
PL_LABEL = {"same_pair": "same pair (X1, X2 in columns 0, 1)", "diff_pair": "different pairs (X1, X2 in columns 0, 2)"}
_first_fig = True


def headroom(fig: go.Figure, log: bool, extra: tuple = ()) -> None:
    """y-range that leaves room for outside bar labels and includes the reference lines."""
    ys = [float(v) for tr in fig.data if tr.type == "bar" for v in tr.y if v is not None] + [float(v) for v in extra]
    if log:
        ys = [v for v in ys if v > 0]
        if ys:
            fig.update_yaxes(range=[np.log10(min(ys)) - 0.15, np.log10(max(ys)) + 0.35])
    elif ys:
        fig.update_yaxes(range=[min(0.0, min(ys)), max(ys) * 1.2])


def fig_html(fig: go.Figure, height: int = 420) -> str:
    global _first_fig
    fig.update_layout(height=height, margin=dict(l=60, r=40, t=60, b=95), template="plotly_white",
                      font=dict(family="Segoe UI, Arial, sans-serif", size=12),
                      legend=dict(orientation="h", x=0, y=-0.16, yanchor="top"))
    inc = "inline" if _first_fig else False
    _first_fig = False
    return pio.to_html(fig, include_plotlyjs=inc, full_html=False, config={"displaylogo": False})


def section(sid: str, title: str, body: str) -> str:
    return f'<section id="{sid}"><details open><summary><h2>{html.escape(title)}</h2></summary>\n{body}\n</details></section>\n'


def esc(x) -> str:
    return html.escape(str(x))


def heat(values: list[float]) -> list[str]:
    """Diverging green-white-red background around the median, alpha capped at 0.4."""
    v = np.asarray(values, dtype=float)
    med = np.median(v)
    lo, hi = v.min(), v.max()
    out = []
    for x in v:
        if x >= med:
            a = 0.0 if hi == med else 0.4 * (x - med) / (hi - med)
            out.append(f"background: rgba(217,0,18,{a:.2f})")
        else:
            a = 0.0 if lo == med else 0.4 * (med - x) / (med - lo)
            out.append(f"background: rgba(0,140,60,{a:.2f})")
    return out


def table(headers: list[str], widths: list[int], rows: list[list], styles: list[list[str]] | None = None) -> str:
    cols = "".join(f'<col style="width:{w}%">' for w in widths)
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for i, r in enumerate(rows):
        cells = []
        for j, c in enumerate(r):
            st = styles[i][j] if styles and styles[i][j] else ""
            cls = ' class="num"' if isinstance(c, (int, float)) or (isinstance(c, str) and c[:1].isdigit()) else ""
            cells.append(f'<td{cls} style="{st}">{esc(c)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table><colgroup>{cols}</colgroup><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def bar_by_site(rows: list[dict], key: str, title: str, ytitle: str, fmt: str = ".2f", hlines=(), log_y=False) -> go.Figure:
    fig = go.Figure()
    for site in ("feat", "item", "mlp"):
        rs = [r for r in rows if r["site"] == site]
        fig.add_bar(x=[f"L{r['layer']}" for r in rs], y=[r[key] for r in rs], name=SITE_LABEL[site],
                    marker_color=SITE_COLOR[site], text=[format(r[key], fmt) for r in rs],
                    textposition="outside", textfont=dict(size=9), cliponaxis=False)
    for y, label, color, dash in hlines:
        fig.add_hline(y=y, line_dash=dash, line_color=color)
        # legend entry for the line (hline annotations are misplaced on log axes)
        fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=color, dash=dash), name=f"{label} = {y:.3g}")
    fig.update_layout(barmode="group", title=title, yaxis_title=ytitle, xaxis_title="layer")
    if log_y:
        fig.update_yaxes(type="log")
    headroom(fig, log_y, tuple(h[0] for h in hlines))
    return fig


def estimator_floor(floor_rows: list[dict], key: str) -> float:
    """Largest no-ablation change from re-estimating B on the SAME context (query and grid resamples only)."""
    vals = [r["delta"][key] for r in floor_rows if r["source"] in ("query", "grid")]
    if not vals:
        raise ValueError("no query/grid floor resamples in pilot.json")
    return max(vals)


def plain_section(pls: dict, scored: dict) -> str:
    """Plain-language explanation plus a post hoc within-context re-scoring (not the pre-registered rule)."""
    out = [
        "<p><b>What the pilot does.</b> A toy task where the label depends on X1 <i>times</i> X2, so neither feature "
        "alone carries the signal. TabPFN v2 gets 256 rows as its context, and we measure its <b>interaction picture</b>: "
        "the second-order ALE surface of X1 and X2. Then we switch off one group of components at a time (all feature-attention "
        "heads of one layer, all item-attention heads, or one MLP) and measure how much the picture changes (Delta_target).</p>",
        "<p><b>A change only counts if it passes three checks:</b> (1) <b>bigger than random</b>: switching off random component "
        "sets of the same size changes the picture less (null percentile, at least 95); (2) <b>bigger than noise</b>: it beats how "
        "much the picture wobbles with nothing switched off (noise floor); (3) <b>selective</b>: it changes the interaction more "
        "than an unrelated effect, the X2 main effect (S &gt; 1).</p>",
        "<h3>Finding 1: the behavior itself depends heavily on which 256 rows TabPFN sees</h3>",
        "<p>Median change of the interaction picture with <b>nothing</b> switched off:</p>",
    ]
    rows = []
    for p in pls:
        fl = pls[p]["floor"]
        med = {s: float(np.median([r["delta"]["ale_12"] for r in fl if r["source"] == s])) for s in ("query", "grid", "context")}
        rows.append([PL_LABEL[p], f"{med['query']:.1%}", f"{med['grid']:.1%}", f"{med['context']:.1%}"])
    out.append(table(["Placement", "Same rows, new evaluation sample", "Same rows, shifted ALE grid", "Fresh 256-row context"],
                     [34, 22, 22, 22], rows))
    out.append("<p>Re-estimating the picture on the same rows moves it by a few percent. Handing TabPFN a fresh sample from the "
               "<i>same</i> task moves it by about half its size. TabPFN builds a new function from every context, so a claim "
               "like \"component C implements the interaction\", made on one context, may not hold on the next. That makes "
               "RQ3 (do mechanisms carry over across contexts?) central rather than a side question.</p>")

    out.append("<h3>Finding 2: almost nothing passed the pre-registered rule, partly because its noise bar was the wrong one</h3>")
    frows = []
    for p in pls:
        frows.append([PL_LABEL[p], f"{scored[p][0]['floor_target']:.3f}", f"{estimator_floor(pls[p]['floor'], 'ale_12'):.3f}"])
    out.append(table(["Placement", "Pre-registered floor (includes fresh contexts)", "Same-context floor (re-estimation only)"],
                     [40, 30, 30], frows))
    out.append("<p>The pre-registered floor includes the context-to-context variation from Finding 1. But an ablation is compared "
               "with the <i>same</i> context with nothing switched off, so the fair bar for that comparison is the much smaller "
               "same-context floor. (Like testing whether removing a part slows a car, but only counting changes bigger than the "
               "speed difference between different cars.) The large floor belongs to RQ3, not to within-context claims. "
               "This mix-up came from the proposal review's wording, not from the pipeline.</p>")

    out.append("<h3>Finding 3 (post hoc, not pre-registered): with the same-context floor, one attention set per placement survives</h3>")
    srows, passing = [], {}
    for p in pls:
        est_t, est_x2 = estimator_floor(pls[p]["floor"], "ale_12"), estimator_floor(pls[p]["floor"], "ale_x2")
        passing[p] = []
        for r in scored[p]:
            if r["site"] == "mlp":
                continue
            s_est = r["delta_target"] / max(r["delta_nontarget"], est_x2)
            ok = r["delta_target"] > est_t and r["null_pct_target"] >= 95 and s_est > 1
            x1_more = r["delta_x1"] >= r["delta_target"]
            if ok and not x1_more:
                passing[p].append(f"{SITE_LABEL[r['site']]} L{r['layer']}")
            if r["null_pct_target"] >= 95:
                srows.append([PL_LABEL[p], f"{SITE_LABEL[r['site']]} L{r['layer']}", f"{r['delta_target']:.3f}",
                              f"{r['null_pct_target']:.0f}", f"{s_est:.2f}", "yes" if x1_more else "no",
                              "yes" if ok and not x1_more else "no"])
    out.append("<p>Attention sets that beat at least 95% of the random null, re-scored against the same-context floor. "
               "MLP layers are left out because their null is not a fair comparison (Finding 4).</p>")
    out.append(table(["Placement", "Component", "Delta target", "Null pct", "S (same-context floor)",
                      "Changes X1 more than the interaction?", "Passes all checks"], [25, 15, 11, 9, 13, 15, 12], srows))
    out.append("<p><b>Reading:</b> "
               + "; ".join(f"{PL_LABEL[p]}: {', '.join(v) if v else 'none'}" for p, v in passing.items())
               + ". Several other sets beat every random set but change the X2 or X1 main effect at least as much as the "
               "interaction, so they look like general damage rather than an interaction mechanism. The selective set switches "
               "from feature attention to item attention when X1 and X2 move into different feature pairs, which is what H2 "
               "predicts. One context per placement: suggestive, not evidence.</p>")

    zs = {(r["site"], r["layer"]): r["delta"]["ale_12"] for r in pls["same_pair"]["zero_scan"]}
    rho = spearman([r["delta_target"] for r in scored["same_pair"]],
                   [zs[(r["site"], r["layer"])] for r in scored["same_pair"]]) if zs else float("nan")
    n_mlp = {p: (sum(r["null_pct_target"] >= 95 for r in scored[p] if r["site"] == "mlp"),
                 sum(r["site"] == "mlp" for r in scored[p])) for p in scored}
    out.append("<h3>Finding 4: two warnings about the method</h3><ul>")
    out.append(f"<li><b>How you switch a component off changes the answer.</b> Replacing it with its average (mean ablation) and "
               f"with zeros (zero ablation) rank the components quite differently: rank correlation {rho:.2f}. The thesis "
               f"has to pick one replacement and justify it.</li>")
    out.append("<li><b>The MLP comparison is unfair.</b> A whole MLP layer is compared with random units scattered across all "
               "layers, and concentrated damage hurts more regardless of what the layer does: "
               + ", ".join(f"{a} of {b} MLP layers beat the null ({p.replace('_', ' ')})" for p, (a, b) in n_mlp.items())
               + ". The null has to match the structure of what it is compared with.</li></ul>")
    out.append("<h3>What it means</h3><p>The pipeline works and its checks run. The main lesson is that one context is not "
               "enough: the next experiment repeats the scan on about 10 fresh contexts per placement, with the floor split "
               "into estimator noise (for within-context claims) and context variation (for RQ3), and a structure-matched "
               "MLP null.</p>")
    return "".join(out)


def build(pilot: dict, check: dict | None, val: dict | None) -> tuple[str, dict]:
    meta = pilot["meta"]
    pls = pilot["placements"]
    scored = {p: score_sets(pls[p]) for p in pls}
    v_h1 = {p: h1_verdict(scored[p]) for p in pls}
    v_h2 = h2_verdict(scored["same_pair"], scored["diff_pair"])
    parts = []
    toc = []

    def add(sid, title, body):
        toc.append((sid, title))
        parts.append(section(sid, title, body))

    # 0. plain-language explanation (post hoc reading, computed from the same JSON)
    add("plain", "0. In plain words: what the pilot found", plain_section(pls, scored))

    # 1. summary
    rs = scored["same_pair"]
    att = [r for r in rs if r["site"] in ("feat", "item")]
    top = max(rs, key=lambda r: r["delta_target"])
    top_s = max(rs, key=lambda r: r["selectivity"])
    n_pass = {p: sum(r["passes"] for r in scored[p]) for p in scored}
    verdict_rows = [
        [f"H1, {PL_LABEL['same_pair']}", v_h1["same_pair"]["verdict"], v_h1["same_pair"]["reason"]],
        [f"H1, {PL_LABEL['diff_pair']}", v_h1["diff_pair"]["verdict"], v_h1["diff_pair"]["reason"]],
        ["H2, same vs different pair", v_h2["verdict"], v_h2["reason"]],
    ]
    bullets = [
        f"Passing sets (floor, null p95 and S &gt; 1): {n_pass['same_pair']} of {len(rs)} in the same-pair placement, "
        f"{n_pass['diff_pair']} of {len(scored['diff_pair'])} in the different-pair placement.",
        f"Largest change of the interaction surface (same pair): {SITE_LABEL[top['site']]} layer {top['layer']}, "
        f"Delta_target = {top['delta_target']:.3f} (floor {top['floor_target']:.3f}), null percentile {top['null_pct_target']:.0f}, "
        f"S = {top['selectivity']:.2f}.",
        f"Most selective set (same pair): {SITE_LABEL[top_s['site']]} layer {top_s['layer']}, S = {top_s['selectivity']:.2f}, "
        f"Delta_target = {top_s['delta_target']:.3f}, null percentile {top_s['null_pct_target']:.0f}.",
        f"Noise floor (largest joint no-ablation resample): interaction {rs[0]['floor_target']:.3f}, "
        f"X2 main effect {rs[0]['floor_nontarget']:.3f}. Attention sets above the interaction floor: "
        f"{sum(r['above_floor'] for r in att)} of {len(att)}.",
    ]
    # data-driven caveats that are not part of the pre-registered rule but change how to read it
    reads = []
    for p in pls:
        med = {s: np.median([r["delta"]["ale_12"] for r in pls[p]["floor"] if r["source"] == s]) for s in ("query", "grid", "context", "joint")}
        reads.append(f"Floor composition, {p}: median no-ablation change of the interaction surface is "
                     + ", ".join(f"{s} {v:.3f}" for s, v in med.items())
                     + ". The floor is mostly context-to-context variability of g_D, not estimator noise.")
    for p in pls:
        for r in scored[p]:
            if r["passes"] and r["delta_x1"] >= r["delta_target"]:
                reads.append(f"{p}: {SITE_LABEL[r['site']]} layer {r['layer']} passes against the X2 main effect but changes the "
                             f"X1 main effect more than the interaction (Delta_X1 {r['delta_x1']:.3f} vs Delta_target "
                             f"{r['delta_target']:.3f}); it is not selective against X1.")
    zs = {(r["site"], r["layer"]): r["delta"]["ale_12"] for r in pls["same_pair"]["zero_scan"]}
    if zs:
        rho = spearman([r["delta_target"] for r in rs], [zs[(r["site"], r["layer"])] for r in rs])
        reads.append(f"Zero vs mean ablation, same pair: Spearman rank correlation of Delta_target over {len(rs)} sets = {rho:.2f}.")
    for p in pls:
        mlp = [r for r in scored[p] if r["site"] == "mlp"]
        reads.append(f"{p}: {sum(r['null_pct_target'] >= 95 for r in mlp)} of {len(mlp)} MLP layer sets beat at least 95% of the "
                     "scattered-unit MLP null, which suggests the MLP null measures concentration rather than relevance (DECISIONS #15).")
    body = ("<p>Verdicts are computed mechanically by the rules pre-registered in README.md and DECISIONS #17 "
            "(code: <code>delta_b.py</code>). One context draw, one task: a pilot signal, not evidence in general.</p>"
            + table(["Hypothesis", "Verdict", "Reason (from the rule)"], [28, 14, 58], verdict_rows)
            + "<h3>Key numbers</h3><ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
            + "<h3>Read the verdicts with these (computed from the same data, not part of the rule)</h3><ul>"
            + "".join(f"<li>{esc(b)}</li>" for b in reads) + "</ul>")
    add("summary", "1. Summary and pre-registered verdicts", body)

    # 2. setup
    ck = meta["checkpoint"]
    setup_rows = [
        ["Model", f"TabPFN v2 classifier, {ck['file']} ({ck['repo']}), {ck['n_params']:,} parameters"],
        ["Checkpoint sha256", ck["sha256"]],
        ["Wrapper", "1 estimator, no feature or class shuffling, no fingerprint, no polynomial or SVD features, "
                    "identity transform; the model's internal z-scoring and pair embedding stay active"],
        ["Task", f"sigmoid(X1 + (X2^2 - 4/3) + {meta['task']['gamma']} X1 X2 1{{X3 > {meta['task']['threshold']}}}); "
                 f"{meta['feature_distribution']}"],
        ["Placements", "; ".join(f"{k}: {' '.join(v)}" for k, v in meta["placements"].items())],
        ["Context / reference rows", f"{meta['n_ctx']} context rows; {meta['n_ref']} reference rows ({meta['n_per_cell']} per (X1, X2) cell)"],
        ["Prediction scale", meta["prediction_scale"]],
        ["ALE grid", f"first order {meta['k1']} equal bins, second order {meta['k2']} x {meta['k2']} cells on [-2, 2]"],
        ["Ablation", meta["ablation_primary"] + "; zero ablation as contrast (same pair only)"],
        ["Null", f"{meta['n_null']} random sets per type: 6 feature heads, 6 item heads, or 768 MLP units drawn across all layers"],
        ["Floor", f"no-ablation resamples {meta['n_floor']}; floor = largest joint value; grid jitter {meta['jitter']} bin widths"],
        ["Runtime", f"{meta['elapsed_s']:.0f} s total, {meta['mean_seconds_per_job']:.2f} s per ablation/floor pass, "
                    f"torch {meta['torch']} with {meta['threads']} threads"],
    ]
    add("setup", "2. Setup", table(["Item", "Value"], [22, 78], setup_rows))

    # 3. verification
    body = ""
    if val:
        body += "<h3>ALE estimator on closed-form functions (validate_ale.py)</h3>"
        body += table(["Case", "Metric", "Value", "Tolerance"], [30, 40, 15, 15],
                      [[c["case"], c["metric"], f"{c['value']:.2e}", f"{c['tolerance']:.0e}"] for c in val["checks"]])
    if check:
        we, hc = check["wrapper_equivalence"], check["hook_checks"]
        body += "<h3>Model and hooks (check_model.py)</h3>"
        body += table(["Check", "Result"], [60, 40], [
            ["max |wrapper input - our columns|", f"{we['max_abs_input_diff']:.1e}"],
            ["max |wrapper raw logits - direct forward|", f"{we['max_abs_logit_diff']:.1e}"],
            ["hooks inactive vs removed, bitwise identical", hc["inactive_hooks_bitwise_identical"]],
            ["self-patch of 5 sites, bitwise identical", hc["self_patch_bitwise_identical"]],
            ["forward pass deterministic", hc["deterministic"]],
            ["timing, s per pass at pilot size", ", ".join(f"{t:.2f}" for t in check["timing"]["seconds"])],
        ])
    add("verify", "3. Verification", body or "<p>No verification files found.</p>")

    # 4. behavior check
    bc = pilot["behavior_check"]
    body = ("<p>Does TabPFN v2 reproduce the known structure at all? Model ALE (red) vs the ALE of the "
            "true data-generating logit (blue), computed by the same estimator on the same rows. "
            "Same-pair placement, same context rows for every task; only the labels differ.</p>")
    norm_rows = []
    for tname, rec in bc.items():
        fig = make_subplots(rows=1, cols=4, subplot_titles=("ALE X1", "ALE X2", "ALE X1*X2, model", "ALE X1*X2, truth"),
                            horizontal_spacing=0.07)
        e1, e2 = np.array(pilot["grid"]["e1"]), np.array(pilot["grid"]["e2"])
        for col, k in ((1, "ale_x1"), (2, "ale_x2")):
            fig.add_scatter(x=e1, y=rec["model"][k], mode="lines+markers", line_color=RED, name="model", showlegend=col == 1, row=1, col=col)
            fig.add_scatter(x=e1, y=rec["truth"][k], mode="lines", line=dict(color=BLUE, dash="dash"), name="truth", showlegend=col == 1, row=1, col=col)
        cm = cell_average(np.array(rec["model"]["ale_12"]))
        ct = cell_average(np.array(rec["truth"]["ale_12"]))
        lim = float(max(np.abs(cm).max(), np.abs(ct).max(), 1e-6))
        mids = (e2[:-1] + e2[1:]) / 2
        for col, c in ((3, cm), (4, ct)):
            fig.add_heatmap(z=c.T, x=mids, y=mids, zmin=-lim, zmax=lim, colorscale="RdBu_r", showscale=col == 4,
                            text=np.round(c.T, 2), texttemplate="%{text}", row=1, col=col)
        fig.update_xaxes(title_text="X1", row=1, col=1)
        fig.update_xaxes(title_text="X2", row=1, col=2)
        fig.update_xaxes(title_text="X1", row=1, col=3)
        fig.update_xaxes(title_text="X1", row=1, col=4)
        fig.update_yaxes(title_text="X2", row=1, col=3)
        g = rec["spec"]["gamma"]
        fig.update_layout(title=f"Task '{tname}' (gamma = {g}, h1 = {rec['spec']['h1']}, h2 = {rec['spec']['h2']})")
        body += fig_html(fig, 380)
        for k in BEHAVIORS:
            norm_rows.append([tname, BEH_LABEL[k], f3(rec["norm_model"][k]), f3(rec["norm_truth"][k]), f3(rec["rel_dist_to_truth"][k])])
    body += "<h3>Norms (log-odds units) and distance to the truth</h3>"
    body += table(["Task", "Behavior", "||model||", "||truth||", "rel. distance to truth"], [16, 36, 16, 16, 16], norm_rows)
    body += ("<p>For the flat task the main-effect norms of the truth are zero, so the relative distance is undefined "
             "(n/a). That is why selectivity is not run on the flat task in this pilot (DECISIONS #14).</p>")
    add("behavior", "4. Behavior check: does the model recover the known effects?", body)

    # 5. noise floor
    body = ("<p>How much B moves with no ablation at all. Bars show the largest relative distance per source "
            "(hover for every value). The joint maximum is the floor used in the pass rule.</p>")
    for p in pls:
        fl = pls[p]["floor"]
        fig = go.Figure()
        for k, color in zip(BEHAVIORS, (ORANGE, BLUE, RED)):
            srcs = ["query", "grid", "context", "joint"]
            ys = [max((r["delta"][k] for r in fl if r["source"] == s), default=0.0) for s in srcs]
            hov = [", ".join(f"{r['delta'][k]:.3f}" for r in fl if r["source"] == s) for s in srcs]
            fig.add_bar(x=srcs, y=ys, name=BEH_LABEL[k], marker_color=color, text=[f"{y:.3f}" for y in ys],
                        textposition="outside", hovertext=hov, cliponaxis=False)
        fig.update_layout(barmode="group", title=f"Noise floor, {PL_LABEL[p]}", yaxis_title="relative L2 distance")
        headroom(fig, False)
        body += fig_html(fig, 380)
    add("floor", "5. Noise floor (no ablation)", body)

    # 6. layer scan, same pair
    def scan_block(p: str) -> str:
        rows = scored[p]
        floor_t = rows[0]["floor_target"]
        p95 = {s: next(r["null_p95_target"] for r in rows if r["site"] == s) for s in ("feat", "item", "mlp")}
        hl = [(floor_t, "floor", "black", "dash")] + [
            (p95[s], f"null p95 {SITE_LABEL[s]}", SITE_COLOR[s], "dot") for s in p95]
        b = fig_html(bar_by_site(rows, "delta_target", f"Delta_target (interaction surface), {PL_LABEL[p]}",
                                 "relative L2 change, log scale", ".3f", hl, log_y=True), 460)
        b += fig_html(bar_by_site(rows, "null_pct_target", "Null percentile of Delta_target (pass needs >= 95)",
                                  "percentile", ".0f", [(95, "95", "black", "dash")]), 400)
        b += fig_html(bar_by_site(rows, "selectivity", "Selectivity S = Delta_target / max(Delta_X2, floor_X2)",
                                  "S", ".2f", [(1, "S = 1", "black", "dash")], log_y=True), 420)
        trs, sts = [], []
        dts = [r["delta_target"] for r in rows]
        ss = [r["selectivity"] for r in rows]
        hd, hs = heat(dts), heat(ss)
        for i, r in enumerate(rows):
            trs.append([SITE_LABEL[r["site"]], r["layer"], f"{r['delta_target']:.3f}", f"{r['delta_nontarget']:.3f}",
                        f"{r['delta_x1']:.3f}", f"{r['null_pct_target']:.0f}", f"{r['selectivity']:.2f}",
                        f"{r['null_pct_selectivity']:.0f}", "yes" if r["passes"] else "no",
                        f"{r['pred']['mean_abs_change_logodds']:.2f}", f"{r['pred']['acc_vs_bayes_ablated']:.2f}"])
            sts.append(["", "", hd[i], "", "", "", hs[i], "", "font-weight:bold" if r["passes"] else "", "", ""])
        b += table(["Component", "Layer", "Delta target", "Delta X2", "Delta X1", "Null pct", "S", "Null pct of S",
                    "Passes", "mean |d log-odds|", "acc vs Bayes"], [14, 6, 9, 9, 8, 8, 8, 9, 7, 11, 11], trs, sts)
        return b

    add("scan", "6. Layer scan, same pair (mean ablation)", scan_block("same_pair"))

    # 7. null distributions
    body = ("<p>Each box is the null: Delta_target for 25 random sets of the same size and type spread across all layers. "
            "Dots are the 12 layer sets of that type.</p>")
    for p in pls:
        fig = go.Figure()
        for site in ("feat", "item", "mlp"):
            nv = [r["delta"]["ale_12"] for r in pls[p]["null"] if r["site"] == site]
            fig.add_box(y=nv, name=SITE_LABEL[site], marker_color=SITE_COLOR[site], boxpoints="all", jitter=0.4,
                        pointpos=-1.6, showlegend=False)
            rs_ = [r for r in scored[p] if r["site"] == site]
            fig.add_scatter(x=[SITE_LABEL[site]] * len(rs_), y=[r["delta_target"] for r in rs_], mode="markers+text",
                            text=[f"L{r['layer']}" for r in rs_], textposition="middle right",
                            marker=dict(color="black", symbol="diamond", size=8), showlegend=False)
        fig.update_layout(title=f"Null vs layer sets, {PL_LABEL[p]}", yaxis_title="Delta_target (log scale)")
        fig.update_yaxes(type="log")
        body += fig_html(fig, 460)
    add("null", "7. Null distributions", body)

    # 8. placement comparison
    rows_s, rows_d = scored["same_pair"], scored["diff_pair"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[SITE_LABEL[s] for s in ("feat", "item", "mlp")], horizontal_spacing=0.06)
    corr_rows = []
    for ci, site in enumerate(("feat", "item", "mlp"), start=1):
        a = [r for r in rows_s if r["site"] == site]
        b_ = [r for r in rows_d if r["site"] == site]
        for rr, name, color in ((a, "same pair", BLUE), (b_, "different pairs", ORANGE)):
            fig.add_bar(x=[f"L{r['layer']}" for r in rr], y=[r["delta_target"] for r in rr], name=name, marker_color=color,
                        showlegend=ci == 1, text=[f"{r['delta_target']:.2f}" for r in rr], textposition="outside",
                        textfont=dict(size=8), cliponaxis=False, row=1, col=ci)
        corr_rows.append([SITE_LABEL[site], f"{spearman([r['delta_target'] for r in a], [r['delta_target'] for r in b_]):.2f}",
                          ", ".join(f"L{r['layer']}" for r in a if r["passes"]) or "none",
                          ", ".join(f"L{r['layer']}" for r in b_ if r["passes"]) or "none"])
    fig.update_yaxes(type="log")
    headroom(fig, True)
    fig.update_layout(barmode="group", title="Delta_target by placement (log scale)")
    nrm = [[PL_LABEL[p], f3(pls[p]["clean"]["norm"]["ale_12"]), f3(pls[p]["clean"]["norm"]["ale_x2"]),
            f3(pls[p]["clean"]["norm"]["ale_x1"]), f"{pls[p]['clean']['acc_vs_bayes']:.2f}"] for p in pls]
    body = (f"<p>H2 verdict: <b>{esc(v_h2['verdict'])}</b>, {esc(v_h2['reason'])}.</p>"
            + table(["Placement", "||ALE_12||", "||ALE_X2||", "||ALE_X1||", "acc vs Bayes"], [40, 15, 15, 15, 15], nrm)
            + fig_html(fig, 440)
            + table(["Component", "Spearman of layer profiles", "Passing, same pair", "Passing, different pairs"],
                    [25, 25, 25, 25], corr_rows))
    body += "<h3>Full scan, different-pair placement</h3>" + scan_block("diff_pair")
    add("placement", "8. Same pair vs different pairs (H2)", body)

    # 9. zero vs mean
    zs = {(r["site"], r["layer"]): r for r in pls["same_pair"]["zero_scan"]}
    fig = go.Figure()
    mean_v, zero_v = [], []
    for site in ("feat", "item", "mlp"):
        rr = [r for r in rows_s if r["site"] == site and (site, r["layer"]) in zs]
        xm = [r["delta_target"] for r in rr]
        yz = [zs[(site, r["layer"])]["delta"]["ale_12"] for r in rr]
        mean_v += xm
        zero_v += yz
        fig.add_scatter(x=xm, y=yz, mode="markers+text", text=[f"L{r['layer']}" for r in rr], textposition="top center",
                        marker=dict(color=SITE_COLOR[site], size=9), name=SITE_LABEL[site])
    if mean_v:
        lo, hi = min(mean_v + zero_v), max(mean_v + zero_v)
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="gray", dash="dash"), name="equal")
    fig.update_xaxes(type="log", title="Delta_target, mean ablation")
    fig.update_yaxes(type="log", title="Delta_target, zero ablation")
    fig.update_layout(title="Zero vs mean ablation, same pair")
    rho = spearman(mean_v, zero_v) if len(mean_v) > 2 else float("nan")
    body = (f"<p>Spearman rank correlation of Delta_target between the two ablation types over {len(mean_v)} sets: "
            f"<b>{rho:.2f}</b>. Low agreement would mean conclusions depend on the replacement value (proposal section 6).</p>"
            + fig_html(fig, 460))
    add("zero", "9. Zero vs mean ablation", body)

    # 10. prediction-level diagnostics
    body = ("<p>Secondary diagnostics on the 128 unmodified reference rows: mean absolute change of the log-odds, "
            "and accuracy against the Bayes decision (sign of the true logit; clean accuracy "
            f"{pls['same_pair']['clean']['acc_vs_bayes']:.2f}). Large values here with S near 1 mean general damage, "
            "not a selective mechanism.</p>")
    body += fig_html(_pred_fig(rows_s), 420)
    add("pred", "10. Prediction-level diagnostics", body)

    # 11. caveats
    cav = [
        "One context draw (256 rows) and one task. The verdicts are a pilot signal; RQ3 (reuse across contexts) is untested here.",
        "25 null sets per type: the null percentile moves in steps of 4 points, and passing means beating at least 24 of 25.",
        "The floor (how much B moves between independent estimates) is conservative for Delta_B, which compares the same rows "
        "on the same grid; a paired bootstrap of Delta_B would be tighter (DECISIONS #16).",
        "The null spreads its heads or units across layers, while candidates concentrate them in one layer. If every layer set beats "
        "the null, the null measures concentration rather than relevance (DECISIONS #15).",
        "Whole-layer sets only. Single heads, cross-layer sets and activation patching from a gamma = 0 context are not run here.",
        "Mean ablation uses means from the same context and reference rows; zero ablation is the only alternative replacement tested.",
        "Item attention for query rows is multi-query in TabPFN v2 (all query heads read key/value head 0).",
    ]
    add("caveats", "11. Caveats", "<ul>" + "".join(f"<li>{esc(c)}</li>" for c in cav) + "</ul><p>GPU-scale follow-ups: see RUNPOD.md.</p>")

    toc_html = "<nav id='toc'><h2>Contents</h2><ul>" + "".join(f'<li><a href="#{s}">{esc(t)}</a></li>' for s, t in toc) + "</ul></nav>"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feature-effect ablation pilot</title>
<style>
:root {{ --fg:#1d1d1f; --muted:#5a5a66; --line:#d9d9e0; --bg:#ffffff; --accent:#0033A0; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; color: var(--fg); background: var(--bg); max-width: 1180px;
        margin: 0 auto; padding: 16px 16px 60px; line-height: 1.45; }}
h1 {{ font-size: 1.6em; margin-bottom: 0.2em; }}
h2 {{ display: inline; font-size: 1.25em; }}
h3 {{ font-size: 1.05em; margin-top: 1.4em; }}
summary {{ cursor: pointer; padding: 6px 0; border-bottom: 1px solid var(--line); }}
section {{ margin-top: 18px; }}
nav#toc {{ background: #f5f6fa; border: 1px solid var(--line); padding: 8px 16px; border-radius: 6px; }}
nav#toc h2 {{ display: block; font-size: 1.05em; margin: 6px 0; }}
nav#toc ul {{ list-style: none; padding-left: 8px; margin: 4px 0; }}
a {{ color: var(--accent); }}
table {{ table-layout: fixed; width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.86em; }}
th, td {{ border: 1px solid var(--line); padding: 4px 6px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
th {{ background: #f0f1f6; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.meta {{ color: var(--muted); font-size: 0.9em; }}
code {{ background: #f0f1f6; padding: 0 3px; border-radius: 3px; }}
</style></head><body>
<h1>Feature effects as behavioral specifications: TabPFN v2 ablation pilot</h1>
<p class="meta">Run started {esc(meta['started'])}; built {time.strftime('%Y-%m-%d %H:%M')} from results/pilot.json by report.py.
Seed {meta['seed']}. {'SMOKE RUN, not the pilot.' if meta['smoke'] else ''}</p>
<p>Can an ALE interaction surface serve as the target of an ablation study, with a random-component null, a noise floor
and a non-target control? This pilot asks it of the X1*X2 interaction in one synthetic task and one context.</p>
{toc_html}
{''.join(parts)}
</body></html>"""
    summary = {"h1": v_h1, "h2": v_h2, "n_pass": n_pass}
    return page, summary


def _pred_fig(rows: list[dict]) -> go.Figure:
    flat = [{**r, "mabs": r["pred"]["mean_abs_change_logodds"]} for r in rows]
    return bar_by_site(flat, "mabs", "Mean |change in log-odds| on reference rows, same pair", "log-odds, log scale", ".2f", log_y=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=str(RESULTS_DIR / "pilot.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "pilot_report.html"))
    args = ap.parse_args()
    log = setup("report")
    t0 = time.perf_counter()
    pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
    ck_path, val_path = RESULTS_DIR / "check_model.json", RESULTS_DIR / "validate_ale.json"
    check = json.loads(ck_path.read_text(encoding="utf-8")) if ck_path.exists() else None
    val = json.loads(val_path.read_text(encoding="utf-8")) if val_path.exists() else None
    if check is None or val is None:
        log.warning(f"missing verification file(s): check_model={check is not None}, validate_ale={val is not None}")
    page, summary = build(pilot, check, val)
    Path(args.out).write_text(page, encoding="utf-8")
    log.info(f"verdicts: H1 same={summary['h1']['same_pair']['verdict']}, H1 diff={summary['h1']['diff_pair']['verdict']}, "
             f"H2={summary['h2']['verdict']}; passing sets {summary['n_pass']}")
    log.info(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
