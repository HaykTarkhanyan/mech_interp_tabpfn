#!/usr/bin/env python3
"""HTML report for frontier_probe.py: held-out R^2 against the real labels for each target family, and the cost of
exact SHAP for the tractable ones. Reads results/frontier/frontier.json; recomputes nothing.

Usage (from synthetic_data/, ~2 s):
    .venv/Scripts/python.exe scripts/frontier_report.py
Writes results/frontier/frontier_report.html (standalone, plotly inlined).
"""

import html
import json
import logging
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "frontier"
COLORS = {"linear": "#7a7a7a", "additive splines": "#F2A800", "additive boosted": "#D90012",
          "pairwise boosted": "#0033A0", "full boosted": "#2a9d8f"}
CSS = """:root{--bg:#fff;--fg:#1d1d1f;--muted:#5f6368;--line:#e3e3e3;--card:#f7f7f8;--accent:#0033A0}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#e8e8ea;--muted:#a0a3a8;--line:#333;--card:#1d1d21;--accent:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,"Segoe UI",sans-serif}
main{max-width:1050px;margin:0 auto;padding:24px 16px 80px}details{border-top:1px solid var(--line);padding:6px 0}
summary{cursor:pointer}summary h2{display:inline;font-size:20px}.toc{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:10px 16px}.toc a{color:var(--accent)}.tw{overflow-x:auto}table{border-collapse:collapse;
table-layout:fixed;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left}
th{background:var(--card)}td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{background:var(--card);border-left:3px solid var(--accent);padding:8px 12px;margin:10px 0}"""


def heat(v: float, lo: float, mid: float, hi: float) -> str:
    """Diverging green-white-red around the median, alpha capped at 0.4 (higher R^2 = green)."""
    if not np.isfinite(v):
        return ""
    if v >= mid:
        a = 0.4 * min(1.0, (v - mid) / (hi - mid + 1e-12))
        return f"background:rgba(46,160,67,{a:.2f})"
    a = 0.4 * min(1.0, (mid - v) / (mid - lo + 1e-12))
    return f"background:rgba(217,0,18,{a:.2f})"


def main() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "frontier_report.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    data = json.loads((OUT / "frontier.json").read_text(encoding="utf-8"))
    fams, ds = data["families"], data["datasets"]
    names = [d["dataset"] for d in ds]

    fig = go.Figure()
    for f in fams:
        vals = [d["families"][f]["r2_mean"] for d in ds]
        fig.add_bar(name=f, x=names, y=vals, marker_color=COLORS[f], text=[f"{v:.2f}" for v in vals],
                    textposition="outside", error_y=dict(type="data", array=[d["families"][f]["r2_sd"] for d in ds]))
    fig.update_layout(barmode="group", yaxis_title="held-out R^2 against the real labels (3-fold mean)",
                      height=480, template="plotly_white", legend=dict(orientation="h", y=-0.25),
                      margin=dict(l=60, r=20, t=30, b=120))
    fig_r2 = pio.to_html(fig, full_html=False, include_plotlyjs="inline", config={"displaylogo": False})

    # share of the unconstrained model's R^2 that each tractable family keeps
    share = {f: [d["families"][f]["r2_mean"] / d["families"]["full boosted"]["r2_mean"] for d in ds] for f in fams[:-1]}
    fig2 = go.Figure()
    for f in fams[:-1]:
        fig2.add_bar(name=f, y=names, x=share[f], orientation="h", marker_color=COLORS[f],
                     text=[f"{v:.2f}" for v in share[f]], textposition="outside")
    fig2.update_layout(barmode="group", xaxis_title="R^2 of the family / R^2 of full boosting",
                       height=640, template="plotly_white", legend=dict(orientation="h", y=-0.12),
                       margin=dict(l=170, r=60, t=30, b=80))
    fig_share = pio.to_html(fig2, full_html=False, include_plotlyjs=False, config={"displaylogo": False})

    all_r2 = [d["families"][f]["r2_mean"] for d in ds for f in fams]
    lo, mid, hi = float(np.min(all_r2)), float(np.median(all_r2)), float(np.max(all_r2))
    rows = []
    for d in ds:
        cells = "".join(f'<td class="num" style="{heat(d["families"][f]["r2_mean"], lo, mid, hi)}">'
                        f'{d["families"][f]["r2_mean"]:.3f}</td>' for f in fams)
        inter = d["families"]["pairwise boosted"]["r2_mean"] - d["families"]["additive boosted"]["r2_mean"]
        higher = d["families"]["full boosted"]["r2_mean"] - d["families"]["pairwise boosted"]["r2_mean"]
        rows.append(f"<tr><td>{html.escape(d['dataset'])}</td><td class='num'>{d['n']}</td><td class='num'>{d['d']}</td>"
                    f"{cells}<td class='num'>{inter:+.3f}</td><td class='num'>{higher:+.3f}</td></tr>")
    head = "".join(f"<th>{html.escape(f)}</th>" for f in fams)
    tbl = (f"<div class='tw'><table><colgroup><col style='width:19%'><col style='width:6%'><col style='width:4%'>"
           + "<col style='width:9.5%'>" * len(fams) + "<col style='width:12%'><col style='width:12%'></colgroup>"
           f"<thead><tr><th>dataset</th><th>n</th><th>d</th>{head}<th>pairs add</th><th>higher order adds</th></tr>"
           f"</thead><tbody>{''.join(rows)}</tbody></table></div>")

    crow = []
    for d in ds:
        a, p = d["families"]["additive boosted"], d["families"]["pairwise boosted"]
        crow.append(f"<tr><td>{html.escape(d['dataset'])}</td><td class='num'>{d['d']}</td>"
                    f"<td class='num'>{1000 * a['exact_shap_seconds_per_point']:.1f}</td>"
                    f"<td class='num'>{1000 * p['exact_shap_seconds_per_point']:.1f}</td>"
                    f"<td class='num'>{p['efficiency_gap']:.1e}</td>"
                    f"<td class='num'>{d['pairwise_max_third_order_moebius']:.1e}</td>"
                    f"<td class='num'>{d.get('pairwise_vs_shap_exact_max_abs', float('nan')):.1e}</td></tr>")
    ctbl = ("<div class='tw'><table><colgroup><col style='width:22%'><col style='width:6%'><col style='width:14%'>"
            "<col style='width:14%'><col style='width:14%'><col style='width:15%'><col style='width:15%'></colgroup>"
            "<thead><tr><th>dataset</th><th>d</th><th>additive: ms per point</th><th>pairwise: ms per point</th>"
            "<th>pairwise: efficiency gap</th><th>max |3rd-order Moebius|</th><th>vs shap Exact (d &le; 10)</th></tr>"
            f"</thead><tbody>{''.join(crow)}</tbody></table></div>")

    med = {f: float(np.median([d["families"][f]["r2_mean"] / d["families"]["full boosted"]["r2_mean"] for d in ds]))
           for f in fams[:-1]}
    secs = [("results", "Held-out R^2 by target family",
             f"<p>Each family is fitted to the real labels and scored on held-out folds against the same real labels. "
             f"Median share of full boosting's R^2 kept: "
             + ", ".join(f"{html.escape(f)} {v:.2f}" for f, v in med.items()) + ".</p>"
             + fig_r2 + "<h3>Table (green above, red below the median of all cells)</h3>" + tbl
             + "<p>'pairs add' = R^2(pairwise) - R^2(additive boosted); 'higher order adds' = R^2(full) - R^2(pairwise)."
               " Both are differences of held-out scores, so small negative values are noise.</p>"),
            ("share", "Share of the unconstrained signal each family keeps", fig_share),
            ("cost", "Exact SHAP: cost and checks",
             "<p>Exact interventional SHAP on a 200-row background, per explained point. The pairwise formula "
             "uses coalitions of size at most two; its correctness needs every third-order Moebius term of the "
             "fitted model to vanish (checked on 5 random triples) and was checked against shap's exact "
             "enumeration where 2^d is affordable. Efficiency gap = |sum of attributions - (f(x) - E f)|.</p>" + ctbl),
            ("setup", "Setup", f"<pre>{html.escape(json.dumps({k: data[k] for k in ('args', 'versions', 'openml_ids', 'runtime_s', 'created')}, indent=1))}</pre>")]
    toc = "".join(f"<li><a href='#{s}'>{html.escape(t)}</a></li>" for s, t, _ in secs)
    body = "".join(f"<details open id='{s}'><summary><h2>{html.escape(t)}</h2></summary>{b}</details>" for s, t, b in secs)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Realism-tractability frontier</title><style>{CSS}</style>"
            f"</head><body><main><h1>Realism-tractability frontier</h1><p>How much of a real regression task can a "
            f"target family with exact explanations reproduce? {len(ds)} data sets, {len(fams)} families "
            f"(frontier_probe.py, {html.escape(data['created'])}).</p><nav class='toc'><b>Contents</b><ol>{toc}</ol></nav>"
            f"{body}</main></body></html>")
    out = OUT / "frontier_report.html"
    out.write_text(page, encoding="utf-8")
    logging.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
