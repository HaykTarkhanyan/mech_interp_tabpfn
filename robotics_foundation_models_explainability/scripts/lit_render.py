#!/usr/bin/env python3
"""Render the review to Markdown and a standalone HTML page.

Citation markers in lit_review/review_source.md (keys are arXiv ids, or w_* slugs for workshop items):
  [@2509.00328]                 -> (Haon et al., 2025)
  [@2509.00328; @2603.19183]    -> (Haon et al., 2025; Smith et al., 2026)
  @2509.00328                   -> Haon et al. (2025)          (narrative form)
Fails loudly on an unknown key and on em/en dashes or curly quotes (house style).

Outputs (lit_review/):
  robotics_fm_explainability_review.md    citations resolved, reference list appended
  robotics_fm_explainability_review.html  standalone: inline CSS, inline SVG charts, collapsible
                                          sections, table of contents, paper table built from notes/

Usage (from robotics_foundation_models_explainability/), ~3 s (pandoc):
    .venv/Scripts/python.exe scripts/lit_render.py
"""

import html
import json
import logging
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "lit_review"
OUT = REVIEW / "robotics_fm_explainability_review"
KEY_RE = r"(?:\d{4}\.\d{4,5}|w_[a-z0-9_]+)"
BAD_CHARS = {"—": "em dash", "–": "en dash", "“": "curly quote", "”": "curly quote",
             "‘": "curly quote", "’": "curly quote"}
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
# First "authors" that are organizations: cited by name alone, e.g. (NVIDIA, 2025).
ORG_AUTHORS = {"Physical Intelligence", "NVIDIA", "Octo Model Team", "Open X-Embodiment Collaboration"}


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / "lit_render.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def surname(author: str) -> str:
    parts = author.strip().split()
    return parts[-1] if parts else ""


def cite_name(w: Dict) -> str:
    authors = [a.strip() for a in (w.get("authors") or "").split(", ") if a.strip() not in ("", ":")]
    if authors and authors[0] in ORG_AUTHORS:
        return authors[0]
    if not authors:
        short = " ".join(w["title"].split(":")[0].split()[:5])
        return f"\"{short}\""
    if len(authors) == 1:
        return surname(authors[0])
    if len(authors) == 2:
        return f"{surname(authors[0])} and {surname(authors[1])}"
    return f"{surname(authors[0])} et al."


def resolve(text: str, works: Dict[str, Dict], cited: Counter) -> str:
    def one(key: str, narrative: bool) -> str:
        if key not in works:
            raise KeyError(f"unknown citation key @{key}")
        cited[key] += 1
        w = works[key]
        return (f"{cite_name(w)} ([{w['year']}](#ref-{key}))" if narrative
                else f"[{cite_name(w)}, {w['year']}](#ref-{key})")

    def group(m: re.Match) -> str:
        keys = re.findall(rf"@({KEY_RE})", m.group(1))
        return "(" + "; ".join(one(k, False) for k in keys) + ")"

    text = re.sub(rf"\[((?:\s*@{KEY_RE}\s*;?)+)\]", group, text)
    return re.sub(rf"(?<![\w(\[])@({KEY_RE})", lambda m: one(m.group(1), True), text)


def read_level(w: Dict, fetch: Dict) -> str:
    """How much of a work was read, for works without an arXiv full text (papers/workshop/fetch_status.json)."""
    if w["arxiv_id"] or w["tier"] == "background":
        return ""
    st = fetch.get(w["key"], {})
    if st.get("abstract_found") or (st.get("page_is_work") and st.get("chars", 0) > 3000):
        return " Read: abstract or page text only, not the full paper."
    if not w.get("verified", True):
        return " Read: one-line listing summary only; the page blocks automated access and the title is unverified."
    return " Read: one-line listing summary only."


def reference_md(works: List[Dict], sections: Dict[str, str], cited: Counter) -> str:
    fetch_path = ROOT / "papers" / "workshop" / "fetch_status.json"
    fetch = json.loads(fetch_path.read_text(encoding="utf-8")) if fetch_path.exists() else {}
    lines = ["## Included works", "",
             f"All {len(works)} included works, grouped by review section. Core works are marked **core**. "
             "Works with an arXiv id were read in full (core) or from the abstract (adjacent, unless downloaded). "
             "Workshop items without an arXiv id link to the page they were found on, and say how much was read.", ""]
    for sec, title in sections.items():
        group = [w for w in works if w["section"] == sec]
        if not group:
            continue
        lines += [f"### {title}", ""]
        for w in sorted(group, key=lambda w: (w["tier"] != "core", str(w["year"]), w["title"])):
            ident = f"arXiv:{w['arxiv_id']}" if w["arxiv_id"] else (w.get("venue") or "web")
            authors = w.get("authors") or "authors not recorded"
            if authors.count(",") >= 3:
                authors = authors.split(",")[0] + " et al."
            tier = " **core**" if w["tier"] == "core" else ""
            lines.append(f"- <a id=\"ref-{w['key']}\"></a>{authors} ({w['year']}). [{w['title']}]({w['url']}). "
                         f"{ident}.{tier}{read_level(w, fetch)}")
        lines.append("")
    return "\n".join(lines)


def svg_hbar(rows: List[tuple], title: str, colors: List[str], legend: List[str]) -> str:
    """Horizontal stacked bars, value labels at the end of each bar, right padding for labels."""
    label_w, bar_w, row_h, pad_r = 300, 360, 24, 60
    width, height = label_w + bar_w + pad_r, 40 + row_h * len(rows) + 30
    vmax = max(sum(r[1]) for r in rows) or 1
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}" class="chart">',
           f'<text x="0" y="16" class="ct">{html.escape(title)}</text>']
    for i, (label, vals) in enumerate(rows):
        y = 34 + i * row_h
        out.append(f'<text x="{label_w - 8}" y="{y + 15}" text-anchor="end" class="cl">{html.escape(label)}</text>')
        x = label_w
        for v, c in zip(vals, colors):
            w = bar_w * v / vmax
            if v:
                out.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{w:.1f}" height="{row_h - 8}" fill="{c}"/>')
            x += w
        out.append(f'<text x="{x + 6:.1f}" y="{y + 15}" class="cv">{" + ".join(str(v) for v in vals if v) if len(vals) > 1 else vals[0]}</text>')
    ly = 34 + len(rows) * row_h + 14
    for j, (c, name) in enumerate(zip(colors, legend)):
        out.append(f'<rect x="{label_w + j * 120}" y="{ly - 10}" width="12" height="12" fill="{c}"/>'
                   f'<text x="{label_w + j * 120 + 18}" y="{ly}" class="cl">{html.escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def charts(works: List[Dict], sections: Dict[str, str]) -> str:
    rows = []
    for sec, title in sections.items():
        core = sum(w["section"] == sec and w["tier"] == "core" for w in works)
        adj = sum(w["section"] == sec and w["tier"] == "adjacent" for w in works)
        if core or adj:
            rows.append((title, [core, adj]))
    by_section = svg_hbar(rows, "Included works per section", [BLUE, ORANGE], ["core", "adjacent"])
    quarters = Counter()
    for w in works:
        if w["tier"] == "core" and re.fullmatch(r"\d{4}\.\d{4,5}", w["arxiv_id"] or ""):
            yy, mm = int(w["arxiv_id"][:2]), int(w["arxiv_id"][2:4])
            quarters[f"20{yy} Q{(mm - 1) // 3 + 1}"] += 1
    qrows = [(q, [quarters[q]]) for q in sorted(quarters)]
    by_quarter = svg_hbar(qrows, "Core papers with an arXiv id, by quarter of first submission", [RED], ["core papers"])
    return f'<div class="charts">{by_section}{by_quarter}</div>'


def notes_table(works: List[Dict]) -> str:
    notes_dir = REVIEW / "notes"
    rows = []
    for w in works:
        p = notes_dir / f"{w['arxiv_id']}.json"
        if w["tier"] != "core" or not w["arxiv_id"] or not p.exists():
            continue
        n = json.loads(p.read_text(encoding="utf-8"))
        ev = n.get("evidence") or {}
        cells = [str(w["year"]), ", ".join(n.get("models") or []), str(n.get("action_head", "")),
                 ", ".join(n.get("methods") or []), str(ev.get("setting", "")), str(n.get("closed_loop", ""))[:140],
                 str(n.get("causal_validation", ""))[:160]]
        first = f'<td><a href="#ref-{w["key"]}">{html.escape(str(n.get("first_author", "")))}</a></td>'
        rows.append("<tr>" + first + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    if not rows:
        return ""
    head = ("<table class=\"pt\"><colgroup><col style=\"width:8%\"><col style=\"width:5%\"><col style=\"width:13%\">"
            "<col style=\"width:11%\"><col style=\"width:15%\"><col style=\"width:8%\"><col style=\"width:18%\">"
            "<col style=\"width:22%\"></colgroup><thead><tr><th>First author</th><th>Year</th><th>Models</th>"
            "<th>Action head</th><th>Methods</th><th>Setting</th><th>Closed loop?</th><th>Causal validation</th>"
            "</tr></thead><tbody>")
    return head + "\n".join(rows) + "</tbody></table>"


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#5f6368;--card:#f6f7f9;--line:#d9dce1;--link:#0033A0;--accent:#D90012}
@media (prefers-color-scheme:dark){:root{--bg:#141518;--fg:#e8e8ea;--muted:#a0a4ab;--card:#1d1f23;--line:#33363c;--link:#8fb0ff;--accent:#ff5a66}}
html{background:var(--bg)}body{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1050px;margin:0 auto;padding:24px 16px 80px}
a{color:var(--link)}h1{font-size:1.9rem;line-height:1.25;margin:.2em 0}.sub{color:var(--muted);margin-bottom:1.2em}
details{border:1px solid var(--line);border-radius:8px;margin:14px 0;padding:0 16px;background:var(--bg)}
details>summary{cursor:pointer;list-style:none;padding:10px 0}details>summary::-webkit-details-marker{display:none}
details>summary h2{display:inline;font-size:1.35rem;margin:0}details>summary:before{content:"\\25B8  ";color:var(--muted)}
details[open]>summary:before{content:"\\25BE  "}h3{font-size:1.1rem;margin-top:1.4em}
nav.toc{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 18px}nav.toc ol{margin:.3em 0;padding-left:1.4em}
blockquote{border-left:4px solid var(--accent);margin:1em 0;padding:.3em 1em;background:var(--card)}
code{background:var(--card);padding:1px 4px;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:1em 0}th,td{border:1px solid var(--line);padding:5px 7px;vertical-align:top;text-align:left;overflow-wrap:anywhere}
th{background:var(--card)}table.pt{table-layout:fixed}
.charts{display:grid;gap:18px;margin:1em 0}svg.chart{width:100%;max-width:760px;height:auto}
svg .ct{font:600 14px system-ui;fill:var(--fg)}svg .cl{font:12px system-ui;fill:var(--fg)}svg .cv{font:600 12px system-ui;fill:var(--fg)}
.wrap{overflow-x:auto}
"""


def main() -> None:
    setup_logging()
    start = time.time()
    inc = json.loads((REVIEW / "included.json").read_text(encoding="utf-8"))
    works_list, sections = inc["works"], inc["sections"]
    works = {w["key"]: w for w in works_list}
    src = (REVIEW / "review_source.md").read_text(encoding="utf-8")
    bad = [(name, src.count(ch)) for ch, name in BAD_CHARS.items() if ch in src]
    if bad:
        raise ValueError(f"house style: review_source.md contains {bad}")
    cited: Counter = Counter()
    body = resolve(src, works, cited)
    uncited_core = [w["key"] for w in works_list if w["tier"] == "core" and not cited[w["key"]]]
    if uncited_core:
        logging.warning(f"{len(uncited_core)} core works are never cited in the text: {uncited_core}")
    full_md = body.rstrip() + "\n\n" + reference_md(works_list, sections, cited)
    OUT.with_suffix(".md").write_text(full_md, encoding="utf-8")

    # --wrap=none: pandoc otherwise breaks long <section> tags across lines and the regexes below miss them.
    frag = subprocess.run(["pandoc", "-f", "markdown", "-t", "html", "--section-divs", "--wrap=none"], input=full_md,
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout
    title_m = re.search(r"<h1[^>]*>(.*?)</h1>", frag, re.S)
    title = re.sub(r"<[^>]+>", "", title_m.group(1)) if title_m else "Review"
    frag = re.sub(r"<section[^>]*class=\"level1\"[^>]*>\s*<h1[^>]*>.*?</h1>", "", frag, count=1, flags=re.S)
    toc = []

    def to_details(m: re.Match) -> str:
        sid, h = m.group(1), m.group(2)
        toc.append((sid, re.sub(r"<[^>]+>", "", h)))
        return f'<details open id="{sid}"><summary><h2>{h}</h2></summary>'

    frag = re.sub(r'<section id="([^"]+)" class="level2">\s*<h2>(.*?)</h2>', to_details, frag, flags=re.S)
    # level2 sections close with </section>; level3 sections keep their own. Close details at level2 ends.
    out, depth, stack = [], 0, []
    for tok in re.split(r"(<details open[^>]*>|<section[^>]*>|</section>)", frag):
        if tok.startswith("<details open"):
            stack.append("details")
            out.append(tok)
        elif tok.startswith("<section"):
            stack.append("section")
            out.append(tok)
        elif tok == "</section>":
            if not stack:
                continue  # closes the level-1 section whose opening tag was stripped with the h1
            out.append("</details>" if stack.pop() == "details" else "</section>")
        else:
            out.append(tok)
    frag = "".join(out)
    frag = frag.replace("<!--CHARTS-->", charts(works_list, sections))
    frag = frag.replace("<!--PAPER_TABLE-->", f'<div class="wrap">{notes_table(works_list)}</div>')
    toc_html = "<nav class=\"toc\"><strong>Contents</strong><ol>" + "".join(
        f'<li><a href="#{sid}">{html.escape(t)}</a></li>' for sid, t in toc) + "</ol></nav>"
    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><h1>{title}</h1>"
            f"<div class=\"sub\">Literature review, built {time.strftime('%Y-%m-%d')}. "
            f"{len(works_list)} included works ({sum(w['tier'] == 'core' for w in works_list)} core).</div>"
            f"{toc_html}{frag}</body></html>")
    OUT.with_suffix(".html").write_text(page, encoding="utf-8")
    logging.info(f"Rendered {OUT.with_suffix('.md').name} and .html ({len(page) / 1e3:.0f} kB), "
                 f"{sum(cited.values())} citations of {len(cited)} works, in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
