#!/usr/bin/env python3
"""Draw the PRISMA flow diagram from the pipeline's own count files; no number is typed in.

Reads data/raw/search_counts.json, data/processed/prisma_counts.json, shortlist.json, screened_all.json,
references.json and verification_report.json, and writes lit_review/figures/prisma_flow.png. A cited work counts
as "included through screening" only if it is in the manually screened shortlist; the others are split into not
retrieved, failed the automated screen, and passed it but ranked below the shortlist cut.

Usage (from synthetic_data/, ~3 s):
    .venv/Scripts/python.exe scripts/lit_prisma.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from lit_references import MATCH, similarity

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "lit_review"
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def box(ax, x, y, w, h, text, color, fill=False, size=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02", linewidth=1.3,
                                edgecolor=color, facecolor=color if fill else "white"))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color="white" if fill else "black",
            linespacing=1.45)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color="#555555"))


def main() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "lit_prisma.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    search = load(REVIEW / "data" / "raw" / "search_counts.json")
    counts = load(REVIEW / "data" / "processed" / "prisma_counts.json")
    refs = load(REVIEW / "references.json")
    verify = load(REVIEW / "verification_report.json")

    raw_db = {}
    for row in search["per_theme"]:
        for k, v in row.items():
            if k != "theme":
                raw_db[k] = raw_db.get(k, 0) + v
    shortlist = load(REVIEW / "data" / "processed" / "shortlist.json")
    corpus = load(REVIEW / "data" / "processed" / "screened_all.json")
    passed = [r for r in corpus if r["passes"]]

    def hit(ref, pool):
        return any(similarity(ref["title"], r.get("title", "")) >= MATCH for r in pool)

    route = {ref["key"]: ("screened" if hit(ref, shortlist) else "below_cut" if hit(ref, passed)
                          else "failed_screen" if hit(ref, corpus) else "not_retrieved") for ref in refs}
    n = {k: sum(1 for v in route.values() if v == k) for k in ("screened", "below_cut", "failed_screen", "not_retrieved")}
    logging.info(f"routes into the review: {n}")
    seeds_ok = sum(1 for s in search["seeds"].values() if s["resolved"])

    fig, ax = plt.subplots(figsize=(9.6, 8.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 2.7, 9.2, 4.6, 1.15,
        f"Keyword search ({len(search['per_theme'])} themes)\n{search['keyword_records']:,} records\n"
        + " · ".join(f"{k} {v}" for k, v in raw_db.items()), BLUE, fill=True)
    box(ax, 7.5, 9.2, 4.4, 1.15,
        f"Citation chaining (OpenAlex)\n{seeds_ok}/{len(search['seeds'])} seed papers resolved\n"
        f"{search['citing_records']:,} citing records", BLUE, fill=True)
    box(ax, 5.0, 7.45, 6.4, 0.95,
        f"Identification: {counts['identification']:,} records\n"
        f"{counts['duplicates_merged']:,} duplicates merged, {counts['dropped_no_title']} dropped without a title "
        f"-> {counts['after_deduplication']:,} unique", RED)
    box(ax, 5.0, 5.85, 6.4, 1.05,
        f"Automated relevance screen (6 rules)\n{counts['excluded_by_screen']:,} excluded "
        f"({counts['excluded_domain_title']} on application-domain titles)\n"
        f"{counts['after_relevance_screen']:,} passed, {counts['passed_only_via_citation_chaining']} found only by chaining",
        RED, size=8.5)
    box(ax, 5.0, 4.25, 6.4, 0.95,
        f"Manual title / abstract / full-text screening\nof the top {counts['shortlist_size']} "
        f"-> {n['screened']} included through screening", RED)
    box(ax, 8.35, 2.5, 2.8, 1.55,
        f"Added outside screening\n(targeted lookup or\nbackward chaining): {len(refs) - n['screened']}\n"
        f"{n['not_retrieved']} not retrieved,\n{n['failed_screen']} failed the automated screen,\n"
        f"{n['below_cut']} ranked below the cut", ORANGE, size=7.8)
    box(ax, 4.2, 2.55, 4.6, 0.95,
        f"{len(refs)} references\n{verify['verified']}/{verify['total']} verified against registries, "
        f"{verify['title_mismatches']} title mismatches", BLUE, fill=True, size=8.5)
    arrow(ax, 2.7, 8.6, 4.0, 7.95); arrow(ax, 7.5, 8.6, 6.0, 7.95)
    arrow(ax, 5.0, 6.95, 5.0, 6.4); arrow(ax, 5.0, 5.3, 5.0, 4.75); arrow(ax, 4.6, 3.75, 4.3, 3.05)
    arrow(ax, 6.9, 2.55, 6.55, 2.55)
    unavailable = "; ".join(f"{k} ({v})" for k, v in search["databases_unavailable"].items())
    ax.text(5.0, 1.35, f"Search date {search['search_date']}. Unavailable: {unavailable}.",
            ha="center", fontsize=8, style="italic", color="#444444", wrap=True)
    out = REVIEW / "figures" / "prisma_flow.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    logging.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
