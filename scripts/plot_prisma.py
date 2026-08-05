#!/usr/bin/env python3
"""Render the PRISMA flow diagram from data/processed/prisma_counts.json.

Every number in the figure is read from the counts file written by
scripts/screen_papers.py - none is typed in. That is the whole point: a PRISMA
diagram's content IS its record counts, so the figure must be generated from the
same artefact the text cites (DECISIONS.md #4).

Usage:
    python scripts/plot_prisma.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def box(ax, x, y, w, h, text, facecolor, textcolor="black", fontsize=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02", linewidth=1.2,
                                edgecolor=facecolor, facecolor=facecolor, alpha=0.16
                                if facecolor != BLUE else 1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.1, color="#555555"))


def main() -> None:
    setup_logging(Path("logs") / "plot_prisma.log")
    counts = json.loads(Path("data/processed/prisma_counts.json").read_text(encoding="utf-8"))
    refs = json.loads(Path("lit_review/references.json").read_text(encoding="utf-8"))

    # The identification box must show RAW retrieval per database. counts[
    # "database_contribution_unique"] is the post-deduplication contribution and is a
    # different quantity - labelling it "records retrieved" understates the raw sweep.
    search = json.loads(Path("data/raw/search_counts.json").read_text(encoding="utf-8"))
    raw_per_db: dict = {}
    for theme in search["per_theme"]:
        for name, n in theme.items():
            if name != "theme":
                raw_per_db[name] = raw_per_db.get(name, 0) + n

    db = counts["database_contribution_unique"]
    n_corpus = sum(1 for r in refs if r["provenance"] == "corpus")
    n_arxiv = sum(1 for r in refs if r["provenance"] == "arxiv")
    n_web = sum(1 for r in refs if r["provenance"] == "web-not-indexed")

    fig, ax = plt.subplots(figsize=(9.0, 8.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    box(ax, 3.4, 9.3, 6.0, 1.0,
        f"Identification\n{counts['identification']:,} records retrieved\n"
        f"OpenAlex {raw_per_db.get('OpenAlex', 0)} · Crossref {raw_per_db.get('Crossref', 0)} · "
        f"DBLP {raw_per_db.get('DBLP', 0)}",
        BLUE, "white", 9)
    box(ax, 3.4, 7.6, 6.0, 0.9,
        f"Deduplication\n{counts['duplicates_merged']} duplicate records merged\n"
        f"{counts['after_deduplication']:,} unique records\n"
        f"(unique contribution: OpenAlex {db.get('OpenAlex', 0)} · "
        f"Crossref {db.get('Crossref', 0)} · DBLP {db.get('DBLP', 0)})", RED, fontsize=8)
    box(ax, 3.4, 5.9, 6.0, 0.9,
        f"Automated relevance screen\n{counts['excluded_total']:,} excluded "
        f"({counts['excluded_out_of_scope_domain']} on out-of-scope domain)\n"
        f"{counts['after_relevance_screen']} passed", RED)
    box(ax, 3.4, 4.2, 6.0, 0.8,
        f"Manual title / abstract screening\ntop {counts['shortlist_size']} shortlisted "
        f"→ {n_corpus} selected", RED)
    box(ax, 8.1, 4.2, 3.4, 1.0,
        f"Targeted supplementary\nretrieval\n{n_arxiv} arXiv (title-verified)\n"
        f"{n_web} not indexed anywhere", ORANGE, fontsize=8)
    box(ax, 5.0, 2.2, 5.4, 0.9,
        f"{len(refs)} references included\n{len(refs)}/{len(refs)} verified against registries",
        BLUE, "white")

    for y1, y2 in [(8.8, 8.1), (7.15, 6.4), (5.45, 4.65)]:
        arrow(ax, 3.4, y1, 3.4, y2)
    arrow(ax, 3.4, 3.8, 4.4, 2.7)
    arrow(ax, 8.1, 3.7, 5.9, 2.7)

    ax.text(5.0, 0.75, f"Search date {counts.get('search_date', '2026-08-05')} · "
                       f"cross-database overlap: "
                       f"{counts['found_by_n_databases'].get('1', 0):,} records found by one "
                       f"database only, {counts['found_by_n_databases'].get('3', 0)} by all three",
            ha="center", fontsize=8, style="italic", color="#444444")

    out = Path("lit_review/figures/prisma_flow.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    logging.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
