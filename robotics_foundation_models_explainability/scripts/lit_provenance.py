#!/usr/bin/env python3
"""Count where the included works came from, for the methods section and the PRISMA counts.

For each work in lit_review/included.json: which retrieval routes found it (arXiv keyword search,
Crossref, web sweep, PDF reference lists) and what the rule screen said about it (pass, or rescued
by hand from a fail). Writes lit_review/data/processed/provenance.json.

Usage (from robotics_foundation_models_explainability/), ~1 s, offline:
    .venv/Scripts/python.exe scripts/lit_provenance.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "lit_review" / "data" / "processed"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", stream=sys.stdout)
    dedup = json.loads((PROC / "deduplicated.json").read_text(encoding="utf-8"))
    by_id = {r["arxiv_id"]: r for r in dedup if r["arxiv_id"]}
    by_title = {r["title"].lower(): r for r in dedup}
    works = [w for w in json.loads((ROOT / "lit_review" / "included.json").read_text(encoding="utf-8"))["works"]
             if w["tier"] != "background"]  # background references are cited for context, not screened
    rows, routes, screens = [], Counter(), Counter()
    for w in works:
        rec = by_id.get(w["arxiv_id"]) if w["arxiv_id"] else by_title.get(w["title"].lower())
        if rec is None:
            raise KeyError(f"included work not found in deduplicated records: {w['key']} {w['title'][:60]}")
        srcs = set(rec["sources"])
        route = "+".join(sorted(srcs))
        screen = rec["screen"].split(":")[0] if "web-sweep" not in srcs else "web"
        if screen == "fail":
            screen = f"rescued ({rec['screen'][6:]})"
        routes[route] += 1
        screens[f"{w['tier']}: {screen}"] += 1
        rows.append({"key": w["key"], "tier": w["tier"], "sources": sorted(srcs), "screen": rec["screen"]})
    only_web = sum(1 for r in rows if r["sources"] == ["web-sweep"])
    only_refs = sum(1 for r in rows if r["sources"] == ["pdf-references"])
    summary = {"included": len(works), "by_route": dict(routes), "by_screen": dict(screens),
               "found_only_by_web_sweep": only_web, "found_only_by_reference_lists": only_refs}
    (PROC / "provenance.json").write_text(json.dumps({"summary": summary, "works": rows}, indent=1), encoding="utf-8")
    logging.info(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
