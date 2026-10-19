#!/usr/bin/env python3
"""Backward citation chaining from the reference lists of the downloaded core papers.

Semantic Scholar's keyless pool 429'd every chaining call on 2026-10-19 and OpenAlex lists 0
references for the main seed, so the references are read from the PDFs themselves: every arXiv id
in the text after the last "References"/"Bibliography" heading of papers/text/<id>.txt is collected,
counted across papers, and resolved to title + abstract through the arXiv API.

Limitation (recorded in the output): a reference without an arXiv id in its text (a conference
paper cited by venue only) is invisible to this method.

Outputs (lit_review/data/raw/):
  pdf_reference_counts.json   cited id -> list of core papers citing it, plus per-paper stats
  pdf_reference_records.json  resolved records, same shape as combined_raw.json, for lit_screen.py

Usage (from robotics_foundation_models_explainability/), ~1-2 min (one arXiv API call per 50 ids):
    .venv/Scripts/python.exe scripts/lit_refchain.py
"""

import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lit_search import arxiv_by_ids  # noqa: E402  (same retrying arXiv client)

ROOT = Path(__file__).resolve().parents[1]
TEXTDIR = ROOT / "papers" / "text"
RAW = ROOT / "lit_review" / "data" / "raw"
ID_RE = re.compile(r"(?:arxiv(?:\s*preprint)?\s*(?:abs/)?\s*:?\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})", re.I)
REF_HEAD = re.compile(r"^\s*(references|bibliography|reference)\s*$", re.I | re.M)


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / "lit_refchain.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def main() -> None:
    setup_logging()
    start = time.time()
    cited_by = defaultdict(set)
    per_paper = {}
    texts = sorted(TEXTDIR.glob("*.txt"))
    if not texts:
        raise FileNotFoundError(f"no extracted texts in {TEXTDIR}; run download_papers.py first")
    for path in texts:
        src = path.stem
        text = path.read_text(encoding="utf-8", errors="replace")
        heads = list(REF_HEAD.finditer(text))
        if not heads:
            logging.error(f"{src}: no References heading found; scanning the whole text instead")
        body = text[heads[-1].start():] if heads else text
        ids = {i for i in ID_RE.findall(body) if i != src}
        per_paper[src] = {"reference_heading_found": bool(heads), "arxiv_ids_found": len(ids)}
        for i in ids:
            cited_by[i].add(src)
    logging.info(f"{len(texts)} papers, {len(cited_by)} distinct cited arXiv ids")

    meta = arxiv_by_ids(sorted(cited_by))
    unresolved = sorted(set(cited_by) - set(meta))
    if unresolved:
        logging.error(f"{len(unresolved)} cited ids not returned by the arXiv API: {unresolved[:20]}")
    records = []
    for aid, rec in meta.items():
        rec = dict(rec, source="pdf-references", theme=f"refs:{len(cited_by[aid])}",
                   cited_by_core=sorted(cited_by[aid]))
        records.append(rec)
    records.sort(key=lambda r: -len(r["cited_by_core"]))
    (RAW / "pdf_reference_records.json").write_text(json.dumps(records, indent=1, ensure_ascii=False),
                                                    encoding="utf-8")
    (RAW / "pdf_reference_counts.json").write_text(json.dumps({
        "method": "arXiv ids regex-matched after the last References heading of each downloaded paper",
        "limitation": "references without an arXiv id in their text are not captured",
        "papers": per_paper,
        "cited_ids": {aid: {"title": (meta.get(aid) or {}).get("title", ""), "cited_by": sorted(s)}
                      for aid, s in sorted(cited_by.items(), key=lambda kv: -len(kv[1]))},
        "unresolved": unresolved,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    logging.info(f"Wrote {len(records)} reference records in {time.time() - start:.0f}s; top cited:")
    for r in records[:15]:
        logging.info(f"  {len(r['cited_by_core']):3d}x {r['arxiv_id']} {r['title'][:80]}")


if __name__ == "__main__":
    main()
