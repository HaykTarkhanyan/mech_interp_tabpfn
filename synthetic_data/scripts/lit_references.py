#!/usr/bin/env python3
"""Build the review's reference list from lit_review/included.json, resolving every entry from a source.

Each entry in included.json names a paper by title and, optionally, by arXiv ID or DOI. Resolution
order, all title-checked (a resolving identifier for the WRONG paper is a hard failure, see
mech_interp/_learnings/2026-08-06-0040_verify-citations-against-registry-title.md):
  corpus - the screened search results (data/processed/screened_all.json), matched by title
  arxiv  - arxiv.org/abs/<id> citation_* meta tags (the export API is not needed)
  doi    - Crossref /works/<doi> for journal and proceedings DOIs
Authors, title, year and venue come from the resolving source. The one exception: when the source only
holds a preprint of a paper published later, included.json gives the published venue and its year.

Usage (from synthetic_data/, ~1-2 min for ~60 entries because of polite delays):
    .venv/Scripts/python.exe scripts/lit_references.py
"""

import html
import json
import logging
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "lit_review"
USER_AGENT = "synthetic-data-xai-lit-review/1.0 (mailto:tarkhanyan02@gmail.com)"
MATCH = 0.85
DELAY_S = 1.5


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / "lit_references.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", (t or "").lower())
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in t if not unicodedata.combining(c)))


def similarity(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    r = SequenceMatcher(None, na, nb).ratio()
    if min(len(na), len(nb)) >= 25 and (na in nb or nb in na):  # subtitle differences
        r = max(r, 0.95)
    return r


def surname(name: str) -> str:
    name = re.sub(r"\s+\d{4}$", "", name.strip())  # DBLP homonym suffix
    s = name.split(",")[0] if "," in name else (name.split() or ["anon"])[-1]
    s = "".join(c for c in unicodedata.normalize("NFKD", s.replace("ß", "ss")) if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]", "", s).lower() or "anon"


def get(url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
    for attempt in range(1, 5):
        r = requests.get(url, params=params or {}, timeout=45, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * 2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"{url} failed repeatedly")


def from_corpus(title: str, corpus: List[Dict]) -> Optional[Dict]:
    best = max(corpus, key=lambda r: similarity(title, r.get("title", "")))
    ratio = similarity(title, best.get("title", ""))
    if ratio < MATCH:
        return None
    return {"title": best["title"], "authors": [a.strip() for a in (best.get("authors") or "").split(",") if a.strip()],
            "year": best.get("year"), "venue": best.get("venue", ""), "doi": best.get("doi", ""),
            "url": best.get("url", ""), "citations": best.get("citations"), "provenance": "corpus",
            "found_by": best.get("found_by", []), "match_ratio": round(ratio, 3)}


def from_arxiv(arxiv_id: str, title: str) -> Dict:
    r = get(f"https://arxiv.org/abs/{arxiv_id}")
    if r is None:
        raise LookupError(f"arXiv {arxiv_id}: 404")
    # meta tag contents carry HTML entities ("Practitioner&#39;s"); unescape them here, not only when rendering
    meta = lambda k: [html.unescape(v) for v in re.findall(rf'name=["\']citation_{k}["\'][^>]+content=["\']([^"\']+)', r.text, re.I)]
    got = (meta("title") or [""])[0]
    ratio = similarity(title, got)
    if ratio < MATCH:
        raise LookupError(f"arXiv {arxiv_id}: TITLE MISMATCH ({ratio:.2f}) expected {title!r}, got {got!r}")
    return {"title": got, "authors": meta("author"), "year": (meta("date") or [""])[0][:4], "venue": "arXiv preprint",
            "doi": f"10.48550/arxiv.{arxiv_id}", "url": f"https://arxiv.org/abs/{arxiv_id}", "arxiv_id": arxiv_id,
            "provenance": "arxiv", "match_ratio": round(ratio, 3)}


def from_doi(doi: str, title: str) -> Dict:
    r = get(f"https://api.crossref.org/works/{doi}")
    if r is None:
        raise LookupError(f"DOI {doi}: not in Crossref")
    m = r.json()["message"]
    got = (m.get("title") or [""])[0]
    ratio = similarity(title, got)
    if ratio < MATCH:
        raise LookupError(f"DOI {doi}: TITLE MISMATCH ({ratio:.2f}) expected {title!r}, got {got!r}")
    parts = ((m.get("issued") or {}).get("date-parts") or [[None]])[0]
    return {"title": got, "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author") or []],
            "year": str(parts[0] or ""), "venue": (m.get("container-title") or [""])[0], "doi": doi.lower(),
            "url": f"https://doi.org/{doi}", "citations": m.get("is-referenced-by-count"), "provenance": "doi",
            "match_ratio": round(ratio, 3)}


def main() -> None:
    setup_logging()
    start = time.time()
    corpus = json.loads((REVIEW / "data" / "processed" / "screened_all.json").read_text(encoding="utf-8"))
    entries = json.loads((REVIEW / "included.json").read_text(encoding="utf-8"))
    refs, failures = [], []
    for i, e in enumerate(entries, 1):
        try:
            # An explicit identifier wins: corpus title matches can pick a same-titled different version.
            if e.get("arxiv"):
                ref = from_arxiv(e["arxiv"], e["title"]); time.sleep(DELAY_S)
            elif e.get("doi"):
                ref = from_doi(e["doi"], e["title"]); time.sleep(DELAY_S)
            else:
                ref = from_corpus(e["title"], corpus)
                if ref is None:
                    raise LookupError(f"not in corpus and no identifier given: {e['title']!r}")
        except LookupError as exc:
            failures.append(str(exc))
            logging.error(str(exc))
            continue
        if e.get("venue"):  # published venue the source lacks (e.g. arXiv record of a NeurIPS paper)
            ref["venue_note"] = e["venue"]
        if e.get("year"):  # that venue's year; the arXiv record only carries the preprint's year
            ref["source_year"], ref["year"] = ref["year"], e["year"]
        ref.update(theme=e.get("theme", ""), role=e.get("role", ""))
        # Resolution route is not the same as retrieval: an explicit arXiv ID or DOI often pins a paper the
        # search also found. PRISMA counts need the retrieval fact, so record it separately.
        ref["found_by_search"] = ref["provenance"] == "corpus" or from_corpus(ref["title"], corpus) is not None
        ref["key"] = f"{surname(ref['authors'][0]) if ref['authors'] else 'anon'}{ref['year'] or 'nd'}"
        refs.append(ref)
        logging.info(f"[{i}/{len(entries)}] {ref['provenance']:6s} {ref['match_ratio']:.2f} {ref['key']:18s} {ref['title'][:60]}")
    groups: Dict[str, List[Dict]] = {}
    for ref in refs:  # same first author and year -> a, b, c suffixes, ordered by title as lit_render prints them
        groups.setdefault(ref["key"], []).append(ref)
    for base, members in groups.items():
        if len(members) > 1:
            for i, ref in enumerate(sorted(members, key=lambda r: r["title"].lower())):
                ref["key"] = base + chr(ord("a") + i)
    (REVIEW / "references.json").write_text(json.dumps(refs, indent=1, ensure_ascii=False), encoding="utf-8")
    logging.info(f"{len(refs)} references written, {len(failures)} failures, {time.time() - start:.0f}s")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
