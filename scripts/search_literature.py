#!/usr/bin/env python3
"""Search academic databases for mechanistic interpretability literature.

Databases (all keyless):
  - OpenAlex   : primary. Broad coverage, citation counts, abstracts, indexes arXiv.
  - Crossref   : published-venue metadata and citation counts.
  - DBLP       : authoritative CS venue names (NeurIPS/ICML/ICLR/ACL).
  - arXiv      : PROBED. Returns HTTP 429 to this network (checked 2026-08-05).
  - SemScholar : PROBED. Keyless shared pool returns HTTP 429 under load.

Probed sources are never silently skipped: the probe result is written to
data/raw/search_counts.json and belongs in the review's methods section.
arXiv preprint coverage is retained regardless, because OpenAlex indexes arXiv.

Usage:
    python scripts/search_literature.py --test        # single pilot query
    python scripts/search_literature.py               # full sweep
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

OPENALEX_DELAY_S = 1.0
CROSSREF_DELAY_S = 1.0
DBLP_DELAY_S = 2.0
S2_DELAY_S = 3.5
MAX_RETRIES = 4
TIMEOUT_S = 45
CONTACT_EMAIL = "tarkhanyan02@gmail.com"

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
DBLP_ENDPOINT = "https://dblp.org/search/publ/api"
ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,authors,citationCount,externalIds,venue,url"

USER_AGENT = f"mech-interp-lit-review/1.0 (mailto:{CONTACT_EMAIL})"

# "query"      -> natural language, for OpenAlex / Crossref / Semantic Scholar rankers.
# "dblp"       -> short keyword form; DBLP ANDs every term, so long queries return 0 hits.
THEMES: List[Dict[str, str]] = [
    {"key": "superposition",
     "query": "superposition polysemanticity toy models neural network features",
     "dblp": "superposition polysemanticity"},
    {"key": "sparse_autoencoders",
     "query": "sparse autoencoder dictionary learning language model monosemantic features",
     "dblp": "sparse autoencoder interpretability"},
    {"key": "circuits",
     "query": "circuit analysis transformer mechanistic interpretability",
     "dblp": "circuit transformer interpretability"},
    {"key": "induction_heads",
     "query": "induction heads in-context learning attention transformer circuits",
     "dblp": "induction heads transformer"},
    {"key": "activation_patching",
     "query": "activation patching causal tracing localization language model",
     "dblp": "activation patching"},
    {"key": "causal_abstraction",
     "query": "causal abstraction interchange intervention distributed alignment search",
     "dblp": "causal abstraction neural"},
    {"key": "linear_representation",
     "query": "linear representation hypothesis concept directions language model",
     "dblp": "linear representation language model"},
    {"key": "probing",
     "query": "probing classifiers internal representations language models",
     "dblp": "probing language model representations"},
    {"key": "logit_lens",
     "query": "logit lens tuned lens intermediate layer predictions transformer",
     "dblp": "logit lens transformer"},
    {"key": "knowledge_editing",
     "query": "locating and editing factual associations GPT knowledge editing",
     "dblp": "knowledge editing language model"},
    {"key": "steering",
     "query": "activation steering representation engineering controlling language model behavior",
     "dblp": "activation steering language model"},
    {"key": "grokking",
     "query": "grokking phase transitions emergent structure training dynamics interpretability",
     "dblp": "grokking neural networks"},
    {"key": "transcoders",
     "query": "transcoders attribution graphs cross-layer circuit tracing",
     "dblp": "transcoder interpretability"},
    {"key": "auto_interp",
     "query": "automated interpretability explaining neurons language models scaling",
     "dblp": "automated interpretability neurons"},
    {"key": "evaluation",
     "query": "evaluating interpretability methods faithfulness benchmark critique",
     "dblp": "faithfulness interpretability evaluation"},
    {"key": "surveys",
     "query": "mechanistic interpretability survey review open problems",
     "dblp": "mechanistic interpretability survey"},
]


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def request_with_retry(url: str, params: Dict, source: str) -> requests.Response:
    """GET with backoff on 429/5xx. Raises after MAX_RETRIES - never returns bad data."""
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(url, params=params, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
        if response.status_code == 200:
            return response
        if response.status_code in (429, 500, 502, 503, 504):
            backoff = 4.0 * (2 ** attempt)
            logging.warning(
                f"{source} returned {response.status_code} (attempt {attempt}/{MAX_RETRIES}); "
                f"retrying in {backoff:.0f}s"
            )
            time.sleep(backoff)
            continue
        response.raise_for_status()
    raise RuntimeError(f"{source} failed after {MAX_RETRIES} attempts for params={params}")


def probe(url: str, params: Dict, name: str) -> Optional[str]:
    """One cheap request. Returns None if usable, else a reason string to record."""
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        return f"request failed: {exc}"
    return None if response.status_code == 200 else f"HTTP {response.status_code}"


def strip_tags(text: str) -> str:
    """Crossref abstracts arrive as JATS XML."""
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """OpenAlex stores abstracts as {word: [positions]}. Rebuild the text."""
    if not inverted_index:
        return ""
    positions: Dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


def record(title, authors, year, abstract, doi, url, venue, source, citations, arxiv_id="") -> Dict:
    author_list = [a for a in authors if a]
    return {
        "title": " ".join((title or "").split()),
        "authors": ", ".join(author_list),
        "first_author": author_list[0].split()[-1] if author_list else "",
        "year": str(year or ""),
        "abstract": " ".join((abstract or "").split()),
        "doi": (doi or "").replace("https://doi.org/", "").lower(),
        "arxiv_id": arxiv_id,
        "url": url or "",
        "venue": venue or "",
        "source": source,
        "citations": citations,
    }


def search_openalex(query: str, max_results: int) -> List[Dict]:
    params = {
        "search": query,
        "per-page": min(max_results, 200),
        "mailto": CONTACT_EMAIL,
        "select": "id,doi,title,publication_year,cited_by_count,authorships,"
                  "abstract_inverted_index,primary_location,type",
    }
    response = request_with_retry(OPENALEX_ENDPOINT, params, "OpenAlex")
    papers = []
    for item in response.json().get("results") or []:
        doi = (item.get("doi") or "").replace("https://doi.org/", "").lower()
        arxiv_id = doi.split("arxiv.", 1)[1] if doi.startswith("10.48550/arxiv.") else ""
        venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        papers.append(record(
            title=item.get("title"),
            authors=[(a.get("author") or {}).get("display_name", "") for a in (item.get("authorships") or [])],
            year=item.get("publication_year"),
            abstract=reconstruct_abstract(item.get("abstract_inverted_index")),
            doi=doi, url=item.get("id"), venue=venue,
            source="OpenAlex", citations=item.get("cited_by_count"), arxiv_id=arxiv_id,
        ))
    return papers


def search_crossref(query: str, max_results: int) -> List[Dict]:
    params = {
        "query": query,
        "rows": max_results,
        "mailto": CONTACT_EMAIL,
        "select": "DOI,title,author,issued,container-title,is-referenced-by-count,abstract,type",
    }
    response = request_with_retry(CROSSREF_ENDPOINT, params, "Crossref")
    papers = []
    for item in (response.json().get("message") or {}).get("items") or []:
        date_parts = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
        authors = [
            " ".join(filter(None, [a.get("given"), a.get("family")]))
            for a in (item.get("author") or [])
        ]
        doi = (item.get("DOI") or "").lower()
        papers.append(record(
            title=(item.get("title") or [""])[0],
            authors=authors,
            year=date_parts[0] if date_parts else None,
            abstract=strip_tags(item.get("abstract", "")),
            doi=doi,
            url=f"https://doi.org/{doi}" if doi else "",
            venue=(item.get("container-title") or [""])[0],
            source="Crossref", citations=item.get("is-referenced-by-count"),
        ))
    return papers


def search_dblp(query: str, max_results: int) -> List[Dict]:
    params = {"q": query, "format": "json", "h": max_results}
    response = request_with_retry(DBLP_ENDPOINT, params, "DBLP")
    hits = ((response.json().get("result") or {}).get("hits") or {}).get("hit") or []
    papers = []
    for hit in hits:
        info = hit.get("info") or {}
        raw_authors = (info.get("authors") or {}).get("author") or []
        if isinstance(raw_authors, dict):          # DBLP collapses single-author lists
            raw_authors = [raw_authors]
        authors = [a.get("text", "") if isinstance(a, dict) else str(a) for a in raw_authors]
        papers.append(record(
            title=info.get("title"), authors=authors, year=info.get("year"),
            abstract="",                            # DBLP carries no abstracts
            doi=info.get("doi", ""), url=info.get("ee") or info.get("url", ""),
            venue=info.get("venue", ""), source="DBLP", citations=None,
        ))
    return papers


def search_semantic_scholar(query: str, max_results: int) -> List[Dict]:
    response = request_with_retry(
        S2_ENDPOINT, {"query": query, "limit": max_results, "fields": S2_FIELDS}, "SemanticScholar"
    )
    papers = []
    for item in response.json().get("data") or []:
        external = item.get("externalIds") or {}
        papers.append(record(
            title=item.get("title"),
            authors=[a.get("name", "") for a in (item.get("authors") or [])],
            year=item.get("year"), abstract=item.get("abstract"),
            doi=external.get("DOI", ""), url=item.get("url"), venue=item.get("venue"),
            source="SemanticScholar", citations=item.get("citationCount"),
            arxiv_id=external.get("ArXiv", "") or "",
        ))
    return papers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run a single pilot query and exit")
    parser.add_argument("--max-results", type=int, default=50, help="results per query per source")
    parser.add_argument("--outdir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    setup_logging(Path("logs") / "search_literature.log")
    args.outdir.mkdir(parents=True, exist_ok=True)

    probes = {
        "arXiv": probe(ARXIV_ENDPOINT, {"search_query": "all:superposition", "max_results": 1}, "arXiv"),
        "SemanticScholar": probe(S2_ENDPOINT, {"query": "interpretability", "limit": 1, "fields": "title"}, "S2"),
    }
    for name, status in probes.items():
        if status is None:
            logging.info(f"{name} probe: OK - will be used")
        else:
            logging.error(f"{name} UNAVAILABLE ({status}). Recorded in search_counts.json; "
                          f"must be disclosed in the review methods section.")

    # (display name, function, inter-request delay, THEMES key holding that source's query)
    searchers = [("OpenAlex", search_openalex, OPENALEX_DELAY_S, "query"),
                 ("Crossref", search_crossref, CROSSREF_DELAY_S, "query"),
                 ("DBLP", search_dblp, DBLP_DELAY_S, "dblp")]
    if probes["SemanticScholar"] is None:
        searchers.append(("SemanticScholar", search_semantic_scholar, S2_DELAY_S, "query"))

    themes = THEMES[:1] if args.test else THEMES
    limit = 5 if args.test else args.max_results
    logging.info(f"Searching {len(themes)} theme(s) x {len(searchers)} source(s), up to {limit} results each")

    all_papers: List[Dict] = []
    per_source: Dict[str, List[Dict]] = {name: [] for name, _, _, _ in searchers}
    counts: List[Dict] = []

    for i, theme in enumerate(themes, 1):
        logging.info(f"[{i}/{len(themes)}] theme={theme['key']}")
        theme_counts = {"theme": theme["key"]}
        for name, fn, delay, query_key in searchers:
            hits = fn(theme[query_key], limit)
            logging.info(f"  {name:16s} {len(hits):3d} hits")
            for paper in hits:
                paper["theme"] = theme["key"]
            per_source[name].extend(hits)
            all_papers.extend(hits)
            theme_counts[name] = len(hits)
            time.sleep(delay)
        counts.append(theme_counts)

    if args.test:
        logging.info("Pilot results:")
        for paper in all_papers:
            cites = paper["citations"] if paper["citations"] is not None else "-"
            logging.info(f"  [{paper['source']:15s}] {paper['year']} cites={str(cites):<6} {paper['title'][:65]}")
        return

    for source, papers in per_source.items():
        path = args.outdir / f"{source.lower()}_raw.json"
        path.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info(f"Wrote {len(papers)} records to {path}")

    (args.outdir / "combined_raw.json").write_text(
        json.dumps(all_papers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.outdir / "search_counts.json").write_text(
        json.dumps({
            "search_date": time.strftime("%Y-%m-%d"),
            "databases_used": [name for name, _, _, _ in searchers],
            "databases_unavailable": {k: v for k, v in probes.items() if v is not None},
            "results_per_query_per_source": limit,
            "themes": [t["key"] for t in themes],
            "queries": {t["key"]: {"default": t["query"], "dblp": t["dblp"]} for t in themes},
            "per_theme": counts,
            "total_records": len(all_papers),
        }, indent=2),
        encoding="utf-8",
    )
    logging.info(f"Wrote {len(all_papers)} total records to {args.outdir / 'combined_raw.json'}")


if __name__ == "__main__":
    main()
