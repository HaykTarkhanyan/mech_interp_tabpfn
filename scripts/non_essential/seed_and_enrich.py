#!/usr/bin/env python3
"""Supplementary retrieval: targeted seed lookups + abstract enrichment.

Two problems in the systematic sweep this repairs:

A. SEED SEARCH. Keyword search misses foundational works whose titles do not contain
   the field's current vocabulary, and much of this field's founding literature was
   published outside DOI-issuing venues (transformer-circuits.pub, Distill) so it is
   thinly indexed. Each title below is looked up explicitly. Titles that come back
   NOT FOUND are reported, not quietly dropped - they need citing from source.

B. ABSTRACT ENRICHMENT. 49% of deduplicated records carried no abstract (Crossref
   often omits them; DBLP never has them). The relevance screen reads title+abstract,
   so those records were being excluded for missing data rather than for being
   off-topic - a silent bias against exactly the peer-reviewed venue records DBLP and
   Crossref contribute. Abstracts are refetched from OpenAlex by DOI, batched 50 per
   request via the `doi:a|b|c` filter.

Usage:
    python scripts/seed_and_enrich.py
"""

import json
import logging
import sys
import time
import unicodedata
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import requests

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
CONTACT_EMAIL = "tarkhanyan02@gmail.com"
USER_AGENT = f"mech-interp-lit-review/1.0 (mailto:{CONTACT_EMAIL})"
DELAY_S = 2.5            # OpenAlex enforces a short-term burst limit beyond the documented rate
TIMEOUT_S = 45
MAX_RETRIES = 6
TITLE_MATCH_THRESHOLD = 0.85
DOI_BATCH = 50

# Canonical works of the field, by exact title. Presence/absence is itself a result.
SEED_TITLES = [
    "Zoom In: An Introduction to Circuits",
    "A Mathematical Framework for Transformer Circuits",
    "In-context Learning and Induction Heads",
    "Toy Models of Superposition",
    "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small",
    "Locating and Editing Factual Associations in GPT",
    "Mass-Editing Memory in a Transformer",
    "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning",
    "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet",
    "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task",
    "Progress measures for grokking via mechanistic interpretability",
    "Sparse Autoencoders Find Highly Interpretable Features in Language Models",
    "Scaling and evaluating sparse autoencoders",
    "Improving Dictionary Learning with Gated Sparse Autoencoders",
    "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders",
    "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2",
    "Transcoders Find Interpretable LLM Feature Circuits",
    "Circuit Tracing: Revealing Computational Graphs in Language Models",
    "On the Biology of a Large Language Model",
    "Eliciting Latent Predictions from Transformers with the Tuned Lens",
    "Towards Automated Circuit Discovery for Mechanistic Interpretability",
    "Attribution Patching Outperforms Automated Circuit Discovery",
    "Towards Best Practices of Activation Patching in Language Models",
    "Finding Neurons in a Haystack: Case Studies with Sparse Probing",
    "The Linear Representation Hypothesis and the Geometry of Large Language Models",
    "Language Models Represent Space and Time",
    "Refusal in Language Models Is Mediated by a Single Direction",
    "Steering Llama 2 via Contrastive Activation Addition",
    "Representation Engineering: A Top-Down Approach to AI Transparency",
    "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model",
    "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations",
    "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability",
    "Interpretability at Scale: Identifying Causal Mechanisms in Alpaca",
    "Mechanistic Interpretability for AI Safety: A Review",
    "Open Problems in Mechanistic Interpretability",
    "Sparse Autoencoders Can Interpret Randomly Initialized Transformers",
    "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing",
    "The Quest for the Right Mediator: A History, Survey, and Theoretical Grounding of Causal Interpretability",
    "Interpretability Illusions in the Generalization of Simplified Models",
    "Do Machine Learning Models Memorize or Generalize?",
    "Softmax Linear Units",
    "Attention Is Not Explanation",
    "Rethinking Interpretability in the Era of Large Language Models",
    "Bilinear MLPs enable weight-based mechanistic interpretability",
    "Automatically Interpreting Millions of Features in Large Language Models",
    "Language models can explain neurons in language models",
]


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def normalise_title(title: str) -> str:
    text = unicodedata.normalize("NFKD", (title or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text)


def request_with_retry(params: Dict) -> requests.Response:
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(OPENALEX_ENDPOINT, params=params, timeout=TIMEOUT_S,
                                headers={"User-Agent": USER_AGENT})
        if response.status_code == 200:
            return response
        if response.status_code in (429, 500, 502, 503, 504):
            backoff = 4.0 * (2 ** attempt)
            logging.warning(f"OpenAlex {response.status_code} (attempt {attempt}/{MAX_RETRIES}); "
                            f"retry in {backoff:.0f}s")
            time.sleep(backoff)
            continue
        response.raise_for_status()
    raise RuntimeError(f"OpenAlex failed after {MAX_RETRIES} attempts for {params}")


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not inverted_index:
        return ""
    positions: Dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


SELECT = ("id,doi,title,publication_year,cited_by_count,authorships,"
          "abstract_inverted_index,primary_location,type")


def to_record(item: Dict, source: str) -> Dict:
    authors = [(a.get("author") or {}).get("display_name", "") for a in (item.get("authorships") or [])]
    authors = [a for a in authors if a]
    doi = (item.get("doi") or "").replace("https://doi.org/", "").lower()
    venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name", "") or ""
    return {
        "title": " ".join((item.get("title") or "").split()),
        "authors": ", ".join(authors),
        "first_author": authors[0].split()[-1] if authors else "",
        "year": str(item.get("publication_year") or ""),
        "abstract": " ".join(reconstruct_abstract(item.get("abstract_inverted_index")).split()),
        "doi": doi,
        "arxiv_id": doi.split("arxiv.", 1)[1] if doi.startswith("10.48550/arxiv.") else "",
        "url": item.get("id") or "",
        "venue": venue,
        "source": source,
        "citations": item.get("cited_by_count"),
    }


def seed_search(titles: List[str], cache_path: Path) -> tuple:
    """Look each title up explicitly. Returns (found_records, missing_titles).

    Results are checkpointed after every lookup, and already-resolved titles are
    skipped on re-run. OpenAlex burst-limits aggressively, so a long seed sweep can be
    interrupted; without this the whole sweep would restart and re-trigger the limit.
    """
    found, missing = [], []
    resolved = set()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        found = cached.get("found", [])
        missing = cached.get("missing", [])
        resolved = {r["seed_query"] for r in found} | set(missing)
        logging.info(f"Resuming: {len(resolved)} of {len(titles)} titles already resolved")

    for i, title in enumerate(titles, 1):
        if title in resolved:
            continue
        # OpenAlex reads ? and * as wildcards and rejects them in a stemmed search
        # with HTTP 400. Strip them from the query; match against the real title.
        query = re.sub(r"[?*]", " ", title)
        response = request_with_retry({
            "search": query, "per-page": 5, "mailto": CONTACT_EMAIL, "select": SELECT,
        })
        target = normalise_title(title)
        best, best_ratio = None, 0.0
        for item in response.json().get("results") or []:
            candidate = normalise_title(item.get("title") or "")
            ratio = SequenceMatcher(None, target, candidate).ratio()
            # Containment boost only for substantial strings: without the length guard a
            # 1978 paper titled "Introduction" scores 0.95 against
            # "Zoom In: An Introduction to Circuits".
            if len(candidate) >= 25 and (target in candidate or candidate in target):
                ratio = max(ratio, 0.95)
            if ratio > best_ratio:
                best, best_ratio = item, ratio

        if best and best_ratio >= TITLE_MATCH_THRESHOLD:
            record = to_record(best, "OpenAlex-seed")
            record["seed_query"] = title
            record["match_ratio"] = round(best_ratio, 3)
            found.append(record)
            logging.info(f"[{i}/{len(titles)}] FOUND  ({best_ratio:.2f}) {record['title'][:66]}")
        else:
            missing.append(title)
            got = (best or {}).get("title", "-")
            logging.warning(f"[{i}/{len(titles)}] MISSING  {title[:60]}  (best was: {str(got)[:40]})")

        cache_path.write_text(
            json.dumps({"found": found, "missing": missing}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        time.sleep(DELAY_S)
    return found, missing


def enrich_abstracts(dois: List[str]) -> Dict[str, Dict]:
    """Batch-fetch abstracts/citations from OpenAlex, 50 DOIs per request."""
    enriched: Dict[str, Dict] = {}
    batches = [dois[i:i + DOI_BATCH] for i in range(0, len(dois), DOI_BATCH)]
    for i, batch in enumerate(batches, 1):
        response = request_with_retry({
            "filter": "doi:" + "|".join(batch),
            "per-page": DOI_BATCH,
            "mailto": CONTACT_EMAIL,
            "select": SELECT,
        })
        results = response.json().get("results") or []
        for item in results:
            doi = (item.get("doi") or "").replace("https://doi.org/", "").lower()
            if not doi:
                continue
            enriched[doi] = {
                "abstract": " ".join(reconstruct_abstract(item.get("abstract_inverted_index")).split()),
                "citations": item.get("cited_by_count"),
                "venue": ((item.get("primary_location") or {}).get("source") or {}).get("display_name", "") or "",
            }
        logging.info(f"enrichment batch {i}/{len(batches)}: {len(batch)} requested, {len(results)} returned")
        time.sleep(DELAY_S)
    return enriched


def main() -> None:
    setup_logging(Path("logs") / "seed_and_enrich.log")
    raw_dir = Path("data/raw")

    logging.info(f"=== Stage A: seed search over {len(SEED_TITLES)} canonical titles ===")
    seeds, missing = seed_search(SEED_TITLES, Path("data/processed/seed_cache.json"))
    (raw_dir / "seed_raw.json").write_text(json.dumps(seeds, indent=2, ensure_ascii=False), encoding="utf-8")
    (raw_dir / "seed_missing.json").write_text(
        json.dumps({"not_indexed": missing, "n_searched": len(SEED_TITLES), "n_found": len(seeds)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"Seed search: {len(seeds)} found, {len(missing)} NOT INDEXED")
    for title in missing:
        logging.warning(f"  NOT INDEXED -> must be cited from source: {title}")

    logging.info("=== Stage B: abstract enrichment ===")
    records = json.loads((raw_dir / "combined_raw.json").read_text(encoding="utf-8"))
    dois = sorted({
        (r.get("doi") or "").strip().lower()
        for r in records
        if not (r.get("abstract") or "").strip() and (r.get("doi") or "").strip()
    })
    logging.info(f"{len(dois)} distinct DOIs lack an abstract; fetching from OpenAlex")
    enriched = enrich_abstracts(dois)
    filled = sum(1 for v in enriched.values() if v["abstract"])
    logging.info(f"Recovered {filled} abstracts for {len(enriched)} matched DOIs "
                 f"({len(dois) - len(enriched)} DOIs unknown to OpenAlex)")

    (Path("data/processed") / "abstract_enrichment.json").write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote data/processed/abstract_enrichment.json")


if __name__ == "__main__":
    main()
