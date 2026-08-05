#!/usr/bin/env python3
"""Assemble the review's reference list from verified sources only.

Three provenance classes, in descending order of preference:

  corpus  - pulled from the systematic search results by title match. Metadata comes
            from OpenAlex/Crossref/DBLP as retrieved; nothing is retyped by hand.
  arxiv   - fetched live from arxiv.org/abs/<id>. The arXiv API is HTTP 429 to this
            network but the website serves citation_* meta tags. The fetched title is
            compared against the expected title and a mismatch is a hard failure, so a
            wrong arXiv ID cannot silently enter the bibliography.
  web     - publications no bibliographic database indexes (transformer-circuits.pub,
            Distill). URL-verified; author lists taken from the page where the page
            exposes them, otherwise left empty rather than invented.

Nothing enters lit_review/references.json that was not resolved by one of these paths.

Usage:
    python scripts/build_references.py
"""

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

USER_AGENT = "mech-interp-lit-review/1.0 (mailto:tarkhanyan02@gmail.com)"
TIMEOUT_S = 30
DELAY_S = 2.0
TITLE_MATCH_THRESHOLD = 0.85

# --- Class 1: papers to lift out of the systematic corpus, by title -------------------
CORPUS_TITLES = [
    "In-context Learning and Induction Heads",
    "Progress measures for grokking via mechanistic interpretability",
    "Towards Automated Circuit Discovery for Mechanistic Interpretability",
    "Learning Transformer Programs",
    "Birth of a Transformer: A Memory Viewpoint",
    "Sparse Autoencoders Find Highly Interpretable Features in Language Models",
    "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2",
    "Scaling and evaluating sparse autoencoders",
    "Improving Dictionary Learning with Gated Sparse Autoencoders",
    "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders",
    "Transcoders Find Interpretable LLM Feature Circuits",
    "Automatically Interpreting Millions of Features in Large Language Models",
    "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability",
    "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations",
    "Interpretability at Scale: Identifying Causal Mechanisms in Alpaca",
    "Towards Best Practices of Activation Patching in Language Models",
    "How to use and interpret activation patching",
    "Localizing Model Behavior with Path Patching",
    "Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models",
    "Locating and Editing Factual Associations in GPT",
    "Eliciting Latent Predictions from Transformers with the Tuned Lens",
    "Does Transformer Interpretability Transfer to RNNs?",
    "The Linear Representation Hypothesis and the Geometry of Large Language Models",
    "Steering Language Models With Activation Engineering",
    "Mechanistic Interpretability for AI Safety -- A Review",
    "Open Problems in Mechanistic Interpretability",
    "A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models",
    "Toward Transparent AI: A Survey on Interpreting the Inner Structures of Deep Neural Networks",
    "Rethinking Interpretability in the Era of Large Language Models",
    "Explaining AI through mechanistic interpretability",
    "Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP",
]

# --- Class 2: arXiv IDs, each checked against its expected title ----------------------
ARXIV_SOURCES = [
    ("2209.10652", "Toy Models of Superposition"),
    ("2211.00593", "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"),
    ("2210.13382", "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task"),
    ("2310.01405", "Representation Engineering: A Top-Down Approach to AI Transparency"),
    ("2306.03341", "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"),
    ("2305.01610", "Finding Neurons in a Haystack: Case Studies with Sparse Probing"),
    ("2406.11717", "Refusal in Language Models Is Mediated by a Single Direction"),
    ("2312.06681", "Steering Llama 2 via Contrastive Activation Addition"),
    ("2310.02207", "Language Models Represent Space and Time"),
    ("2310.10348", "Attribution Patching Outperforms Automated Circuit Discovery"),
    # Titles below are the ones arXiv actually serves. Two earlier guesses were wrong and
    # the title-match guard rejected them: 2501.17727 is "Automated Interpretability
    # Metrics..." (not "Sparse Autoencoders Can Interpret Randomly Initialized
    # Transformers"), and 2408.01416 carries the "Surveying ... Causal Mediation
    # Analysis" subtitle. Both would have been silently wrong citations.
    ("2501.17727", "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers"),
    ("2502.16681", "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing"),
    ("2312.03656", "Interpretability Illusions in the Generalization of Simplified Models"),
    ("1902.10186", "Attention is not Explanation"),
    ("2408.01416", "The Quest for the Right Mediator: Surveying Mechanistic Interpretability Through the Lens of Causal Mediation Analysis"),
]

# --- Class 3: not indexed anywhere; cite from source ----------------------------------
WEB_SOURCES = [
    ("Zoom In: An Introduction to Circuits", 2020, "Distill",
     "https://distill.pub/2020/circuits/zoom-in/"),
    ("A Mathematical Framework for Transformer Circuits", 2021, "Transformer Circuits Thread",
     "https://transformer-circuits.pub/2021/framework/index.html"),
    ("Softmax Linear Units", 2022, "Transformer Circuits Thread",
     "https://transformer-circuits.pub/2022/solu/index.html"),
    ("Towards Monosemanticity: Decomposing Language Models With Dictionary Learning", 2023,
     "Transformer Circuits Thread",
     "https://transformer-circuits.pub/2023/monosemantic-features/index.html"),
    ("Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", 2024,
     "Transformer Circuits Thread",
     "https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html"),
    ("Circuit Tracing: Revealing Computational Graphs in Language Models", 2025,
     "Transformer Circuits Thread",
     "https://transformer-circuits.pub/2025/attribution-graphs/methods.html"),
    ("On the Biology of a Large Language Model", 2025, "Transformer Circuits Thread",
     "https://transformer-circuits.pub/2025/attribution-graphs/biology.html"),
]


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def normalise(title: str) -> str:
    text = unicodedata.normalize("NFKD", (title or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text)


def similarity(a: str, b: str) -> float:
    na, nb = normalise(a), normalise(b)
    ratio = SequenceMatcher(None, na, nb).ratio()
    if len(nb) >= 25 and (na in nb or nb in na):
        ratio = max(ratio, 0.95)
    return ratio


def cite_key(authors: List[str], year, title: str) -> str:
    """Stable ASCII citation key. Handles two source quirks:

    - DBLP disambiguates homonyms with a numeric suffix ("Kiho Park 0001"), which
      naively yields the key "00012024".
    - Crossref/OpenAlex return "Surname, Given" while DBLP returns "Given Surname".
    """
    if authors:
        name = re.sub(r"\s+\d{4}$", "", authors[0].strip())        # drop DBLP suffix
        surname = name.split(",")[0].strip().split()[-1] if "," not in name else name.split(",")[0].strip()
        surname = surname.split()[-1] if surname.split() else surname
    else:
        surname = title.split()[0]
    ascii_surname = "".join(
        c for c in unicodedata.normalize("NFKD", surname) if not unicodedata.combining(c))
    ascii_surname = re.sub(r"[^A-Za-z]", "", ascii_surname).lower() or "anon"
    return f"{ascii_surname}{year or 'nd'}"


def from_corpus(titles: List[str], corpus: List[Dict]) -> tuple:
    found, missing = [], []
    for title in titles:
        best, best_ratio = None, 0.0
        for rec in corpus:
            ratio = similarity(title, rec.get("title", ""))
            if ratio > best_ratio:
                best, best_ratio = rec, ratio
        if best and best_ratio >= TITLE_MATCH_THRESHOLD:
            authors = [a.strip() for a in (best.get("authors") or "").split(",") if a.strip()]
            found.append({
                "key": cite_key(authors, best.get("year"), best["title"]),
                "title": best["title"],
                "authors": authors,
                "year": best.get("year"),
                "venue": best.get("venue", ""),
                "doi": best.get("doi", ""),
                "url": best.get("url", ""),
                "citations": best.get("citations"),
                "provenance": "corpus",
                "found_by": best.get("found_by", []),
                "match_ratio": round(best_ratio, 3),
            })
            logging.info(f"corpus  {best_ratio:.2f}  {best['title'][:66]}")
        else:
            missing.append(title)
            logging.error(f"corpus  NOT FOUND ({best_ratio:.2f})  {title[:60]}")
    return found, missing


def from_arxiv(sources: List[tuple]) -> tuple:
    found, failed = [], []
    for arxiv_id, expected_title in sources:
        try:
            response = requests.get(f"https://arxiv.org/abs/{arxiv_id}", timeout=TIMEOUT_S,
                                    allow_redirects=True, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            failed.append((arxiv_id, expected_title, f"request failed: {exc}"))
            logging.error(f"arxiv   {arxiv_id} request failed: {exc}")
            continue

        if response.status_code != 200:
            failed.append((arxiv_id, expected_title, f"HTTP {response.status_code}"))
            logging.error(f"arxiv   {arxiv_id} HTTP {response.status_code}")
            time.sleep(DELAY_S)
            continue

        html = response.text
        grab = lambda p: re.findall(p, html, re.I)
        titles = grab(r'name=["\']citation_title["\'][^>]+content=["\']([^"\']+)')
        authors = grab(r'name=["\']citation_author["\'][^>]+content=["\']([^"\']+)')
        dates = grab(r'name=["\']citation_date["\'][^>]+content=["\']([^"\']+)')
        doi_meta = grab(r'name=["\']citation_doi["\'][^>]+content=["\']([^"\']+)')

        actual_title = titles[0] if titles else ""
        ratio = similarity(expected_title, actual_title)
        if ratio < TITLE_MATCH_THRESHOLD:
            # A wrong arXiv ID must never enter the bibliography silently.
            failed.append((arxiv_id, expected_title, f"title mismatch ({ratio:.2f}): got {actual_title!r}"))
            logging.error(f"arxiv   {arxiv_id} TITLE MISMATCH ({ratio:.2f})\n"
                          f"          expected: {expected_title}\n"
                          f"          actual:   {actual_title}")
            time.sleep(DELAY_S)
            continue

        year = (dates[0][:4] if dates else "")
        found.append({
            "key": cite_key(authors, year, actual_title),
            "title": actual_title,
            "authors": authors,
            "year": year,
            "venue": "arXiv preprint",
            "doi": (doi_meta[0] if doi_meta else f"10.48550/arXiv.{arxiv_id}").lower(),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "arxiv_id": arxiv_id,
            "provenance": "arxiv",
            "match_ratio": round(ratio, 3),
        })
        logging.info(f"arxiv   {ratio:.2f}  {arxiv_id}  {actual_title[:56]}")
        time.sleep(DELAY_S)
    return found, failed


def from_web(sources: List[tuple]) -> tuple:
    """Read the Playwright-rendered metadata produced by scripts/fetch_web_sources.py.

    These pages render their bylines client-side, so authors cannot be scraped from the
    static HTML. Anything absent from that file is a hard failure rather than a
    reference with an invented author list.
    """
    path = Path("data/processed/web_sources.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/fetch_web_sources.py first")
    fetched = {r["url"]: r for r in json.loads(path.read_text(encoding="utf-8"))}

    found, failed = [], []
    for title, year, venue, url in sources:
        record = fetched.get(url)
        if not record or not record.get("authors"):
            failed.append((title, url, "not present in web_sources.json (or no authors)"))
            logging.error(f"web     MISSING METADATA  {title[:56]}")
            continue
        authors = record["authors"]
        found.append({
            "key": cite_key(authors, year, title),
            "title": title,
            "authors": authors,
            "year": str(year),
            "venue": venue,
            "doi": "",
            "url": url,
            "affiliations": record.get("affiliations", ""),
            "published": record.get("published", ""),
            "provenance": "web-not-indexed",
            "authors_extracted_via": record.get("extracted_via", ""),
            "note": "Not indexed in OpenAlex, Crossref or DBLP; cited from publisher URL",
        })
        logging.info(f"web     OK  {title[:52]:52s} authors={len(authors)}")
    return found, failed


def main() -> None:
    setup_logging(Path("logs") / "build_references.log")

    corpus = json.loads(Path("data/processed/screened_all.json").read_text(encoding="utf-8"))
    logging.info(f"Corpus: {len(corpus)} deduplicated records")

    logging.info("=== Class 1: from systematic corpus ===")
    corpus_refs, corpus_missing = from_corpus(CORPUS_TITLES, corpus)

    logging.info("=== Class 2: from arXiv (title-verified) ===")
    arxiv_refs, arxiv_failed = from_arxiv(ARXIV_SOURCES)

    logging.info("=== Class 3: not indexed, cited from publisher ===")
    web_refs, web_failed = from_web(WEB_SOURCES)

    refs = corpus_refs + arxiv_refs + web_refs

    # Disambiguate duplicate cite keys (same first author + year).
    seen: Dict[str, int] = {}
    for ref in refs:
        base = ref["key"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            ref["key"] = f"{base}{chr(ord('a') + seen[base] - 1)}"

    out = Path("lit_review/references.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(refs, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info(f"=== {len(refs)} references written to {out} ===")
    logging.info(f"    corpus={len(corpus_refs)}  arxiv={len(arxiv_refs)}  web={len(web_refs)}")
    if corpus_missing:
        logging.error(f"    {len(corpus_missing)} corpus titles unresolved: {corpus_missing}")
    if arxiv_failed:
        logging.error(f"    {len(arxiv_failed)} arXiv sources failed: {arxiv_failed}")
    if web_failed:
        logging.error(f"    {len(web_failed)} web sources failed: {web_failed}")

    if corpus_missing or arxiv_failed or web_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
