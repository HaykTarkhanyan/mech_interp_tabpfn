#!/usr/bin/env python3
"""Extract citation metadata from publications no bibliographic database indexes.

transformer-circuits.pub renders its byline client-side, so the static HTML carries no
author information at all - `curl | grep citation_author` returns nothing. Playwright
renders the page first, then the author list, affiliations and publication date are read
out of the DOM. Distill exposes proper citation_* meta tags and is read directly.

Without this, these references would carry empty author lists (or, worse, author lists
written from memory). Output feeds scripts/build_references.py.

Usage:
    python scripts/fetch_web_sources.py
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

from playwright.sync_api import sync_playwright

TIMEOUT_MS = 180000  # monosemanticity pages are very large documents

SOURCES = [
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
    handler = logging.StreamHandler(sys.stdout)
    handler.setStream(open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), handler],
    )


def clean_authors(raw: str) -> List[str]:
    """Split a byline into names, dropping footnote markers and affiliation daggers."""
    raw = re.sub(r"[*∗†‡§¶]", "", raw)
    parts = re.split(r",|\band\b", raw)
    names = []
    for part in parts:
        name = " ".join(part.split())
        if name and len(name) > 2 and not name.isdigit():
            names.append(name)
    return names


def extract(page, url: str) -> Dict:
    meta_authors = page.eval_on_selector_all(
        'meta[name="citation_author"]', "els => els.map(e => e.content)")
    if meta_authors:
        date = page.eval_on_selector_all(
            'meta[name="citation_publication_date"]', "els => els.map(e => e.content)")
        return {"authors": meta_authors, "published": (date or [""])[0], "affiliations": "",
                "extracted_via": "citation_* meta tags"}

    text = page.evaluate("() => document.body.innerText")
    section = lambda name: (
        re.split(r"\n(?:AUTHORS|AFFILIATIONS|PUBLISHED)\b", text.split(name, 1)[1], 1)[0].strip()
        if name in text else "")
    authors_raw = section("AUTHORS")
    if not authors_raw:
        raise RuntimeError(f"no authors found in rendered DOM for {url}")
    return {
        "authors": clean_authors(authors_raw),
        "affiliations": " ".join(section("AFFILIATIONS").split()),
        "published": " ".join(section("PUBLISHED").split()),
        "extracted_via": "rendered DOM byline",
    }


def main() -> None:
    setup_logging(Path("logs") / "fetch_web_sources.log")
    results: List[Dict] = []
    failures: List[Dict] = []

    # Resume: these pages carry heavy interactive figures and some fail intermittently.
    out = Path("data/processed/web_sources.json")
    if out.exists():
        results = json.loads(out.read_text(encoding="utf-8"))
        done = {r["url"] for r in results}
        logging.info(f"Resuming: {len(done)} sources already fetched")
    else:
        done = set()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        for title, year, venue, url in SOURCES:
            if url in done:
                continue
            try:
                # "networkidle" never settles on pages whose visualisations keep
                # streaming data, and "domcontentloaded" still times out on the
                # multi-megabyte monosemanticity write-ups. Commit the navigation,
                # then wait for the byline itself to appear.
                page.goto(url, wait_until="commit", timeout=TIMEOUT_MS)
                # document.body is still null right after "commit", so guard it.
                page.wait_for_function(
                    "() => !!document.body && ("
                    "document.body.innerText.includes('AUTHORS') || "
                    "document.querySelector('meta[name=\"citation_author\"]') !== null)",
                    timeout=TIMEOUT_MS)
                meta = extract(page, url)
            except Exception as exc:
                failures.append({"title": title, "url": url, "error": str(exc)})
                logging.error(f"FAILED {url}: {exc}")
                continue

            page_title = page.title().strip()
            results.append({
                "title": title, "page_title": page_title, "year": str(year),
                "venue": venue, "url": url, **meta,
            })
            logging.info(f"OK  {title[:52]:52s} authors={len(meta['authors'])} "
                         f"published={meta['published'][:20]!r} via={meta['extracted_via']}")
        browser.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"Wrote {len(results)} records to {out}")

    if failures:
        logging.error(f"{len(failures)} sources failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
