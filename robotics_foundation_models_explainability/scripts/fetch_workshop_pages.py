#!/usr/bin/env python3
"""Fetch abstracts of workshop-only included works from their conference listing pages.

These works have no arXiv version. Their OpenReview PDFs answer HTTP 403 to non-browser clients
(checked 2026-10-19), but the neurips.cc / icml.cc virtual pages and LessWrong serve the abstract or
the full post as HTML. Text is saved to papers/workshop/<key>.txt and the status of every fetch is
written to papers/workshop/fetch_status.json, so an unreadable page is recorded, not skipped.

Usage (from robotics_foundation_models_explainability/), ~30 s:
    .venv/Scripts/python.exe scripts/fetch_workshop_pages.py
"""

import html
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "papers" / "workshop"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def page_text(raw: str) -> str:
    raw = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", raw))
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def abstract_of(text: str) -> str:
    """Conference pages put the abstract after an 'Abstract' heading; fall back to the whole text."""
    m = re.search(r"\nAbstract:?\n(.{200,6000}?)(\n(Chat is not available|Show more|Video|Poster|Live content|Successful Page Load)|$)",
                  text, re.S)
    return m.group(1).strip() if m else ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", stream=sys.stdout)
    OUT.mkdir(parents=True, exist_ok=True)
    works = [w for w in json.loads((ROOT / "lit_review" / "included.json").read_text(encoding="utf-8"))["works"]
             if w["key"].startswith("w_")]
    status = {}
    for w in works:
        url = w["url"]
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60, allow_redirects=True)
        except requests.RequestException as exc:
            status[w["key"]] = {"url": url, "status": f"request failed: {exc}"}
            logging.error(f"{w['key']}: request failed: {exc}")
            continue
        if r.status_code != 200:
            status[w["key"]] = {"url": url, "status": f"HTTP {r.status_code}"}
            logging.error(f"{w['key']}: HTTP {r.status_code} for {url}")
            continue
        text = page_text(r.text)
        # A /workshop/ URL is the workshop's own page: its "abstract" describes the workshop, not the paper.
        abstract = "" if "/workshop/" in r.url else abstract_of(text)
        forum = sorted(set(re.findall(r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)", r.text)))
        keep = abstract if abstract else text
        (OUT / f"{w['key']}.txt").write_text(f"URL: {r.url}\nOpenReview: {forum}\n\n{keep}\n", encoding="utf-8")
        status[w["key"]] = {"url": r.url, "status": "ok", "abstract_found": bool(abstract), "chars": len(keep),
                            "page_is_work": "/workshop/" not in r.url and "openreview.net" not in r.url,
                            "openreview_ids": forum}
        logging.info(f"{w['key']}: ok, abstract {'found' if abstract else 'NOT found (whole page kept)'}, {len(keep)} chars")
        time.sleep(2)
    (OUT / "fetch_status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
