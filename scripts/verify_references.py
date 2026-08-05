#!/usr/bin/env python3
"""Verify every reference cited in the review against an authoritative registry.

Why not just use the skill's verify_citations.py: it establishes existence via
doi.org/api/handles (registrar-agnostic, fine) but reads metadata only from Crossref.
arXiv DOIs are registered with DataCite, so Crossref 404s on them and the citation is
reported "verified" with an empty metadata record - a false pass covering most of a
mech interp bibliography (see DECISIONS.md #3).

This script routes by registrar:
  10.48550/arXiv.*  -> OpenAlex  (DataCite-registered preprints)
  everything else   -> Crossref
and compares the registry's title against the title cited in the review, so a DOI that
resolves to the WRONG paper is caught rather than passed.

Usage:
    python scripts/verify_references.py lit_review/references.json
"""

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

CONTACT_EMAIL = "tarkhanyan02@gmail.com"
USER_AGENT = f"mech-interp-lit-review/1.0 (mailto:{CONTACT_EMAIL})"
DELAY_S = 1.5
TIMEOUT_S = 45
MAX_RETRIES = 5
TITLE_MATCH_THRESHOLD = 0.85


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


def get(url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
    """Returns the response, or None on a definitive 404. Raises on repeated failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(url, params=params or {}, timeout=TIMEOUT_S,
                                headers={"User-Agent": USER_AGENT})
        if response.status_code == 200:
            return response
        if response.status_code == 404:
            return None
        if response.status_code in (429, 500, 502, 503, 504):
            backoff = 4.0 * (2 ** attempt)
            logging.warning(f"{url} -> {response.status_code} "
                            f"(attempt {attempt}/{MAX_RETRIES}); retry in {backoff:.0f}s")
            time.sleep(backoff)
            continue
        response.raise_for_status()
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {url}")


def verify_via_crossref(doi: str) -> Tuple[bool, Dict]:
    response = get(f"https://api.crossref.org/works/{doi}", {"mailto": CONTACT_EMAIL})
    if response is None:
        return False, {}
    msg = response.json().get("message", {})
    authors = [" ".join(filter(None, [a.get("given"), a.get("family")]))
               for a in (msg.get("author") or [])]
    date_parts = ((msg.get("issued") or {}).get("date-parts") or [[None]])[0]
    return True, {
        "title": (msg.get("title") or [""])[0],
        "authors": authors,
        "year": date_parts[0] if date_parts else None,
        "venue": (msg.get("container-title") or [""])[0],
        "registry": "Crossref",
    }


def verify_via_arxiv(arxiv_id: str) -> Tuple[bool, Dict]:
    """arXiv's API is HTTP 429 to this network, but the abs page serves citation_* tags."""
    response = get(f"https://arxiv.org/abs/{arxiv_id}")
    if response is None:
        return False, {}
    html = response.text
    grab = lambda p: re.findall(p, html, re.I)
    titles = grab(r'name=["\']citation_title["\'][^>]+content=["\']([^"\']+)')
    authors = grab(r'name=["\']citation_author["\'][^>]+content=["\']([^"\']+)')
    dates = grab(r'name=["\']citation_date["\'][^>]+content=["\']([^"\']+)')
    return bool(titles), {
        "title": titles[0] if titles else "",
        "authors": authors,
        "year": (dates[0][:4] if dates else None),
        "venue": "arXiv preprint",
        "registry": "arXiv",
    }


def verify_via_openalex(doi: str) -> Tuple[bool, Dict]:
    response = get("https://api.openalex.org/works/https://doi.org/" + doi,
                   {"mailto": CONTACT_EMAIL})
    if response is None:
        return False, {}
    item = response.json()
    authors = [(a.get("author") or {}).get("display_name", "")
               for a in (item.get("authorships") or [])]
    venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name", "") or ""
    return True, {
        "title": item.get("title") or "",
        "authors": [a for a in authors if a],
        "year": item.get("publication_year"),
        "venue": venue,
        "citations": item.get("cited_by_count"),
        "registry": "OpenAlex/DataCite",
    }


def verify_url(url: str) -> bool:
    """For sources with no DOI (Distill, transformer-circuits.pub). HEAD, then GET."""
    for method in (requests.head, requests.get):
        try:
            response = method(url, timeout=TIMEOUT_S, allow_redirects=True,
                              headers={"User-Agent": USER_AGENT})
            if response.status_code == 200:
                return True
        except requests.RequestException as exc:
            logging.warning(f"  {url} -> {exc}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=Path, help="JSON list of cited references")
    parser.add_argument("--output", type=Path, default=Path("lit_review/verification_report.json"))
    args = parser.parse_args()

    setup_logging(Path("logs") / "verify_references.log")
    refs = json.loads(args.references.read_text(encoding="utf-8"))
    logging.info(f"Verifying {len(refs)} references from {args.references}")

    report, failures, mismatches = [], [], []
    for i, ref in enumerate(refs, 1):
        doi = (ref.get("doi") or "").strip().lower()
        arxiv_id = (ref.get("arxiv_id") or "").strip()
        # Corpus records carry arXiv DOIs without a separate arxiv_id field; recover it
        # so they verify against arxiv.org rather than falling through to OpenAlex.
        if not arxiv_id and doi.startswith("10.48550/arxiv."):
            arxiv_id = doi.split("arxiv.", 1)[1]
        cited_title = ref.get("title", "")
        entry = {"key": ref.get("key"), "cited_title": cited_title, "doi": doi}

        if doi or arxiv_id:
            # arXiv DOIs are DataCite-registered, so Crossref 404s on them (DECISIONS #3).
            # Prefer arxiv.org directly; OpenAlex is the fallback for 10.48550 DOIs whose
            # arXiv id we do not carry.
            if arxiv_id:
                exists, meta = verify_via_arxiv(arxiv_id)
            elif doi.startswith("10.48550/arxiv."):
                exists, meta = verify_via_openalex(doi)
            else:
                exists, meta = verify_via_crossref(doi)
            entry["registry"] = meta.get("registry", "-")
            entry["verified"] = exists
            entry["registry_title"] = meta.get("title", "")
            entry["registry_year"] = meta.get("year")
            entry["registry_venue"] = meta.get("venue", "")

            if not exists:
                failures.append(entry)
                logging.error(f"[{i}/{len(refs)}] DOI NOT FOUND  {doi}  ({cited_title[:50]})")
            else:
                ratio = SequenceMatcher(None, normalise(cited_title),
                                        normalise(meta.get("title", ""))).ratio()
                entry["title_match"] = round(ratio, 3)
                if ratio < TITLE_MATCH_THRESHOLD:
                    mismatches.append(entry)
                    logging.error(f"[{i}/{len(refs)}] TITLE MISMATCH ({ratio:.2f})\n"
                                  f"        cited:    {cited_title}\n"
                                  f"        registry: {meta.get('title')}")
                else:
                    logging.info(f"[{i}/{len(refs)}] OK  {doi}  ({ratio:.2f})")
            time.sleep(DELAY_S)

        elif ref.get("url"):
            ok = verify_url(ref["url"])
            entry.update({"verified": ok, "registry": "URL (no DOI)", "url": ref["url"]})
            if ok:
                logging.info(f"[{i}/{len(refs)}] OK (URL)  {ref['url']}")
            else:
                failures.append(entry)
                logging.error(f"[{i}/{len(refs)}] URL UNREACHABLE  {ref['url']}")
            time.sleep(DELAY_S)

        else:
            entry.update({"verified": False, "registry": "NONE"})
            failures.append(entry)
            logging.error(f"[{i}/{len(refs)}] NO DOI AND NO URL  {cited_title[:60]}")

        report.append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "total": len(refs),
        "verified": sum(1 for e in report if e.get("verified")),
        "failures": len(failures),
        "title_mismatches": len(mismatches),
        "entries": report,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info(f"=== {sum(1 for e in report if e.get('verified'))}/{len(refs)} verified, "
                 f"{len(failures)} failures, {len(mismatches)} title mismatches ===")
    logging.info(f"Report written to {args.output}")

    if failures or mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
