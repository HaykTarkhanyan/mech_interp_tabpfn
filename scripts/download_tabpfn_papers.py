#!/usr/bin/env python3
"""Download the TabPFN paper set to papers/tabpfn/ with verified metadata.

arXiv identifiers below come from DBLP's `ee` field (authoritative) except where noted.
Before any PDF is written, the arXiv abs page is fetched and its citation_title compared
against the expected title; a mismatch aborts that paper rather than saving a wrong file
under a right-looking name. Downloaded bytes are checked for the %PDF magic number and a
plausible size, so an error page saved as .pdf fails loudly.

The arXiv *API* is HTTP 429 to this network; the *website* serves citation_* meta tags and
/pdf/ downloads normally, which is what this uses.

Usage:
    python scripts/download_tabpfn_papers.py
    python scripts/download_tabpfn_papers.py --dry-run     # verify titles, download nothing
"""

import argparse
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

USER_AGENT = "tabpfn-paper-collection/1.0 (mailto:tarkhanyan02@gmail.com)"
TIMEOUT_S = 60
DELAY_S = 3.0
MAX_RETRIES = 4
TITLE_MATCH_THRESHOLD = 0.80
MIN_PDF_BYTES = 20_000
OUTDIR = Path("papers/tabpfn")

# (arxiv_id, expected title prefix, category). Expected titles are DBLP's, which truncates,
# so matching is prefix-based. 2207.01848 is the one id NOT supplied by DBLP - DBLP has no
# arXiv link for the ICLR'23 paper - so its title check is the load-bearing one.
PAPERS = [
    # --- core model line ---
    ("2207.01848", "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second", "core"),
    ("2112.10510", "Transformers Can Do Bayesian Inference", "core"),
    ("2511.08667", "TabPFN-2.5: Advancing the State of the Art in Tabular Foundation", "core"),
    ("2605.13986", "TabPFN-3", "core"),
    ("2511.03634", "nanoTabPFN: A Lightweight and Educational Reimplementation", "core"),

    # --- analysis: what the model actually does ---
    ("2502.08978", "What exactly has TabPFN learned to do", "analysis"),
    ("2502.17361", "A Closer Look at TabPFN v2", "analysis"),
    ("2505.16226", "Realistic Evaluation of TabPFN v2 in Open Environments", "analysis"),
    ("2505.20003", "TabPFN: One Model to Rule Them All", "analysis"),
    ("2511.07236", "Does TabPFN Understand Causal Structures", "analysis"),
    ("2606.12917", "Where Computation Lives Inside TabPFN", "analysis"),
    # Added 2026-08-06 on a second search pass. Both were missed by the original sweep
    # because it keyed on "TabPFN" in the title and these say "Tabular Foundation Models".
    ("2605.21288", "A Mechanistic Study of Tabular Foundation Models", "analysis"),
    ("2605.06510", "Is One Layer Enough? Understanding Inference Dynamics in Tabular", "analysis"),

    # --- comparison models, for cross-architecture universality work ---
    ("2502.05564", "TabICL: A Tabular Foundation Model for In-Context Learning on Large", "comparison"),
    ("2410.18164", "TabDPT: Scaling Tabular Foundation Models", "comparison"),
    ("2601.08181", "TabPFN Through The Looking Glass", "analysis"),
    ("2511.18278", "From Tables to Signals", "analysis"),
    # Retitled on arXiv since DBLP indexed it ("A principled framework for uncertainty
    # decomposition in TabPFN"). Confirmed same work: abstract still concerns TabPFN, which
    # is a Bayes-filtered transformer. Expected title updated to what arXiv now serves.
    ("2602.04596", "Uncertainty Decomposition for Bayes-Filtered Transformers", "analysis"),

    # --- scaling and efficiency ---
    ("2311.10609", "Scaling TabPFN: Sketching and Feature Selection", "scaling"),
    ("2502.02527", "TabPFN Unleashed", "scaling"),
    ("2402.06971", "In-Context Data Distillation with TabPFN", "scaling"),
    ("2509.00326", "Chunked TabPFN", "scaling"),
    ("2510.06162", "TabPFN-Wide: Continued Pre-Training for Extreme Feature Counts", "scaling"),
    ("2402.11137", "TuneTables: Context Optimization for Scalable Prior-Data Fitted", "scaling"),

    # --- extensions and adaptations ---
    ("2507.03971", "Real-TabPFN", "extension"),
    ("2411.10634", "Drift-Resilient TabPFN", "extension"),
    ("2403.10923", "Interpretable Machine Learning for TabPFN", "extension"),
    ("2406.05216", "TabPFGen", "extension"),
    ("2406.06891", "Tokenize features, enhancing tables", "extension"),
    # This is TabPFN-TS. Retitled for TMLR; DBLP holds the original preprint title
    # ("The Tabular Foundation Model TabPFN Outperforms Specialized Time Series...").
    ("2501.02945", "From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting", "extension"),
    ("2605.20234", "TabPFN-MT", "extension"),
    ("2606.04876", "Towards Pretraining Text Encoders for TabPFN", "extension"),
    ("2603.10254", "Improving TabPFN", "extension"),
]

# Published in Nature, no arXiv preprint. Handled separately: resolved via DOI, and the PDF
# is only saved if the publisher serves one openly.
DOI_PAPERS = [
    ("10.1038/s41586-024-08328-6",
     "Accurate predictions on small data with a tabular foundation model", "core"),
]


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(exist_ok=True)
    handler = logging.StreamHandler(sys.stdout)
    handler.setStream(open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), handler],
    )


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text)


def title_matches(expected: str, actual: str) -> float:
    exp, act = normalise(expected), normalise(actual)
    if not exp or not act:
        return 0.0
    if act.startswith(exp) or exp.startswith(act):      # DBLP truncates titles
        return 1.0
    return SequenceMatcher(None, exp, act).ratio()


def get(url: str, accept: Optional[str] = None) -> Optional[requests.Response]:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=TIMEOUT_S, headers=headers, allow_redirects=True)
        except requests.RequestException as exc:
            logging.warning(f"  {url} attempt {attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(4.0 * attempt)
            continue
        if response.status_code == 200:
            return response
        if response.status_code in (403, 404):
            return None
        if response.status_code in (429, 500, 502, 503, 504):
            backoff = 4.0 * (2 ** attempt)
            logging.warning(f"  {url} -> {response.status_code}, retry in {backoff:.0f}s")
            time.sleep(backoff)
            continue
        response.raise_for_status()
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {url}")


def arxiv_metadata(arxiv_id: str) -> Optional[Dict]:
    response = get(f"https://arxiv.org/abs/{arxiv_id}")
    if response is None:
        return None
    page = response.text          # not `html`: that name is the stdlib module used below
    grab = lambda p: re.findall(p, page, re.I)
    titles = grab(r'name=["\']citation_title["\'][^>]+content=["\']([^"\']+)')
    if not titles:
        return None
    # arXiv's meta tags carry HTML entities: "Improving TabPFN&#39;s Synthetic Data..."
    # Left undecoded these reach both the manifest and the generated filename.
    return {
        "title": html.unescape(titles[0]).strip(),
        "authors": [html.unescape(a) for a in
                    grab(r'name=["\']citation_author["\'][^>]+content=["\']([^"\']+)')],
        "date": (grab(r'name=["\']citation_date["\'][^>]+content=["\']([^"\']+)') or [""])[0],
        "doi": f"10.48550/arXiv.{arxiv_id}",
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def slugify(title: str, arxiv_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalise_spaces(title).lower()).strip("-")[:70]
    return f"{arxiv_id}_{slug}.pdf"


def normalise_spaces(text: str) -> str:
    return " ".join((text or "").split())


def save_pdf(url: str, path: Path) -> int:
    """Download and validate. Returns byte count; raises if the payload is not a PDF."""
    response = get(url, accept="application/pdf")
    if response is None:
        raise RuntimeError(f"PDF unavailable (403/404): {url}")
    body = response.content
    if not body.startswith(b"%PDF"):
        raise RuntimeError(f"not a PDF (magic={body[:8]!r}, {len(body)} bytes): {url}")
    if len(body) < MIN_PDF_BYTES:
        raise RuntimeError(f"implausibly small PDF ({len(body)} bytes): {url}")
    path.write_bytes(body)
    return len(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="verify titles, download nothing")
    args = parser.parse_args()

    setup_logging(Path("logs") / "download_tabpfn_papers.log")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTDIR / "manifest.json"
    existing = {m["arxiv_id"]: m for m in
                json.loads(manifest_path.read_text(encoding="utf-8")).get("papers", [])} \
        if manifest_path.exists() else {}
    if existing:
        logging.info(f"Resuming: {len(existing)} papers already in manifest")

    records: List[Dict] = []
    failures: List[Dict] = []

    for i, (arxiv_id, expected, category) in enumerate(PAPERS, 1):
        prior = existing.get(arxiv_id)
        if prior and (OUTDIR / prior["filename"]).exists():
            records.append(prior)
            logging.info(f"[{i}/{len(PAPERS)}] skip (have) {prior['filename']}")
            continue

        meta = arxiv_metadata(arxiv_id)
        if meta is None:
            failures.append({"arxiv_id": arxiv_id, "expected_title": expected,
                             "error": "abs page unavailable or no citation_title"})
            logging.error(f"[{i}/{len(PAPERS)}] {arxiv_id} UNAVAILABLE")
            time.sleep(DELAY_S)
            continue

        ratio = title_matches(expected, meta["title"])
        if ratio < TITLE_MATCH_THRESHOLD:
            failures.append({"arxiv_id": arxiv_id, "expected_title": expected,
                             "actual_title": meta["title"], "match_ratio": round(ratio, 3),
                             "error": "title mismatch - refusing to save"})
            logging.error(f"[{i}/{len(PAPERS)}] {arxiv_id} TITLE MISMATCH ({ratio:.2f})\n"
                          f"        expected: {expected}\n"
                          f"        actual:   {meta['title']}")
            time.sleep(DELAY_S)
            continue

        filename = slugify(meta["title"], arxiv_id)
        if args.dry_run:
            logging.info(f"[{i}/{len(PAPERS)}] OK ({ratio:.2f}) would save {filename}")
            time.sleep(DELAY_S)
            continue

        try:
            size = save_pdf(meta["pdf_url"], OUTDIR / filename)
        except RuntimeError as exc:
            failures.append({"arxiv_id": arxiv_id, "title": meta["title"], "error": str(exc)})
            logging.error(f"[{i}/{len(PAPERS)}] {arxiv_id} DOWNLOAD FAILED: {exc}")
            time.sleep(DELAY_S)
            continue

        records.append({**meta, "arxiv_id": arxiv_id, "category": category,
                        "filename": filename, "bytes": size,
                        "title_match_ratio": round(ratio, 3)})
        logging.info(f"[{i}/{len(PAPERS)}] OK ({ratio:.2f}) {size/1_048_576:.1f} MB  {filename}")
        time.sleep(DELAY_S)

    # Nature paper: record the citation regardless; save the PDF only if served openly.
    for doi, title, category in DOI_PAPERS:
        entry = {"arxiv_id": None, "doi": doi, "title": title, "category": category,
                 "url": f"https://doi.org/{doi}", "filename": None,
                 "note": "Published in Nature; no arXiv preprint"}
        response = get(f"https://api.crossref.org/works/{doi}")
        if response is not None:
            msg = response.json()["message"]
            registry_title = (msg.get("title") or [""])[0]
            ratio = title_matches(title, registry_title)
            entry.update({
                "registry_title": registry_title,
                "title_match_ratio": round(ratio, 3),
                "authors": [" ".join(filter(None, [a.get("given"), a.get("family")]))
                            for a in (msg.get("author") or [])],
                "venue": (msg.get("container-title") or [""])[0],
                "year": ((msg.get("issued") or {}).get("date-parts") or [[None]])[0][0],
            })
            if ratio < TITLE_MATCH_THRESHOLD:
                failures.append({"doi": doi, "error": f"title mismatch ({ratio:.2f})"})
                logging.error(f"DOI {doi} TITLE MISMATCH: {registry_title}")
                continue
            logging.info(f"DOI  OK ({ratio:.2f}) {registry_title[:60]}")
        else:
            logging.error(f"DOI {doi} not resolvable via Crossref")

        if not args.dry_run:
            pdf_url = "https://www.nature.com/articles/s41586-024-08328-6.pdf"
            try:
                filename = slugify(title, "nature2025")
                size = save_pdf(pdf_url, OUTDIR / filename)
                entry.update({"filename": filename, "bytes": size, "pdf_url": pdf_url})
                logging.info(f"DOI  downloaded {size/1_048_576:.1f} MB  {filename}")
            except RuntimeError as exc:
                entry["pdf_unavailable"] = str(exc)
                logging.warning(f"DOI  PDF not openly available, citation recorded only: {exc}")
        records.append(entry)

    if args.dry_run:
        logging.info(f"DRY RUN complete. {len(failures)} would fail.")
        for f in failures:
            logging.error(f"  {f}")
        return

    manifest_path.write_text(json.dumps({
        "collection": "TabPFN papers",
        "retrieved": time.strftime("%Y-%m-%d"),
        "source": "arxiv.org (identifiers from DBLP); Nature paper via Crossref",
        "n_papers": len(records),
        "n_pdfs": sum(1 for r in records if r.get("filename")),
        "total_bytes": sum(r.get("bytes", 0) for r in records),
        "failures": failures,
        "papers": sorted(records, key=lambda r: (r["category"], r.get("arxiv_id") or "")),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    total_mb = sum(r.get("bytes", 0) for r in records) / 1_048_576
    logging.info(f"=== {sum(1 for r in records if r.get('filename'))} PDFs, {total_mb:.1f} MB, "
                 f"{len(failures)} failures ===")
    for f in failures:
        logging.error(f"  FAILURE: {f}")
    logging.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
