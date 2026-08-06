#!/usr/bin/env python3
"""Regenerate papers/tabpfn/README.md from manifest.json.

The per-category tables are derived from the manifest so they cannot drift out of sync with
what is actually on disk. Prose sections live in this file; re-run after adding papers.

Usage:
    python scripts/non_essential/render_tabpfn_readme.py
"""

import json
import logging
import sys
from pathlib import Path

CATEGORIES = {
    "core": "Core model line",
    "analysis": "Analysis: what the model actually does",
    "scaling": "Scaling and efficiency",
    "extension": "Extensions and adaptations",
    "comparison": "Comparison models (other tabular foundation models)",
    "theory": "PFN theory: what the model is supposed to be computing",
}

HEADER = """# TabPFN papers

{n} papers on TabPFN and tabular foundation models, retrieved 2026-08-06.
{pages} MB across {n} PDFs. Every PDF verified: title matched against the registry before
saving, `%PDF` magic number checked, and confirmed readable.

`manifest.json` carries full metadata per paper - title, authors, date, DOI, arXiv abs URL,
byte count, and the title-match ratio that admitted it.

## Scope

The **core model line plus methodologically significant work**, plus two comparison models
(TabICL, TabDPT) for cross-architecture work, and the PFN theory literature on what these
models are supposed to be computing. Deliberately excludes the ~40 narrow domain
applications DBLP also indexes under "TabPFN" (IoT intrusion detection, drug discovery, crop
yield, insurance pricing, Alzheimer's prediction), which apply TabPFN rather than developing
or analysing it. DBLP reports 69 papers mentioning TabPFN in total.

**Search-scope warning, twice learned.** The first pass keyed on "TabPFN" in the title and
missed two significant mechanistic papers that say "Tabular Foundation Models" instead
(2605.21288, 2605.06510). A third pass searching the underlying concept, **"prior-data fitted
networks"**, surfaced a further ~25 DBLP hits including the theory papers that define what
TabPFN is supposed to be computing (2305.11097, 2410.01565). Product name, architecture name
and concept name each retrieve a different slice. Search all three.
"""

FOOTER = """
## Provenance and how to reproduce

```bash
python scripts/download_tabpfn_papers.py --dry-run   # verify every title, download nothing
python scripts/download_tabpfn_papers.py             # download; resumable
python scripts/non_essential/render_tabpfn_readme.py # regenerate this file
```

arXiv identifiers come from DBLP's `ee` field or from OpenAlex, not from recall. The exception
is `2207.01848` (the original TabPFN paper): DBLP holds only the ICLR'23 entry with no arXiv
link, so that identifier rests purely on a title match against the arXiv abs page.

The script fetches each abs page and compares `citation_title` against the expected title
before writing anything, so a wrong identifier cannot save the wrong paper under a
right-looking name. Two papers tripped this guard because they were **retitled between arXiv
versions** while DBLP kept the original title:

| arXiv | DBLP title | Current arXiv title |
|---|---|---|
| 2501.02945 | The Tabular Foundation Model TabPFN Outperforms Specialized Time Series... | From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting |
| 2602.04596 | A principled framework for uncertainty decomposition in TabPFN | Uncertainty Decomposition for Bayes-Filtered Transformers via Bayesian Predictive Inference |

Both were confirmed to be the intended work (abstracts still concern TabPFN; 2501.02945 is
TabPFN-TS) before the expected titles were updated. Neither was accepted automatically.

The Nature paper has no arXiv preprint. It is resolved via Crossref
(`10.1038/s41586-024-08328-6`) and its PDF is downloadable because the article is open access.

## Notes

- The arXiv **API** returns HTTP 429 to this network; the **website** serves `citation_*` meta
  tags and `/pdf/` downloads normally, which is what this uses.
- Downloads are rate-limited to one request per 3s and the script is resumable.
- Research directions built on this collection:
  [`research_ideas/tabpfn_mech_interp.md`](../../research_ideas/tabpfn_mech_interp.md).
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    manifest = json.loads(Path("papers/tabpfn/manifest.json").read_text(encoding="utf-8"))
    papers = manifest["papers"]

    unknown = {p["category"] for p in papers} - set(CATEGORIES)
    if unknown:
        raise ValueError(f"manifest has categories missing from CATEGORIES: {unknown}")

    parts = [HEADER.format(n=manifest["n_pdfs"],
                           pages=f'{manifest["total_bytes"]/1048576:.1f}')]
    for cat, heading in CATEGORIES.items():
        group = sorted((p for p in papers if p["category"] == cat),
                       key=lambda r: (r.get("date") or str(r.get("year") or "")))
        if not group:
            continue
        parts.append(f"\n### {heading} ({len(group)})\n")
        parts.append("| Year | ID | First author | Title |")
        parts.append("|---|---|---|---|")
        for p in group:
            year = (p.get("date") or "")[:4] or str(p.get("year") or "")
            ident = p.get("arxiv_id") or "Nature"
            link = f"[{ident}]({p.get('abs_url') or p.get('url')})"
            authors = p.get("authors") or []
            first = authors[0].split(",")[0].strip().split()[-1] if authors else "?"
            parts.append(f"| {year} | {link} | {first} et al. | {p['title']} |")

    out = Path("papers/tabpfn/README.md")
    out.write_text("\n".join(parts) + "\n" + FOOTER, encoding="utf-8")
    logging.info(f"Wrote {out} ({manifest['n_pdfs']} papers, "
                 f"{len({p['category'] for p in papers})} categories)")


if __name__ == "__main__":
    main()
