#!/usr/bin/env python3
"""Render the review: resolve citation markers against references.json and append the reference list.

Markers in lit_review/review_source.md:
  [@liu2021]              -> (Liu et al., 2021)
  [@liu2021; @barr2020]   -> (Liu et al., 2021; Barr et al., 2020)
  @liu2021                -> Liu et al. (2021)            (narrative form)
Fails loudly on a key that is not in references.json, on a reference that is never cited, and on
em/en dashes or curly quotes in the written text (house style). Writes lit_review/<name>.md, and a PDF
next to it with pandoc + xelatex when --pdf is given.

Usage (from synthetic_data/, ~1 s; ~20 s with --pdf):
    .venv/Scripts/python.exe scripts/lit_render.py --pdf
"""

import argparse
import html
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "lit_review"
OUT_NAME = "synthetic_data_for_xai_evaluation_review"
# First author's (family name, initials) where the registry's name order splits the family name wrongly.
SURNAME_OVERRIDE = {"franca2024": ("de França", "F. O."), "rahnama2023": ("Akhavan Rahnama", "A. H.")}


def split_name(name: str):
    """'Family, Given' (arXiv) or 'Given Family' (Crossref/OpenAlex) -> (family, initials)."""
    name = re.sub(r"\s+\d{4}$", "", name.strip())
    if "," in name:
        family, given = [p.strip() for p in name.split(",", 1)]
    else:
        parts = name.split()
        particles = {"van", "von", "de", "der", "la", "le", "da", "del", "di"}
        i = len(parts) - 1
        while i > 0 and parts[i - 1].lower() in particles:
            i -= 1
        family, given = " ".join(parts[i:]), " ".join(parts[:i])
    initials = " ".join(f"{p[0]}." for p in re.split(r"[\s\-]+", given) if p)
    return family, initials


def first_author(ref: dict):
    return SURNAME_OVERRIDE.get(ref["key"]) or split_name(ref["authors"][0])


def cite_name(ref: dict) -> str:
    authors = ref["authors"]
    first = first_author(ref)[0]
    if len(authors) == 1:
        return first
    if len(authors) == 2:
        return f"{first} and {split_name(authors[1])[0]}"
    return f"{first} et al."


def year_labels(refs: dict) -> dict:
    """Year as printed: works that would cite identically ('Blesch et al., 2023') get a, b by title, as in APA."""
    groups = {}
    for r in refs.values():
        groups.setdefault((cite_name(r), str(r["year"])), []).append(r)
    labels = {}
    for (_, year), rs in groups.items():
        for i, r in enumerate(sorted(rs, key=lambda r: r["title"].lower())):
            labels[r["key"]] = year + (chr(ord("a") + i) if len(rs) > 1 else "")
    return labels


def reference_entry(ref: dict, year: str) -> str:
    names = [f"{f}, {i}".strip(", ") for f, i in [first_author(ref)] + [split_name(a) for a in ref["authors"][1:3]]]
    who = ", ".join(names) + (", et al." if len(ref["authors"]) > 3 else "")
    venue = html.unescape(ref.get("venue_note") or ref.get("venue") or "")
    if ref.get("arxiv_id") or (ref.get("doi") or "").startswith("10.48550/arxiv."):
        aid = ref.get("arxiv_id") or ref["doi"].split("arxiv.", 1)[1]
        link = f"https://arxiv.org/abs/{aid}"
        if venue in ("", "arXiv preprint", "arXiv (Cornell University)"):
            venue = f"arXiv preprint arXiv:{aid}"
    else:
        link = f"https://doi.org/{ref['doi']}" if ref.get("doi") else ref.get("url", "")
    title = html.unescape(ref["title"])
    for a, b in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("–", " - "), ("—", " - ")):
        title = title.replace(a, b)  # registry titles carry typographic punctuation; house style is ASCII
    end = "" if title.endswith(("?", "!", ".")) else "."
    return f"{who} ({year}). {title}{end} *{venue}*. <{link}>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "lit_render.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    start = time.time()
    refs = {r["key"]: r for r in json.loads((REVIEW / "references.json").read_text(encoding="utf-8"))}
    src = (REVIEW / "review_source.md").read_text(encoding="utf-8")

    bad = [(m.group(0), src.count("\n", 0, m.start()) + 1) for m in re.finditer("[–—‘’“”]", src)]
    if bad:
        raise ValueError(f"house style: dashes or curly quotes in review_source.md at lines {[l for _, l in bad][:20]}")

    used = set()

    def need(key: str) -> dict:
        if key not in refs:
            raise KeyError(f"unknown citation key @{key}")
        used.add(key)
        return refs[key]

    year = year_labels(refs)

    def paren(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        groups = {}  # same authors cited together print once: (Mamalakis et al., 2022a, 2022b)
        for k in keys:
            groups.setdefault(cite_name(need(k)), []).append(year[k])
        return "(" + "; ".join(f"{name}, {', '.join(sorted(ys))}" for name, ys in groups.items()) + ")"

    body = re.sub(r"\[(@[a-z0-9]+(?:\s*;\s*@[a-z0-9]+)*)\]", paren, src)
    body = re.sub(r"(?<![\w/])@([a-z]+\d{4}[a-z]?)\b", lambda m: f"{cite_name(need(m.group(1)))} ({year[m.group(1)]})", body)
    unused = sorted(set(refs) - used)
    if unused:
        raise ValueError(f"references never cited in the text: {unused}")

    def sort_key(r):
        return (first_author(r)[0].lower().replace("de ", ""), year[r["key"]])

    ref_list = "\n\n".join(reference_entry(r, year[r["key"]]) for r in sorted(refs.values(), key=sort_key))
    out = REVIEW / f"{OUT_NAME}.md"
    out.write_text(body.rstrip() + "\n\n## References\n\n" + ref_list + "\n", encoding="utf-8")
    logging.info(f"Wrote {out}: {len(used)} references cited, {len(body.split())} words in the body")

    if args.pdf:
        pdf = out.with_suffix(".pdf")
        cmd = ["pandoc", out.name, "-o", pdf.name, "--pdf-engine=xelatex", "--toc",
               "-V", "geometry:margin=2.2cm", "-V", "mainfont=Calibri", "-V", "fontsize=10.5pt",
               "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
               "--resource-path", str(REVIEW)]
        res = subprocess.run(cmd, cwd=REVIEW, capture_output=True, text=True, encoding="utf-8")
        if res.returncode != 0:
            raise RuntimeError(f"pandoc failed:\n{res.stderr}")
        logging.info(f"Wrote {pdf} ({pdf.stat().st_size / 1e3:.0f} kB)")
    logging.info(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
