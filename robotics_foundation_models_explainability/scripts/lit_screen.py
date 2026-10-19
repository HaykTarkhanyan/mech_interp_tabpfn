#!/usr/bin/env python3
"""Merge, deduplicate and rule-screen the search records; write a sheet for manual screening.

Inputs (lit_review/data/raw/):
  combined_raw.json        keyword search (+ chaining when it ran), from lit_search.py
  web_sweep.json           web discovery leg (subagent, every arXiv id title-checked); passes the screen
                           automatically because each item was already judged relevant by hand
  pdf_reference_ids.json   optional: arXiv ids cited by the downloaded core papers (lit_refchain.py)

Rule screen: a record passes with >=1 robot-context term AND >=1 interpretability term in title+abstract.
Records with no abstract are screened on title only and flagged, never silently dropped
(see mech_interp/_learnings/2026-08-06-0035_screening-on-missing-abstracts...).

Outputs (lit_review/data/processed/): deduplicated.json, screened.json, screening_sheet.tsv, prisma_counts.json

Usage (from robotics_foundation_models_explainability/), ~2 s, offline:
    .venv/Scripts/python.exe scripts/lit_screen.py
"""

import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "lit_review" / "data" / "raw"
PROC = ROOT / "lit_review" / "data" / "processed"

ROBOT_TERMS = [r"vision-language-action", r"\bvlas?\b", r"\brobot", r"manipulation", r"visuomotor", r"embodied",
               r"openvla", r"\bocto\b", r"\bpi0", r"\bπ0", r"\brt-[12]\b", r"gr00t", r"libero", r"\bpolicy\b",
               r"\bpolicies\b", r"autonomous driving", r"world model", r"\bagents?\b", r"imitation learning",
               r"reinforcement learning", r"grasp"]
# Strong terms are specific to model-internals analysis and pass on their own. Medium terms are common in
# passing ("an interpretable policy", "inference-time steering"), so they need two distinct hits or one in
# the title. Measured on the first run: 456 of 669 rule passes had a single term, mostly noise.
STRONG_TERMS = [r"mechanistic", r"sparse autoencoder", r"\bsaes?\b", r"linear prob(e|es|ing)", r"activation patching",
                r"activation steering", r"steering vector", r"logit lens", r"\bneurons?\b", r"attention heads?",
                r"dictionary learning", r"causal (tracing|mediation)", r"representation(s)? analysis",
                r"internal representations?", r"internal (features|activations|states)", r"\bcircuits?\b"]
MEDIUM_TERMS = [r"interpretab", r"explainab", r"explanation", r"\bprob(e|es|ing)\b", r"steer", r"attribution",
                r"saliency", r"attention (map|pattern)", r"failure (prediction|detection|explanation)", r"faithful",
                r"counterfactual", r"concept", r"hidden state", r"introspect", r"causal (intervention|analysis)"]
EXCLUDE = re.compile(r"quasar|radio (source|galax|telescope)|galax(y|ies)|pulsar|very large array|astrophys", re.I)


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / "lit_screen.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def norm(title: str) -> str:
    text = unicodedata.normalize("NFKD", (title or "").lower())
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in text if not unicodedata.combining(c)))


def hits(text: str, terms: List[str]) -> List[str]:
    return [t for t in terms if re.search(t, text, re.I)]


def web_records() -> List[Dict]:
    path = RAW / "web_sweep.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; the web sweep is part of the method, not optional")
    out = []
    for it in json.loads(path.read_text(encoding="utf-8"))["items"]:
        if it["kind"] == "workshop" and not it.get("arxiv_id") and it["title"].lower().startswith(
                ("workshop", "the science of", "robopad", "grounded and faithful")):
            continue  # venue entries, not works
        out.append({"title": it["title"], "authors": "", "year": str(it.get("year") or ""),
                    "abstract": it.get("why_relevant", ""), "arxiv_id": it.get("arxiv_id") or "", "doi": "",
                    "url": it.get("url", ""), "venue": it.get("venue") or "", "source": "web-sweep",
                    "citations": None, "theme": f"web:{it['tier']}", "web_verified": it.get("verified")})
    return out


def merge(a: Dict, b: Dict) -> Dict:
    for k in ("abstract", "authors", "venue", "doi", "arxiv_id", "year"):
        if not a.get(k) and b.get(k):
            a[k] = b[k]
    if b["source"] != "web-sweep" and len(b.get("abstract", "")) > len(a.get("abstract", "")) \
            and a["source"] == "web-sweep":
        a["abstract"] = b["abstract"]  # prefer a real abstract over the sweep's one-line note
    a["sources"] = sorted(set(a["sources"]) | {b["source"]})
    a["themes"] = sorted(set(a["themes"]) | {b["theme"]})
    return a


def main() -> None:
    setup_logging()
    start = time.time()
    PROC.mkdir(parents=True, exist_ok=True)
    raw = json.loads((RAW / "combined_raw.json").read_text(encoding="utf-8"))
    web = web_records()
    refs_path = RAW / "pdf_reference_records.json"
    refchain = json.loads(refs_path.read_text(encoding="utf-8")) if refs_path.exists() else []
    logging.info(f"Loaded {len(raw)} search records, {len(web)} web-sweep works, {len(refchain)} reference-list records")

    by_key: Dict[str, Dict] = {}
    title_to_key: Dict[str, str] = {}
    for rec in web + raw + refchain:  # web first, so its manual relevance judgement survives the merge
        rec = dict(rec, sources=[rec["source"]], themes=[rec["theme"]])
        t = norm(rec["title"])
        if not t:
            continue
        key = rec["arxiv_id"] or title_to_key.get(t) or t
        if rec["arxiv_id"] and t in title_to_key and title_to_key[t] != key and title_to_key[t] in by_key:
            key = title_to_key[t]  # same title seen earlier without an arXiv id
        if key in by_key:
            by_key[key] = merge(by_key[key], rec)
        else:
            by_key[key] = rec
        title_to_key[t] = key
    dedup = list(by_key.values())
    logging.info(f"After deduplication: {len(dedup)} unique works")

    screened, no_abstract = [], 0
    for rec in dedup:
        text = f"{rec['title']} {rec['abstract']}"
        rob, strong, medium = hits(text, ROBOT_TERMS), hits(text, STRONG_TERMS), hits(text, MEDIUM_TERMS)
        in_title = hits(rec["title"], STRONG_TERMS + MEDIUM_TERMS)
        rec["robot_terms"], rec["interp_terms"] = rob, strong + medium
        rec["title_only"] = not rec["abstract"]
        no_abstract += rec["title_only"]
        if "web-sweep" in rec["sources"]:
            rec["screen"] = "pass: web sweep (judged relevant by hand)"
        elif EXCLUDE.search(text):
            rec["screen"] = "fail: astronomy (VLA = Very Large Array)"
        elif not rob:
            rec["screen"] = "fail: no robot term"
        elif strong or len(medium) >= 2 or in_title:
            rec["screen"] = "pass: rule"
        else:
            rec["screen"] = "fail: interpretability terms too weak" if medium else "fail: no interpretability term"
        if rec["screen"].startswith("pass"):
            screened.append(rec)
    logging.info(f"Rule screen: {len(screened)} pass; {no_abstract} records were screened on title only")

    screened.sort(key=lambda r: (-len(set(r["interp_terms"])) - len(set(r["robot_terms"])), r["title"]))
    (PROC / "deduplicated.json").write_text(json.dumps(dedup, indent=1, ensure_ascii=False), encoding="utf-8")
    (PROC / "screened.json").write_text(json.dumps(screened, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(PROC / "screening_sheet.tsv", "w", encoding="utf-8") as fh:
        fh.write("id\tyear\tsources\tthemes\tinterp_terms\ttitle\n")
        for r in screened:
            fh.write(f"{r['arxiv_id'] or r['url']}\t{r['year']}\t{','.join(r['sources'])}\t"
                     f"{','.join(t.split(':')[-1] for t in r['themes'])}\t{','.join(r['interp_terms'])}\t{r['title']}\n")
    counts = {"records_identified": {"search": len(raw), "web_sweep": len(web), "reference_lists": len(refchain)},
              "after_dedup": len(dedup), "screened_in": len(screened), "screened_out": len(dedup) - len(screened),
              "screened_on_title_only": no_abstract}
    (PROC / "prisma_counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    logging.info(f"Wrote {PROC} outputs in {time.time() - start:.1f}s: {counts}")


if __name__ == "__main__":
    main()
