#!/usr/bin/env python3
"""Deduplicate and relevance-screen the raw search results (PRISMA identification -> screening).

Duplicates are MERGED field by field, not dropped: Crossref brings venues, OpenAlex abstracts and
citation counts, arXiv abstracts for preprints. Each record keeps the databases and themes that
found it, so overlap and the contribution of citation chaining are reportable.

The topic spans several communities, so the screen is a set of rules, one per strand of the
review; a record passes if any rule fires. Records that still have no abstract after the
backfill are screened by a looser title-only rule and flagged, so missing metadata does not
silently exclude them.

Usage (from synthetic_data/, offline, ~2 s):
    .venv/Scripts/python.exe scripts/lit_screen.py
"""

import json
import logging
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "lit_review" / "data" / "raw"
PROC = ROOT / "lit_review" / "data" / "processed"

TERMS = {
    "xai": [r"explanation", r"explainab", r"interpretab", r"feature attribution", r"attribution method",
            r"saliency", r"shapley", r"shap\b", r"lime\b", r"feature importance", r"variable importance",
            r"partial dependence", r"accumulated local effect", r"feature effect", r"xai\b"],
    "eval": [r"ground.truth", r"benchmark", r"evaluat", r"faithful", r"fidelity", r"correctness",
             r"sanity check", r"validat"],
    "synthetic": [r"synthetic", r"semi.synthetic", r"simulat", r"generative model", r"data.generating",
                  r"plasmode", r"monte carlo", r"known (?:function|relationship|mechanism)"],
    "semi": [r"semi.synthetic", r"plasmode", r"empirical monte carlo", r"realistic simulation",
             r"simulation design", r"design of (?:monte carlo|simulation)", r"synthetic data generat",
             r"data.generating process"],
    "compare": [r"benchmark", r"evaluat", r"comparison", r"estimator", r"simulation stud"],
    "tabular": [r"tabular", r"mixed.type data"],
    "generate": [r"synthe", r"generat", r"density estimat"],
    "quality": [r"fidelity", r"utility", r"evaluat", r"quality", r"benchmark", r"metric"],
    "sr_quality": [r"benchmark", r"ground.truth", r"interpretab", r"real.world", r"feynman"],
    "dgp": [r"data.generating process", r"true to the (?:model|data)", r"conditional (?:feature importance|"
            r"sampling|permutation|expectation)", r"knockoff", r"on.manifold", r"off.manifold", r"interventional"],
}
# Application domains: an XAI paper about, say, sepsis prediction is out of scope unless it is a
# benchmark. Not applied to the semi-synthetic rule - plasmode work lives in pharmacoepidemiology.
EXCLUDE_TITLE = [r"medical", r"clinical", r"cancer", r"patient", r"covid", r"diagnos", r"hospital", r"\becg\b",
                 r"\beeg\b", r"\bmri\b", r"radiolog", r"histopath", r"alzheimer", r"diabet", r"sepsis", r"financ",
                 r"credit", r"fraud", r"intrusion", r"malware", r"agricultur", r"\bcrop", r"\bsoil", r"traffic",
                 r"\bwind\b", r"solar", r"battery", r"manufactur", r"\bstudent", r"\bemr\b", r"\behr\b"]
SHORTLIST_SIZE = 700


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / "lit_screen.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def norm(title: str) -> str:
    text = unicodedata.normalize("NFKD", (title or "").lower())
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in text if not unicodedata.combining(c)))


def hits(text: str, group: str) -> List[str]:
    return [t for t in TERMS[group] if re.search(rf"\b{t}", text)]


def merge(new: Dict, old: Dict) -> Dict:
    out = dict(old)
    for f in ("abstract", "venue", "doi", "arxiv_id", "url", "authors", "openalex_id"):
        if len(new.get(f) or "") > len(out.get(f) or ""):
            out[f] = new[f]
    out["citations"] = max(new.get("citations") or 0, out.get("citations") or 0) or out.get("citations")
    out["year"] = out.get("year") or new.get("year")
    out["found_by"] = sorted(set(out["found_by"]) | set(new["found_by"]))
    out["themes"] = sorted(set(out["themes"]) | set(new["themes"]))
    return out


def deduplicate(records: List[Dict]) -> tuple:
    """One canonical list; DOI, OpenAlex ID and normalised title all alias to the same index.

    Returns (records, number dropped for an empty or near-empty title), so drops are not counted as duplicates.
    """
    canon: List[Dict] = []
    dropped = 0
    idx_of: Dict[str, int] = {}
    for raw in records:
        rec = dict(raw)
        rec["found_by"] = [rec.pop("source")]
        rec["themes"] = [rec.pop("theme")]
        tkey = norm(rec.get("title"))
        if len(tkey) < 10:  # empty or "introduction"-style titles cannot be matched or screened
            dropped += 1
            continue
        keys = [f"title:{tkey}"]
        if rec.get("doi"):
            keys.append(f"doi:{rec['doi']}")
        if rec.get("openalex_id"):
            keys.append(f"oa:{rec['openalex_id']}")
        idx = next((idx_of[k] for k in keys if k in idx_of), None)
        if idx is None:
            idx = len(canon)
            canon.append(rec)
        else:
            canon[idx] = merge(rec, canon[idx])
        for k in keys + [f"doi:{canon[idx]['doi']}" if canon[idx].get("doi") else ""]:
            if k:
                idx_of[k] = idx
    return canon, dropped


def screen(rec: Dict) -> Dict:
    title = (rec.get("title") or "").lower()
    has_abstract = bool((rec.get("abstract") or "").strip())
    text = f"{title} {rec.get('abstract') or ''}".lower() if has_abstract else title
    g = {k: hits(text, k) for k in TERMS}
    domain = [t for t in EXCLUDE_TITLE if re.search(t, title)]
    rules = []
    if has_abstract:
        if g["xai"] and g["eval"] and g["synthetic"]:
            rules.append("xai_eval_synthetic")
        if g["xai"] and "ground.truth" in g["eval"]:
            rules.append("xai_ground_truth")
        if g["semi"] and g["compare"]:
            rules.append("semi_synthetic_methodology")
        if re.search(r"adversarial random forest", text) or (g["tabular"] and g["generate"] and g["quality"]):
            rules.append("tabular_generative")
        if re.search(r"symbolic regression", text) and g["sr_quality"]:
            rules.append("symbolic_regression")
        if g["dgp"] and g["xai"]:
            rules.append("model_vs_data")
    else:  # title only: looser, flagged for manual screening
        if (g["xai"] and (g["eval"] or g["synthetic"])) or g["semi"] or "symbolic regression" in title \
                or "adversarial random forest" in title or (g["tabular"] and g["generate"]):
            rules.append("title_only")
    if domain and "semi_synthetic_methodology" not in rules and "benchmark" not in title:
        rules = []
    rec.update(rules=rules, passes=bool(rules), has_abstract=has_abstract, excluded_domain=domain,
               score=4 * len(rules) + sum(len(v) for v in g.values()),
               via_citation=any(t.startswith("cites:") for t in rec["themes"]))
    return rec


def main() -> None:
    setup_logging()
    start = time.time()
    raw = json.loads((RAW / "combined_raw.json").read_text(encoding="utf-8"))
    logging.info(f"Identification: {len(raw)} records")
    unique, no_title = deduplicate(raw)
    logging.info(f"After deduplication: {len(unique)} ({len(raw) - len(unique) - no_title} merged, {no_title} dropped without a title)")
    scored = [screen(r) for r in unique]
    passed = sorted((r for r in scored if r["passes"]), key=lambda r: (r["score"], r.get("citations") or 0),
                    reverse=True)
    shortlist = passed[:SHORTLIST_SIZE]
    rule_counts = Counter(rule for r in passed for rule in r["rules"])
    overlap = Counter(len([s for s in r["found_by"] if s != "OpenAlex-citing"]) for r in scored)
    by_db = Counter(s for r in scored for s in r["found_by"])
    only_citing = sum(1 for r in passed if r["found_by"] == ["OpenAlex-citing"])
    logging.info(f"Passed relevance screen: {len(passed)}; rules: {dict(rule_counts)}")
    logging.info(f"Passed records found ONLY by citation chaining: {only_citing}")
    logging.info(f"Records per source (after merge): {dict(by_db)}")

    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "screened_all.json").write_text(json.dumps(scored, indent=1, ensure_ascii=False), encoding="utf-8")
    (PROC / "shortlist.json").write_text(json.dumps(shortlist, indent=1, ensure_ascii=False), encoding="utf-8")
    # Compact sheet for manual title/abstract screening: one line per record.
    sheet = [f"{i}\t{r.get('year') or '----'}\t{r.get('citations') or 0}\t{','.join(r['rules'])}\t"
             f"{'C' if r['via_citation'] else '-'}\t{r['title']}" for i, r in enumerate(shortlist)]
    (PROC / "screening_sheet.tsv").write_text("idx\tyear\tcites\trules\tchained\ttitle\n" + "\n".join(sheet),
                                              encoding="utf-8")
    (PROC / "prisma_counts.json").write_text(json.dumps({
        "identification": len(raw), "after_deduplication": len(unique),
        "duplicates_merged": len(raw) - len(unique) - no_title, "dropped_no_title": no_title, "after_relevance_screen": len(passed),
        "excluded_by_screen": len(scored) - len(passed),
        "excluded_domain_title": sum(1 for r in scored if r["excluded_domain"] and not r["passes"]),
        "passed_without_abstract": sum(1 for r in passed if not r["has_abstract"]),
        "passed_only_via_citation_chaining": only_citing,
        "rules_fired": dict(rule_counts), "shortlist_size": len(shortlist),
        "records_per_source": dict(by_db),
        "found_by_n_keyword_databases": {str(k): v for k, v in sorted(overlap.items())},
    }, indent=1), encoding="utf-8")
    logging.info(f"Shortlist {len(shortlist)} -> {PROC / 'screening_sheet.tsv'} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
