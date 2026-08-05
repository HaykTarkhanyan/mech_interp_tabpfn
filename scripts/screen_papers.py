#!/usr/bin/env python3
"""Deduplicate and relevance-screen the raw literature search results.

Pipeline (PRISMA stages):
  identification -> merge duplicates -> automated relevance screen -> shortlist

Duplicates are MERGED, not dropped: DBLP supplies venue strings that OpenAlex leaves
blank for preprints, OpenAlex supplies citation counts and abstracts that DBLP lacks.
Each surviving record keeps the list of databases that found it, so cross-database
overlap is reportable.

Usage:
    python scripts/screen_papers.py
"""

import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Unambiguous mechanistic-interpretability vocabulary. One hit (plus context) includes.
STRONG_TERMS = [
    "mechanistic interpretability", "sparse autoencoder", "polysemantic", "monosemantic",
    "induction head", "activation patching", "causal tracing", "logit lens", "tuned lens",
    "activation steering", "steering vector", "representation engineering", "transcoder",
    "crosscoder", "attribution graph", "interchange intervention", "causal abstraction",
    "distributed alignment search", "circuit discovery", "residual stream", "grokking",
    "feature splitting", "path patching", "causal scrubbing", "automated circuit",
    "reverse.engineer", "circuit tracing", "feature steering", "toy models of",
]

# Ambiguous outside this field. Needs MIN_MEDIUM_TERMS distinct hits to include on
# their own. Deliberately excludes "interpretab"/"explainab"/"neuron"/"ablation":
# those are the general-XAI vocabulary and let medical-XAI surveys through (measured).
MEDIUM_TERMS = [
    "superposition", "dictionary learning", "circuit", "probing classifier", "probing task",
    "linear representation", "attention head", "knowledge editing", "factual association",
    "feature visualization", "linear probe", "concept vector", "latent direction",
    "world model", "feature direction", "sparse feature",
]

# At least one is required: keeps the corpus on transformer/LM territory (DECISION #1).
CONTEXT_TERMS = [
    "transformer", "language model", "\\bllm\\b", "\\bgpt\\b", "\\bbert\\b", "neural network",
    "deep learning", "attention", "\\bnlp\\b", "large language",
]

# Out-of-scope application domains (DECISION #1). Applied to the TITLE only, and only
# when the record has no STRONG term - so a genuine mech interp paper that happens to
# mention the brain or a protein model survives.
EXCLUDE_TERMS = [
    "medical", "clinical", "cancer", "patient", "diagnos", "drug", "protein", "genom",
    "molecul", "essay scoring", "quantum", "financ", "credit scoring", "recommender",
    "federated", "intrusion detection", "malware", "remote sensing", "fault detection",
    "eeg", "fmri", "neuroscience", "student", "education", "agricultur", "wireless",
]

MIN_MEDIUM_TERMS = 3     # three DISTINCT medium terms, not a weighted sum
SHORTLIST_SIZE = 250


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def normalise_title(title: str) -> str:
    """Aggressive normalisation for duplicate detection across databases."""
    text = unicodedata.normalize("NFKD", title.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text)


def count_terms(text: str, terms: List[str]) -> List[str]:
    """Return which terms matched, using word boundaries to avoid substring false hits."""
    return [t for t in terms if re.search(rf"\b{t}", text)]


def better(new: Dict, old: Dict) -> Dict:
    """Field-wise merge; prefer the more informative value for each field."""
    merged = dict(old)
    for field in ("abstract", "venue", "doi", "arxiv_id", "url", "authors"):
        if len(new.get(field) or "") > len(merged.get(field) or ""):
            merged[field] = new[field]
    if (new.get("citations") or 0) > (merged.get("citations") or 0):
        merged["citations"] = new["citations"]
    if new.get("year") and not merged.get("year"):
        merged["year"] = new["year"]
    merged["found_by"] = sorted(set(merged.get("found_by", []) + new.get("found_by", [])))
    merged["themes"] = sorted(set(merged.get("themes", []) + new.get("themes", [])))
    return merged


def deduplicate(records: List[Dict]) -> List[Dict]:
    """Merge on DOI (authoritative) and normalised title.

    Uses ONE canonical list with several keys aliasing to the same index. An earlier
    version kept two dicts of records and merged with a non-mutating helper, so the
    two indexes drifted apart and identical DOIs survived twice.
    """
    canonical: List[Dict] = []
    key_to_idx: Dict[str, int] = {}

    for raw in records:
        rec = dict(raw)
        rec["found_by"] = [rec.pop("source")]
        rec["themes"] = [rec.pop("theme")] if rec.get("theme") else []

        title_key = normalise_title(rec.get("title") or "")
        if not title_key:
            continue
        doi = (rec.get("doi") or "").strip().lower()
        keys = [f"doi:{doi}"] if doi else []
        keys.append(f"title:{title_key}")

        idx = next((key_to_idx[k] for k in keys if k in key_to_idx), None)
        if idx is None:
            idx = len(canonical)
            canonical.append(rec)
        else:
            canonical[idx] = better(rec, canonical[idx])

        for key in keys:                 # register every alias, including newly learned DOIs
            key_to_idx[key] = idx

    return canonical


def score(record: Dict) -> Dict:
    title = (record.get("title") or "").lower()
    text = f"{title} {record.get('abstract', '')}".lower()
    strong = count_terms(text, STRONG_TERMS)
    medium = count_terms(text, MEDIUM_TERMS)
    context = count_terms(text, CONTEXT_TERMS)
    excluded_by = count_terms(title, EXCLUDE_TERMS) if not strong else []

    # A STRONG term is already unambiguous mech-interp vocabulary, so it does not need a
    # CONTEXT word to corroborate it. Requiring context rejected DBLP records like
    # "Transcoders Beat Sparse Autoencoders for Interpretability" - two strong terms, but
    # no "transformer"/"language model" in the title and no abstract to supply one.
    # The context gate exists only to disambiguate the MEDIUM vocabulary.
    included = (
        (bool(strong) or (len(medium) >= MIN_MEDIUM_TERMS and bool(context)))
        and not excluded_by
    )

    record["relevance_score"] = 3 * len(strong) + 2 * len(medium)
    record["matched_strong"] = strong
    record["matched_medium"] = medium
    record["excluded_by"] = excluded_by
    record["passes_screen"] = included
    return record


ALLOW_MISSING = "--allow-missing-supplementary" in sys.argv


def main() -> None:
    setup_logging(Path("logs") / "screen_papers.log")

    raw_path = Path("data/raw/combined_raw.json")
    records = json.loads(raw_path.read_text(encoding="utf-8"))
    logging.info(f"Identification (systematic search): {len(records)} records from {raw_path}")

    supplementary_status = "complete"
    seed_path = Path("data/raw/seed_raw.json")
    seeds = []
    if seed_path.exists():
        seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    elif Path("data/processed/seed_cache.json").exists():
        seeds = json.loads(Path("data/processed/seed_cache.json").read_text(encoding="utf-8")).get("found", [])
        supplementary_status = "partial (seed search still running)"
    elif ALLOW_MISSING:
        supplementary_status = "SKIPPED - no seed search results available"
        logging.warning("No seed results found. Running WITHOUT the targeted seed stage; "
                        "this is recorded in prisma_counts.json as supplementary_status.")
    else:
        raise FileNotFoundError(f"{seed_path} missing - run scripts/seed_and_enrich.py first, "
                                f"or pass --allow-missing-supplementary")
    for seed in seeds:
        seed.setdefault("theme", "seed")
        seed.setdefault("source", "OpenAlex-seed")
    records += seeds
    logging.info(f"Identification (targeted seed search): {len(seeds)} records [{supplementary_status}]")

    # Restore abstracts that Crossref/DBLP omitted, before scoring reads title+abstract.
    enrich_path = Path("data/processed/abstract_enrichment.json")
    if enrich_path.exists():
        enrichment = json.loads(enrich_path.read_text(encoding="utf-8"))
    elif ALLOW_MISSING:
        enrichment = {}
        supplementary_status += "; abstract enrichment SKIPPED"
        logging.warning("No abstract enrichment available. 49% of records lack abstracts and "
                        "will be screened on title alone - recorded in prisma_counts.json.")
    else:
        raise FileNotFoundError(f"{enrich_path} missing - run scripts/seed_and_enrich.py first, "
                                f"or pass --allow-missing-supplementary")
    filled = 0
    for rec in records:
        doi = (rec.get("doi") or "").strip().lower()
        extra = enrichment.get(doi)
        if extra and not (rec.get("abstract") or "").strip() and extra.get("abstract"):
            rec["abstract"] = extra["abstract"]
            rec["venue"] = rec.get("venue") or extra.get("venue", "")
            if (extra.get("citations") or 0) > (rec.get("citations") or 0):
                rec["citations"] = extra["citations"]
            filled += 1
    logging.info(f"Abstract enrichment: filled {filled} previously empty abstracts")

    unique = deduplicate(records)
    logging.info(f"After deduplication: {len(unique)} unique records "
                 f"({len(records) - len(unique)} duplicates merged)")

    scored = [score(r) for r in unique]
    passed = [r for r in scored if r["passes_screen"]]
    domain_excluded = [r for r in scored if r["excluded_by"]]
    logging.info(f"After automated relevance screen: {len(passed)} records "
                 f"({len(scored) - len(passed)} excluded)")
    logging.info(f"  of which excluded on out-of-scope title terms: {len(domain_excluded)}")
    logging.info(f"  most common exclusion terms: "
                 f"{Counter(t for r in domain_excluded for t in r['excluded_by']).most_common(8)}")

    # Rank by topical relevance FIRST. Citation counts are unreliable in this field:
    # much foundational work is published outside DOI-issuing venues
    # (transformer-circuits.pub, Distill) and is badly undercounted by these databases.
    passed.sort(key=lambda r: (r["relevance_score"], r.get("citations") or 0), reverse=True)
    shortlist = passed[:SHORTLIST_SIZE]

    overlap = Counter(len(r["found_by"]) for r in scored)
    by_db = Counter(db for r in scored for db in r["found_by"])
    logging.info(f"Database contribution (unique records): {dict(by_db)}")
    logging.info(f"Records found by N databases: {dict(sorted(overlap.items()))}")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screened_all.json").write_text(
        json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "screened_passed.json").write_text(
        json.dumps(passed, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "shortlist.json").write_text(
        json.dumps(shortlist, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "prisma_counts.json").write_text(json.dumps({
        "supplementary_status": supplementary_status,
        "identification": len(records),
        "after_deduplication": len(unique),
        "duplicates_merged": len(records) - len(unique),
        "after_relevance_screen": len(passed),
        "excluded_total": len(scored) - len(passed),
        "excluded_out_of_scope_domain": len(domain_excluded),
        "shortlist_size": len(shortlist),
        "database_contribution_unique": dict(by_db),
        "found_by_n_databases": {str(k): v for k, v in sorted(overlap.items())},
    }, indent=2), encoding="utf-8")

    logging.info(f"Shortlist of {len(shortlist)} written to {out_dir / 'shortlist.json'}")
    logging.info("Top 25 by citations:")
    for r in shortlist[:25]:
        logging.info(f"  {str(r.get('citations') or 0):>6}  {r.get('year', '????'):4s}  "
                     f"{r['title'][:78]}")


if __name__ == "__main__":
    main()
