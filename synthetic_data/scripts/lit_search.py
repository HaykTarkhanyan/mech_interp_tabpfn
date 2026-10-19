#!/usr/bin/env python3
"""Literature search for the review on synthetic data built from real data for evaluating XAI.

Three stages, all keyless:
  1. Keyword search: THEMES x {OpenAlex, Crossref, arXiv; DBLP when reachable}, up to --max-results each.
  2. Citation chaining: every work that cites one of the SEEDS (OpenAlex `cites:` filter).
     Any benchmark that pairs a learned real-data generator with a known target for XAI
     would almost certainly cite XAI-BENCH, OpenXAI or the ARF paper, so this is the
     check that keyword search alone cannot give.
  3. Abstract backfill: Crossref and DBLP often carry no abstract; fetch them from OpenAlex
     by DOI so the relevance screen does not silently reject records for missing data
     (see mech_interp/_learnings/2026-08-06-0035_screening-on-missing-abstracts...).

arXiv and Semantic Scholar are probed and their status recorded, never silently skipped.
OpenAlex cost: about 35 searches ($0.001 each) plus list pages ($0.0001 each), roughly $0.05
of the free daily allowance.

Usage (from synthetic_data/):
    .venv/Scripts/python.exe scripts/lit_search.py --test   # one theme, 5 results, ~10 s
    .venv/Scripts/python.exe scripts/lit_search.py          # full run, ~6-8 min
"""

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "lit_review" / "data" / "raw"

CONTACT_EMAIL = "tarkhanyan02@gmail.com"
USER_AGENT = f"synthetic-data-xai-lit-review/1.0 (mailto:{CONTACT_EMAIL})"
OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works"
DBLP = "https://dblp.org/search/publ/api"
ARXIV = "https://export.arxiv.org/api/query"
S2 = "https://api.semanticscholar.org/graph/v1/paper/search"
OA_SELECT = ("id,doi,title,publication_year,cited_by_count,authorships,"
             "abstract_inverted_index,primary_location,type")
DELAY = {"OpenAlex": 1.0, "Crossref": 1.0, "DBLP": 2.0, "arXiv": 3.0}
MAX_RETRIES = 4
TIMEOUT_S = 45
MAX_CITING_PER_SEED = 600
SEED_MATCH = 0.90

# "query": natural language for OpenAlex/Crossref rankers. "dblp": short keywords (DBLP ANDs them).
THEMES: List[Dict[str, str]] = [
    {"key": "xai_ground_truth_benchmark",
     "query": "benchmark evaluating feature attribution explanations against ground truth synthetic data",
     "dblp": "ground truth explanation benchmark"},
    {"key": "synthetic_tabular_xai",
     "query": "synthetic tabular datasets with known feature importance for evaluating explainability methods",
     "dblp": "synthetic data explainability evaluation"},
    {"key": "semi_synthetic_xai",
     "query": "semi-synthetic data real data with injected known signal to evaluate explanation methods",
     "dblp": "semi-synthetic explanation evaluation"},
    {"key": "shapley_ground_truth",
     "query": "ground truth Shapley values accuracy of SHAP estimation with dependent features",
     "dblp": "Shapley value estimation accuracy"},
    {"key": "attribution_metrics",
     "query": "faithfulness metrics for feature attribution disagree evaluation of explanation evaluation",
     "dblp": "faithfulness attribution evaluation"},
    {"key": "saliency_ground_truth",
     "query": "saliency map evaluation with ground truth relevance synthetic image benchmark",
     "dblp": "saliency ground truth benchmark"},
    {"key": "plasmode",
     "query": "plasmode simulation real covariates simulated outcome evaluation of statistical methods",
     "dblp": "plasmode simulation"},
    {"key": "semi_synthetic_causal",
     "query": "semi-synthetic benchmark for causal effect estimation realistic simulation from real data",
     "dblp": "semi-synthetic causal benchmark"},
    {"key": "generative_monte_carlo",
     "query": "generative models fitted to real data to design Monte Carlo simulation studies",
     "dblp": "Monte Carlo simulation generative adversarial networks design"},
    {"key": "empirical_monte_carlo",
     "query": "empirical Monte Carlo simulation design based on real data estimator comparison",
     "dblp": "empirical Monte Carlo"},
    {"key": "tabular_generative",
     "query": "generative model for tabular data synthetic data generation adversarial random forest",
     "dblp": "tabular data generation"},
    {"key": "synthetic_data_fidelity",
     "query": "evaluating fidelity and utility of synthetic tabular data metrics",
     "dblp": "synthetic tabular data evaluation"},
    {"key": "symbolic_regression_benchmark",
     "query": "symbolic regression benchmark real-world and ground-truth problems",
     "dblp": "symbolic regression benchmark"},
    {"key": "model_vs_data",
     "query": "feature importance explanations true to the model or true to the data generating process",
     "dblp": "feature importance data generating process"},
    {"key": "conditional_feature_importance",
     "query": "conditional feature importance conditional sampling knockoffs dependent features",
     "dblp": "conditional feature importance"},
    {"key": "feature_effects_ground_truth",
     "query": "partial dependence accumulated local effects estimation error ground truth feature effects",
     "dblp": "accumulated local effects"},
    {"key": "simulation_study_design",
     "query": "design of simulation studies neutral comparison of statistical methods realistic data",
     "dblp": "simulation study design methods comparison"},
    {"key": "graph_xai_ground_truth",
     "query": "synthetic graph generator ground truth explanations benchmark graph neural network explainers",
     "dblp": "graph explanation ground truth benchmark"},
]

# Forward-citation seeds: (key, exact title[, fallback DOI]). Resolved to OpenAlex IDs by title search;
# every OpenAlex record matching the title (preprint and published versions) is used. The phrase search
# missed two titles on 2026-10-19, so those carry their arXiv DOI as a fallback.
SEEDS: List[tuple] = [
    ("xai_bench", "Synthetic Benchmarks for Scientific Research in Explainable Machine Learning"),
    ("openxai", "OpenXAI: Towards a Transparent Evaluation of Model Explanations"),
    ("arf", "Adversarial random forests for density estimation and generative modeling", "10.48550/arxiv.2205.09435"),
    ("wgan_mc", "Using Wasserstein Generative Adversarial Networks for the design of Monte Carlo simulations"),
    ("realcause", "RealCause: Realistic Causal Inference Benchmarking"),
    # The 2026-10-19 run resolved only the arXiv record (0 citing works); the journal version is a separate
    # OpenAlex record, W2938351214 (doi 10.1002/jae.2724, 16 citing works), not chained for lack of budget.
    ("mostly_harmless_sims", "Mostly Harmless Simulations? Using Monte Carlo Studies for Estimator Selection",
     "10.48550/arxiv.1809.09527"),
    ("mamalakis_benchmark", "Neural network attribution methods for problems in geoscience: A novel synthetic benchmark dataset"),
    ("guidotti_gt", "Evaluating local explanation methods on ground truth"),
    ("bim", "Benchmarking Attribution Methods with Relative Feature Importance"),
    ("true_to_model", "True to the Model or True to the Data?", "10.48550/arxiv.2006.16234"),
    ("pdp_pfi_dgp", "Relating the Partial Dependence Plot and Permutation Feature Importance to the Data Generating Process"),
    ("srbench", "Contemporary Symbolic Regression Methods and their Relative Performance"),
    ("plasmode_franklin", "Plasmode simulation for the evaluation of pharmacoepidemiologic methods in complex healthcare databases"),
    ("feynman_interp", "Interpretability in Symbolic Regression: a benchmark of Explanatory Methods using the Feynman data set"),
]


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / "lit_search.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def get(url: str, params: Dict, source: str) -> requests.Response:
    """GET with backoff on 429/5xx. Raises after MAX_RETRIES - never returns bad data."""
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.get(url, params=params, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 4.0 * 2 ** attempt
            logging.warning(f"{source} HTTP {r.status_code} (attempt {attempt}/{MAX_RETRIES}), retry in {wait:.0f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"{source} failed after {MAX_RETRIES} attempts: {params}")


def probe(url: str, params: Dict) -> Optional[str]:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        return f"request failed: {exc}"
    return None if r.status_code == 200 else f"HTTP {r.status_code}"


def norm(title: str) -> str:
    text = unicodedata.normalize("NFKD", (title or "").lower())
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in text if not unicodedata.combining(c)))


def abstract_from_index(inv: Optional[Dict[str, List[int]]]) -> str:
    if not inv:
        return ""
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))


def record(title, authors, year, abstract, doi, url, venue, source, citations, theme, oa_id="") -> Dict:
    doi = (doi or "").replace("https://doi.org/", "").lower()
    return {"title": " ".join((title or "").split()), "authors": ", ".join(a for a in authors if a),
            "year": str(year or ""), "abstract": " ".join((abstract or "").split()), "doi": doi,
            "arxiv_id": doi.split("arxiv.", 1)[1] if doi.startswith("10.48550/arxiv.") else "",
            "url": url or "", "venue": venue or "", "source": source, "citations": citations,
            "theme": theme, "openalex_id": oa_id}


def from_openalex(item: Dict, source: str, theme: str) -> Dict:
    return record(item.get("title"),
                  [(a.get("author") or {}).get("display_name", "") for a in item.get("authorships") or []],
                  item.get("publication_year"), abstract_from_index(item.get("abstract_inverted_index")),
                  item.get("doi"), item.get("id"),
                  ((item.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
                  source, item.get("cited_by_count"), theme, (item.get("id") or "").rsplit("/", 1)[-1])


def search_openalex(query: str, n: int, theme: str) -> List[Dict]:
    r = get(OPENALEX, {"search": query, "per-page": min(n, 100), "mailto": CONTACT_EMAIL, "select": OA_SELECT},
            "OpenAlex")
    return [from_openalex(it, "OpenAlex", theme) for it in r.json().get("results") or []]


def search_crossref(query: str, n: int, theme: str) -> List[Dict]:
    r = get(CROSSREF, {"query": query, "rows": n, "mailto": CONTACT_EMAIL,
                       "select": "DOI,title,author,issued,container-title,is-referenced-by-count,abstract"},
            "Crossref")
    out = []
    for it in (r.json().get("message") or {}).get("items") or []:
        parts = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
        out.append(record((it.get("title") or [""])[0],
                          [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in it.get("author") or []],
                          parts[0] if parts else None, re.sub(r"<[^>]+>", " ", it.get("abstract", "")),
                          it.get("DOI"), f"https://doi.org/{it.get('DOI', '')}",
                          (it.get("container-title") or [""])[0], "Crossref", it.get("is-referenced-by-count"), theme))
    return out


def search_dblp(query: str, n: int, theme: str) -> List[Dict]:
    r = get(DBLP, {"q": query, "format": "json", "h": n}, "DBLP")
    out = []
    for hit in ((r.json().get("result") or {}).get("hits") or {}).get("hit") or []:
        info = hit.get("info") or {}
        raw = (info.get("authors") or {}).get("author") or []
        raw = [raw] if isinstance(raw, dict) else raw  # DBLP collapses single-author lists
        out.append(record(info.get("title"), [a.get("text", "") if isinstance(a, dict) else str(a) for a in raw],
                          info.get("year"), "", info.get("doi", ""), info.get("ee") or info.get("url", ""),
                          info.get("venue", ""), "DBLP", None, theme))
    return out


def search_arxiv(query: str, n: int, theme: str) -> List[Dict]:
    """arXiv API with the short keyword form, every word ANDed (like DBLP's matcher)."""
    q = " AND ".join(f"all:{w}" for w in query.split())
    r = get(ARXIV, {"search_query": q, "max_results": n, "sortBy": "relevance"}, "arXiv")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        arxiv_id = re.sub(r"v\d+$", "", e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1])
        out.append(record(e.findtext("a:title", "", ns), [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
                          e.findtext("a:published", "", ns)[:4], e.findtext("a:summary", "", ns),
                          f"10.48550/arxiv.{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}", "arXiv preprint",
                          "arXiv", None, theme))
    return out


def resolve_seed(title: str, doi: str = "") -> List[Dict]:
    """All OpenAlex records whose title matches the seed title (preprint + published versions)."""
    r = get(OPENALEX, {"search": f'"{title}"', "per-page": 10, "mailto": CONTACT_EMAIL,
                       "select": "id,title,publication_year,cited_by_count,doi"}, "OpenAlex")
    if doi and not any(SequenceMatcher(None, norm(title), norm(it.get("title") or "")).ratio() >= SEED_MATCH
                       for it in r.json().get("results") or []):
        logging.warning(f"seed title search missed {title[:50]!r}; trying DOI {doi}")
        r = get(OPENALEX, {"filter": f"doi:{doi}", "mailto": CONTACT_EMAIL,
                           "select": "id,title,publication_year,cited_by_count,doi"}, "OpenAlex")
    hits = []
    for it in r.json().get("results") or []:
        ratio = SequenceMatcher(None, norm(title), norm(it.get("title") or "")).ratio()
        if ratio >= SEED_MATCH:
            hits.append({"id": it["id"].rsplit("/", 1)[-1], "title": it.get("title"), "year": it.get("publication_year"),
                         "cited_by_count": it.get("cited_by_count"), "doi": it.get("doi"), "match": round(ratio, 3)})
    return hits


def citing_works(ids: List[str], theme: str) -> List[Dict]:
    """Numbered pages: keyless OpenAlex answers 401 to a second `cursor` page (checked 2026-10-19)."""
    out, page = [], 1
    while len(out) < MAX_CITING_PER_SEED:
        r = get(OPENALEX, {"filter": f"cites:{'|'.join(ids)}", "per-page": 100, "page": page,
                           "mailto": CONTACT_EMAIL, "select": OA_SELECT}, "OpenAlex")
        body = r.json()
        batch = body.get("results") or []
        out += [from_openalex(it, "OpenAlex-citing", theme) for it in batch]
        log_budget(r)
        if len(batch) < 100 or page * 100 >= (body.get("meta") or {}).get("count", 0):
            break
        page += 1
        time.sleep(DELAY["OpenAlex"])
    return out[:MAX_CITING_PER_SEED]


def log_budget(r: requests.Response) -> None:
    left = r.headers.get("x-ratelimit-remaining-usd")
    if left is not None:
        logging.info(f"    OpenAlex daily budget left: ${float(left):.4f}")


def backfill_abstracts(records: List[Dict]) -> Dict[str, Dict]:
    """OpenAlex abstracts and citation counts for DOIs whose records have no abstract."""
    need = sorted({r["doi"] for r in records if r["doi"] and not r["abstract"]})
    logging.info(f"Abstract backfill: {len(need)} DOIs without an abstract")
    found: Dict[str, Dict] = {}
    for i in range(0, len(need), 50):
        batch = need[i:i + 50]
        r = get(OPENALEX, {"filter": "doi:" + "|".join(batch), "per-page": 100, "mailto": CONTACT_EMAIL,
                           "select": "doi,abstract_inverted_index,cited_by_count,primary_location"}, "OpenAlex")
        for it in r.json().get("results") or []:
            doi = (it.get("doi") or "").replace("https://doi.org/", "").lower()
            found[doi] = {"abstract": abstract_from_index(it.get("abstract_inverted_index")),
                          "citations": it.get("cited_by_count"),
                          "venue": ((it.get("primary_location") or {}).get("source") or {}).get("display_name", "")}
        logging.info(f"  backfill {min(i + 50, len(need))}/{len(need)}")
        time.sleep(DELAY["OpenAlex"])
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true", help="one theme, 5 results per source, no chaining")
    ap.add_argument("--max-results", type=int, default=50)
    ap.add_argument("--reuse-keyword", action="store_true", help="load the saved keyword stage, run chaining only")
    args = ap.parse_args()
    setup_logging()
    start = time.time()
    RAW.mkdir(parents=True, exist_ok=True)

    probes = {"DBLP": probe(DBLP, {"q": "shapley", "format": "json", "h": 1}),
              "arXiv": probe(ARXIV, {"search_query": "all:explainability", "max_results": 1}),
              "SemanticScholar": probe(S2, {"query": "explainability", "limit": 1, "fields": "title"})}
    # DBLP answers HTTP 202 with an anti-bot challenge page (checked 2026-10-19), not JSON.
    if probes["DBLP"] is None:
        try:
            requests.get(DBLP, params={"q": "shapley", "format": "json", "h": 1}, timeout=TIMEOUT_S,
                         headers={"User-Agent": USER_AGENT}).json()
        except ValueError:
            probes["DBLP"] = "anti-bot challenge page instead of JSON"
    for name, status in probes.items():
        if status:
            logging.error(f"{name} UNAVAILABLE ({status}); recorded in search_counts.json for the methods section")
        else:
            logging.info(f"{name} probe OK")

    searchers = [("OpenAlex", search_openalex, "query"), ("Crossref", search_crossref, "query")]
    if probes["DBLP"] is None:
        searchers.append(("DBLP", search_dblp, "dblp"))
    if probes["arXiv"] is None:
        searchers.append(("arXiv", search_arxiv, "dblp"))
    if len(searchers) < 3:
        raise RuntimeError(f"Only {len(searchers)} keyword databases reachable; the review needs at least 3")
    themes = THEMES[:1] if args.test else THEMES
    n = 5 if args.test else args.max_results
    records: List[Dict] = []
    per_theme = []
    checkpoint = RAW / "keyword_stage.json"
    if args.reuse_keyword:
        if not checkpoint.exists():
            raise FileNotFoundError(f"--reuse-keyword needs {checkpoint}; run once without it first")
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        records, per_theme, themes = saved["records"], saved["per_theme"], []
        logging.info(f"Reusing keyword stage from {checkpoint}: {len(records)} records")
    for i, th in enumerate(themes, 1):
        row = {"theme": th["key"]}
        for name, fn, qkey in searchers:
            hits = fn(th[qkey], n, th["key"])
            records += hits
            row[name] = len(hits)
            time.sleep(DELAY[name])
        per_theme.append(row)
        elapsed = time.time() - start
        logging.info(f"[{i}/{len(themes)}] {th['key']}: {row}  ({elapsed:.0f}s elapsed, "
                     f"~{elapsed / i * (len(themes) - i):.0f}s left for keyword stage)")
    if not args.test and not args.reuse_keyword:
        checkpoint.write_text(json.dumps({"records": records, "per_theme": per_theme}, ensure_ascii=False),
                              encoding="utf-8")
        logging.info(f"Keyword stage saved to {checkpoint}")
    themes = THEMES[:1] if args.test else THEMES
    if args.test:
        for r in records:
            logging.info(f"  [{r['source']:8s}] {r['year']} cites={r['citations']} {r['title'][:70]}")
        return

    seed_info, chained = {}, []
    for j, (key, title, *fallback) in enumerate(SEEDS, 1):
        hits = resolve_seed(title, fallback[0] if fallback else "")
        time.sleep(DELAY["OpenAlex"])
        if not hits:
            logging.error(f"seed {key} NOT RESOLVED in OpenAlex: {title!r} (recorded; no chaining for it)")
            seed_info[key] = {"title": title, "resolved": [], "citing": 0}
            continue
        citing = citing_works([h["id"] for h in hits], f"cites:{key}")
        chained += citing
        seed_info[key] = {"title": title, "resolved": hits, "citing": len(citing)}
        logging.info(f"[seed {j}/{len(SEEDS)}] {key}: {len(hits)} record(s) "
                     f"{[h['id'] for h in hits]}, {len(citing)} citing works")
    records += chained

    fill = backfill_abstracts(records)
    filled = 0
    for r in records:
        extra = fill.get(r["doi"])
        if extra and not r["abstract"] and extra["abstract"]:
            r["abstract"] = extra["abstract"]
            r["venue"] = r["venue"] or extra["venue"]
            r["citations"] = max(r["citations"] or 0, extra["citations"] or 0)
            filled += 1
    missing = sum(1 for r in records if not r["abstract"])
    logging.info(f"Backfilled {filled} abstracts; {missing}/{len(records)} records still have none")

    (RAW / "combined_raw.json").write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    (RAW / "search_counts.json").write_text(json.dumps({
        "search_date": time.strftime("%Y-%m-%d"),
        "databases_used": [s[0] for s in searchers] + ["OpenAlex citation chaining"],
        "databases_unavailable": {k: v for k, v in probes.items() if v},
        "results_per_query_per_source": n,
        "queries": {t["key"]: {"default": t["query"], "dblp": t["dblp"]} for t in themes},
        "per_theme": per_theme,
        "keyword_records": len(records) - len(chained),
        "seeds": seed_info,
        "citing_records": len(chained),
        "abstracts_backfilled": filled,
        "records_without_abstract": missing,
        "total_records": len(records),
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    logging.info(f"Wrote {len(records)} records to {RAW / 'combined_raw.json'} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
