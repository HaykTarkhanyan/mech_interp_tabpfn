#!/usr/bin/env python3
"""Literature search for the review on explaining and interpreting robot foundation models.

Three stages, all keyless:
  1. Keyword search: THEMES x {arXiv (field-scoped boolean query), Semantic Scholar, Crossref}.
  2. Citation chaining in both directions (who cites a seed, what a seed cites) via Semantic Scholar.
     Seeds are arXiv ids, title-checked against the arXiv API before use.
  3. Abstract backfill for records that carry an arXiv id but no abstract, from the arXiv API.

OpenAlex is not used: on 2026-10-19 the shared keyless budget for this IP had $0.017 of $0.10 left
after the synthetic-data review, and OpenAlex lists 0 references for the main seed paper
(2509.00328), so its citation graph is thin for this arXiv-native literature.

Semantic Scholar's keyless pool rate-limits hard (200 then 429 on consecutive requests, checked
2026-10-19). Chaining results are cached per seed in data/raw/chain_cache.json, so a rerun resumes.
If any seed still fails after all retries, outputs are written and the script exits with code 1.

Usage (from robotics_foundation_models_explainability/):
    .venv/Scripts/python.exe scripts/lit_search.py --test --no-chain  # one theme, ~20 s
    .venv/Scripts/python.exe scripts/lit_search.py --no-chain         # keyword stage only, ~2-3 min
    .venv/Scripts/python.exe scripts/lit_search.py --reuse-keyword    # S2 chaining + backfill; 26 S2 calls, can take
                                                                      # an hour or more under 429 backoff (guess)
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
USER_AGENT = f"robotics-fm-explainability-lit-review/1.0 (mailto:{CONTACT_EMAIL})"
ARXIV = "https://export.arxiv.org/api/query"
S2 = "https://api.semanticscholar.org/graph/v1"
CROSSREF = "https://api.crossref.org/works"
S2_FIELDS = "title,authors,year,abstract,externalIds,venue,citationCount"
DELAY = {"arXiv": 3.0, "S2": 4.0, "Crossref": 1.0}
RETRIES = {"arXiv": 4, "S2": 7, "Crossref": 4}
TIMEOUT_S = 60
SEED_MATCH = 0.85
NS = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}

# Interpretability vocabulary reused across the arXiv queries.
INTERP = ('(abs:interpretability OR abs:interpretable OR abs:mechanistic OR abs:explainability OR '
          'abs:explainable OR abs:"sparse autoencoder" OR abs:"sparse autoencoders" OR abs:probing OR '
          'abs:"linear probe" OR abs:"linear probes" OR abs:steering)')

# "arxiv": boolean query over arXiv fields. "short": plain keywords for Semantic Scholar and Crossref.
THEMES: List[Dict[str, str]] = [
    {"key": "vla_interp", "tier": "core",
     "arxiv": f'abs:"vision-language-action" AND {INTERP}',
     "short": "vision-language-action model mechanistic interpretability"},
    {"key": "vla_attribution", "tier": "core",
     "arxiv": 'abs:"vision-language-action" AND (abs:attribution OR abs:saliency OR abs:"attention map" OR '
              'abs:"attention maps" OR abs:"attention heads" OR abs:"attention patterns" OR abs:neurons)',
     "short": "vision-language-action attention attribution saliency analysis"},
    {"key": "vla_short_name", "tier": "core",
     "arxiv": '(ti:VLA OR ti:VLAs) AND (abs:interpretability OR abs:mechanistic OR abs:"sparse autoencoder" OR '
              'abs:"activation steering" OR abs:probing OR abs:"linear probe" OR abs:neurons OR '
              'abs:"internal representations")',
     "short": "VLA interpretability sparse autoencoder steering probing"},
    {"key": "vla_representations", "tier": "core",
     "arxiv": 'abs:"vision-language-action" AND (abs:"hidden states" OR abs:"intermediate layers" OR '
              'abs:"layer-wise" OR abs:"representation analysis" OR abs:"internal representations" OR '
              'abs:"latent representations")',
     "short": "vision-language-action internal representations analysis layers"},
    {"key": "robot_policy_interp", "tier": "core",
     "arxiv": '(abs:"robot policy" OR abs:"robot policies" OR abs:"visuomotor policy" OR abs:"visuomotor policies" OR '
              'abs:"generalist policy" OR abs:"generalist policies" OR abs:"robot foundation model" OR '
              f'abs:"robot foundation models" OR abs:"robot learning") AND {INTERP}',
     "short": "interpretability of robot policy internal representations"},
    {"key": "diffusion_policy_interp", "tier": "core",
     "arxiv": '(abs:"diffusion policy" OR abs:"diffusion policies" OR abs:"flow matching policy" OR '
              f'abs:"flow-matching policy" OR abs:"action expert") AND ({INTERP} OR abs:attribution)',
     "short": "diffusion policy robot interpretability explanation"},
    {"key": "vla_failure_monitoring", "tier": "core",
     "arxiv": 'abs:"vision-language-action" AND (abs:"failure detection" OR abs:"failure prediction" OR '
              'abs:"runtime monitoring" OR abs:"failure explanation" OR abs:"uncertainty estimation")',
     "short": "vision-language-action failure detection internal features"},
    {"key": "vla_language_grounding", "tier": "core",
     "arxiv": 'abs:"vision-language-action" AND (abs:"language grounding" OR abs:"instruction following" OR '
              'abs:"ignore language" OR abs:"ignores language" OR abs:"language conditioning" OR '
              'abs:"counterfactual")',
     "short": "do vision-language-action models follow language instructions analysis"},
    {"key": "vla_safety_internals", "tier": "core",
     "arxiv": 'abs:"vision-language-action" AND (abs:backdoor OR abs:adversarial OR abs:jailbreak OR '
              'abs:unlearning OR abs:hallucination) AND (abs:mechanistic OR abs:interpretability OR '
              'abs:neurons OR abs:"sparse autoencoder" OR abs:features OR abs:"attention")',
     "short": "vision-language-action adversarial backdoor mechanistic defense"},
    {"key": "concept_policies", "tier": "core",
     "arxiv": '(abs:"concept bottleneck" OR abs:"concept-based" OR abs:"concept learning") AND '
              '(abs:robot OR abs:robotic OR abs:manipulation) AND (abs:policy OR abs:policies OR '
              'abs:"imitation learning" OR abs:"vision-language-action")',
     "short": "concept bottleneck robot manipulation policy interpretable"},
    {"key": "robot_xai", "tier": "adjacent",
     "arxiv": '(abs:robot OR abs:robotic OR abs:robotics) AND (abs:"foundation model" OR abs:"foundation models" OR '
              'abs:"vision-language model" OR abs:"large language model") AND (abs:explainability OR '
              'abs:"explainable AI" OR abs:"explainable robotics" OR abs:"explain its" OR abs:"self-explanation")',
     "short": "explainable robotics foundation models explanation of robot behavior"},
    {"key": "world_model_interp", "tier": "adjacent",
     "arxiv": '(abs:"world model" OR abs:"world models") AND (abs:robot OR abs:robotic OR abs:embodied) AND '
              '(abs:mechanistic OR abs:interpretability OR abs:probing OR abs:"sparse autoencoder" OR abs:"linear probe")',
     "short": "interpretability of world models for robotics probing"},
    {"key": "embodied_llm_interp", "tier": "adjacent",
     "arxiv": '(abs:embodied OR abs:robot OR abs:robotic) AND (abs:"mechanistic interpretability" OR '
              'abs:"sparse autoencoder" OR abs:"activation steering" OR abs:"activation patching")',
     "short": "embodied agent mechanistic interpretability"},
    {"key": "rl_agent_mech_interp", "tier": "adjacent",
     "arxiv": 'abs:"mechanistic interpretability" AND (abs:"reinforcement learning" OR abs:"RL agent" OR '
              'abs:"RL agents" OR abs:"deep RL" OR abs:"policy network")',
     "short": "mechanistic interpretability reinforcement learning agent"},
    {"key": "driving_interp", "tier": "adjacent",
     "arxiv": '(abs:"autonomous driving" OR abs:"end-to-end driving") AND (abs:"vision-language-action" OR '
              'abs:"foundation model") AND (abs:mechanistic OR abs:"sparse autoencoder" OR abs:probing OR '
              'abs:"linear probe" OR abs:"activation steering")',
     "short": "autonomous driving foundation model mechanistic interpretability"},
    {"key": "robot_visual_repr_probing", "tier": "adjacent",
     "arxiv": '(abs:"visual representations" OR abs:"visual representation" OR abs:"vision encoder") AND '
              '(abs:robot OR abs:manipulation) AND (abs:probing OR abs:"linear probe" OR abs:"linear probing" OR '
              'abs:interpretability)',
     "short": "pretrained visual representations for robot manipulation probing"},
]

# (key, arXiv id, exact arXiv title). Core interpretability-of-VLA papers found by the 2026-10-19 probe query.
SEEDS: List[tuple] = [
    ("haon_steering", "2509.00328", "Mechanistic interpretability for steering vision-language-action models"),
    ("sae_vla", "2603.19183", "Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models"),
    ("observe_control", "2603.05487", "Observing and Controlling Features in Vision-Language-Action Models"),
    ("not_all_features", "2603.19233",
     "Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models"),
    ("probing_symbolic", "2502.04558",
     "Probing a Vision-Language-Action Model for Symbolic States and Integration into a Cognitive Architecture"),
    ("event_sae", "2605.17204", "Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies"),
    ("language_grounding", "2610.10178",
     "Do Vision-Language-Action Models Understand Instructions? A Mechanistic Interpretability Study on Language Grounding"),
    ("concept_dict_safety", "2602.01834",
     "Concept-Based Dictionary Learning for Inference-Time Safety in Vision Language Action Models"),
    ("embodied_interp", "2605.00321",
     "Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models"),
    ("mech_finetune", "2511.22697", "Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations"),
    ("task_progress", "2608.13474", "Decoding Task Progress from VLA Representations"),
    ("illusion_reasoning", "2604.18000", "Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models"),
    ("latent_cluster", "2609.02634", "Latent Cluster Analysis for Vision-Language-Action Models"),
]


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / "lit_search.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def get(url: str, params: Dict, source: str) -> requests.Response:
    """GET with exponential backoff on 429/5xx. Raises after the source's retry budget."""
    for attempt in range(1, RETRIES[source] + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            logging.warning(f"{source} request error (attempt {attempt}): {exc}")
            time.sleep(5.0 * attempt)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(5.0 * 2 ** attempt, 150.0)
            logging.warning(f"{source} HTTP {r.status_code} (attempt {attempt}/{RETRIES[source]}), retry in {wait:.0f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"{source} failed after {RETRIES[source]} attempts: {url} {params}")


def norm(title: str) -> str:
    text = unicodedata.normalize("NFKD", (title or "").lower())
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in text if not unicodedata.combining(c)))


def record(title, authors, year, abstract, arxiv_id, doi, url, venue, source, citations, theme) -> Dict:
    doi = (doi or "").replace("https://doi.org/", "").lower()
    if not arxiv_id and doi.startswith("10.48550/arxiv."):
        arxiv_id = doi.split("arxiv.", 1)[1]
    return {"title": " ".join((title or "").split()), "authors": ", ".join(a for a in authors if a),
            "year": str(year or ""), "abstract": " ".join((abstract or "").split()),
            "arxiv_id": re.sub(r"v\d+$", "", arxiv_id or ""), "doi": doi, "url": url or "", "venue": venue or "",
            "source": source, "citations": citations, "theme": theme}


def parse_arxiv_feed(text: str, source: str, theme: str) -> tuple:
    root = ET.fromstring(text)
    total = int(root.findtext("os:totalResults", "0", NS))
    out = []
    for e in root.findall("a:entry", NS):
        arxiv_id = e.findtext("a:id", "", NS).rsplit("/abs/", 1)[-1]
        out.append(record(e.findtext("a:title", "", NS), [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)],
                          e.findtext("a:published", "", NS)[:4], e.findtext("a:summary", "", NS), arxiv_id, "",
                          f"https://arxiv.org/abs/{re.sub(r'v[0-9]+$', '', arxiv_id)}", "arXiv preprint", source, None,
                          theme))
    return out, total


def search_arxiv(query: str, n: int, theme: str) -> tuple:
    r = get(ARXIV, {"search_query": query, "max_results": n, "sortBy": "relevance"}, "arXiv")
    return parse_arxiv_feed(r.text, "arXiv", theme)


def from_s2(p: Dict, source: str, theme: str) -> Dict:
    ext = p.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv") or ""
    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (
        f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}")
    return record(p.get("title"), [a.get("name", "") for a in p.get("authors") or []], p.get("year"),
                  p.get("abstract"), arxiv_id, ext.get("DOI", ""), url, p.get("venue"), source,
                  p.get("citationCount"), theme)


def search_s2(query: str, n: int, theme: str) -> tuple:
    r = get(f"{S2}/paper/search", {"query": query, "limit": n, "fields": S2_FIELDS}, "S2")
    body = r.json()
    return [from_s2(p, "SemanticScholar", theme) for p in body.get("data") or []], body.get("total", 0)


def search_crossref(query: str, n: int, theme: str) -> tuple:
    r = get(CROSSREF, {"query": query, "rows": n, "mailto": CONTACT_EMAIL,
                       "select": "DOI,title,author,issued,container-title,is-referenced-by-count,abstract"}, "Crossref")
    msg = r.json().get("message") or {}
    out = []
    for it in msg.get("items") or []:
        parts = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
        out.append(record((it.get("title") or [""])[0],
                          [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in it.get("author") or []],
                          parts[0] if parts else None, re.sub(r"<[^>]+>", " ", it.get("abstract", "")), "",
                          it.get("DOI"), f"https://doi.org/{it.get('DOI', '')}",
                          (it.get("container-title") or [""])[0], "Crossref", it.get("is-referenced-by-count"), theme))
    return out, msg.get("total-results", 0)


def arxiv_by_ids(ids: List[str]) -> Dict[str, Dict]:
    """arXiv metadata for up to 50 ids per request, keyed by version-less id."""
    found: Dict[str, Dict] = {}
    for i in range(0, len(ids), 50):
        r = get(ARXIV, {"id_list": ",".join(ids[i:i + 50]), "max_results": 50}, "arXiv")
        recs, _ = parse_arxiv_feed(r.text, "arXiv", "")
        found.update({rec["arxiv_id"]: rec for rec in recs})
        time.sleep(DELAY["arXiv"])
    return found


def verify_seeds() -> None:
    """Fail loudly if a seed id resolves to a different title (see _learnings 2026-08-06-0040)."""
    meta = arxiv_by_ids([s[1] for s in SEEDS])
    bad = []
    for key, arxiv_id, title in SEEDS:
        got = (meta.get(arxiv_id) or {}).get("title", "")
        ratio = SequenceMatcher(None, norm(title), norm(got)).ratio()
        if ratio < SEED_MATCH:
            bad.append(f"{key} {arxiv_id}: expected {title!r}, arXiv has {got!r} (ratio {ratio:.2f})")
    if bad:
        raise RuntimeError("Seed id/title mismatch:\n" + "\n".join(bad))
    logging.info(f"All {len(SEEDS)} seeds title-checked against arXiv")


def probe_s2() -> Optional[str]:
    """Three spaced tries; the keyless pool answered 200 then 429 on back-to-back requests (2026-10-19)."""
    status = "no attempt"
    for _ in range(3):
        try:
            r = requests.get(f"{S2}/paper/arXiv:{SEEDS[0][1]}", params={"fields": "title"}, timeout=TIMEOUT_S,
                             headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return None
            status = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            status = f"request failed: {exc}"
        time.sleep(20)
    return status


def chain_seed(arxiv_id: str, direction: str, key: str) -> List[Dict]:
    """direction 'citations' (works citing the seed) or 'references' (works the seed cites)."""
    r = get(f"{S2}/paper/arXiv:{arxiv_id}/{direction}", {"fields": S2_FIELDS, "limit": 1000}, "S2")
    inner = "citingPaper" if direction == "citations" else "citedPaper"
    return [from_s2(row[inner], f"S2-{direction}", f"{direction}:{key}")
            for row in r.json().get("data") or [] if (row.get(inner) or {}).get("title")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true", help="one theme, 5 results per source, no chaining")
    ap.add_argument("--arxiv-max", type=int, default=100)
    # S2 keyword search is off by default: the keyless pool 429s on the request after a success
    # (2026-10-19), and arXiv already covers this arXiv-native literature.
    ap.add_argument("--s2-max", type=int, default=0)
    ap.add_argument("--no-chain", action="store_true", help="skip Semantic Scholar citation chaining")
    ap.add_argument("--crossref-max", type=int, default=20)
    ap.add_argument("--reuse-keyword", action="store_true", help="load the saved keyword stage, run chaining only")
    args = ap.parse_args()
    setup_logging()
    start = time.time()
    RAW.mkdir(parents=True, exist_ok=True)

    verify_seeds()
    s2_status = "skipped: --no-chain and --s2-max 0" if args.no_chain and args.s2_max == 0 else probe_s2()
    if s2_status and s2_status.startswith("skipped"):
        logging.info(f"Semantic Scholar not used this run ({s2_status}); recorded in search_counts.json")
    elif s2_status:
        logging.error(f"Semantic Scholar UNAVAILABLE ({s2_status}): no S2 keyword search and no citation chaining "
                      f"this run; recorded in search_counts.json. Rerun with --reuse-keyword later to chain.")
    else:
        logging.info("Semantic Scholar probe OK")
    themes = THEMES[:1] if args.test else THEMES
    n = {"arXiv": 5, "S2": 5, "Crossref": 5} if args.test else \
        {"arXiv": args.arxiv_max, "S2": args.s2_max, "Crossref": args.crossref_max}
    searchers = [("arXiv", search_arxiv, "arxiv"), ("Crossref", search_crossref, "short")]
    if not s2_status and n["S2"] > 0:
        searchers.insert(1, ("S2", search_s2, "short"))

    records: List[Dict] = []
    per_theme: List[Dict] = []
    failures: List[str] = []
    checkpoint = RAW / "keyword_stage.json"
    if args.reuse_keyword:
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        records, per_theme = saved["records"], saved["per_theme"]
        logging.info(f"Reusing keyword stage from {checkpoint}: {len(records)} records")
    else:
        for i, th in enumerate(themes, 1):
            row = {"theme": th["key"], "tier": th["tier"]}
            for name, fn, qkey in searchers:
                try:
                    hits, total = fn(th[qkey], n[name], th["key"])
                except RuntimeError as exc:
                    logging.error(f"{name} FAILED for theme {th['key']}: {exc}")
                    failures.append(f"keyword {name} {th['key']}")
                    row[name] = "failed"
                    continue
                records += hits
                row[name] = {"retrieved": len(hits), "total_matches": total}
                time.sleep(DELAY[name])
            per_theme.append(row)
            elapsed = time.time() - start
            logging.info(f"[{i}/{len(themes)}] {th['key']}: {row}  ({elapsed:.0f}s elapsed, "
                         f"~{elapsed / i * (len(themes) - i):.0f}s left for keyword stage)")
        if args.test:
            for r in records:
                logging.info(f"  [{r['source']:15s}] {r['year']} {r['arxiv_id']:11s} {r['title'][:80]}")
            return
        checkpoint.write_text(json.dumps({"records": records, "per_theme": per_theme}, ensure_ascii=False),
                              encoding="utf-8")
        logging.info(f"Keyword stage saved to {checkpoint}: {len(records)} records")

    cache_path = RAW / "chain_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    jobs = [] if s2_status or args.no_chain else [(key, aid, d) for key, aid, _ in SEEDS for d in ("citations", "references")]
    for j, (key, aid, direction) in enumerate(jobs, 1):
        ck = f"{key}:{direction}"
        if ck not in cache:
            try:
                cache[ck] = chain_seed(aid, direction, key)
            except RuntimeError as exc:
                logging.error(f"S2 chaining FAILED for {ck}: {exc}")
                failures.append(f"chain {ck}")
                continue
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            time.sleep(DELAY["S2"])
        elapsed = time.time() - start
        logging.info(f"[chain {j}/{len(jobs)}] {ck}: {len(cache[ck])} works ({elapsed:.0f}s elapsed)")
    chained = [rec for ck, recs in cache.items() for rec in recs]
    records += chained

    need = sorted({r["arxiv_id"] for r in records if r["arxiv_id"] and not r["abstract"]})
    logging.info(f"Abstract backfill: {len(need)} arXiv ids without an abstract")
    filled_meta = arxiv_by_ids(need) if need else {}
    filled = 0
    for r in records:
        extra = filled_meta.get(r["arxiv_id"])
        if extra and not r["abstract"] and extra["abstract"]:
            r["abstract"] = extra["abstract"]
            filled += 1
    missing = sum(1 for r in records if not r["abstract"])
    logging.info(f"Backfilled {filled} abstracts; {missing}/{len(records)} records still have none")

    (RAW / "combined_raw.json").write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    (RAW / "search_counts.json").write_text(json.dumps({
        "search_date": time.strftime("%Y-%m-%d"),
        "databases_used": [s[0] for s in searchers] + ([] if s2_status else ["Semantic Scholar citation chaining"]),
        "semantic_scholar_status": s2_status or "ok",
        "databases_not_used": {"OpenAlex": "keyless daily budget for this IP at $0.017 of $0.10 on 2026-10-19; "
                                           "0 references listed for seed 2509.00328",
                               "DBLP": "anti-bot challenge page instead of JSON (checked 2026-10-19 for the "
                                       "synthetic-data review)"},
        "results_per_query": n,
        "queries": {t["key"]: {"tier": t["tier"], "arxiv": t["arxiv"], "short": t["short"]} for t in THEMES},
        "per_theme": per_theme,
        "keyword_records": len(records) - len(chained),
        "seeds": {key: {"arxiv_id": aid, "title": title,
                        "citations": len(cache.get(f"{key}:citations", [])),
                        "references": len(cache.get(f"{key}:references", []))} for key, aid, title in SEEDS},
        "chained_records": len(chained),
        "abstracts_backfilled": filled,
        "records_without_abstract": missing,
        "total_records": len(records),
        "failures": failures,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    logging.info(f"Wrote {len(records)} records to {RAW / 'combined_raw.json'} in {time.time() - start:.0f}s")
    if failures:
        logging.error(f"{len(failures)} request groups failed: {failures}. Outputs are partial; rerun to resume "
                      f"(chaining is cached; add --reuse-keyword if the keyword stage was complete).")
        sys.exit(1)


if __name__ == "__main__":
    main()
