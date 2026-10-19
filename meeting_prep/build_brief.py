"""Build the supervisor meeting brief (standalone HTML) from saved JSON data.

Inputs (all in meeting_prep/): citation_snapshot_<day>.json, venue_deadlines_<day>.json,
brief_template.html, plus papers/tabpfn/manifest.json for author names.
Output: meeting_prep/supervisor_brief.html. Nothing is fetched; rerun any time.

Run: python meeting_prep/build_brief.py --day 2026-10-19   (~1 s)
"""
import argparse
import ast
import json
import logging
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "meeting_prep"
(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(ROOT / "logs" / "build_brief.log", encoding="utf-8")])
log = logging.getLogger("build_brief")

# Papers that introduce a model; excluded from the "realistic citations" comparison.
MODEL_PAPERS = {"10.1038/s41586-024-08328-6", "2112.10510", "2207.01848", "2511.08667",
                "2605.13986", "2502.05564", "2410.18164"}

# Venue corrections taken from the papers themselves where Semantic Scholar lists only arXiv.
OVERRIDES = {
    "2605.06510": ("ICML", "Main ML conference", 2026),
    "2606.12917": ("ICML workshop (FMSD)", "Workshop / blog track", 2026),
    "2511.03634": ("EurIPS workshop", "Workshop / blog track", 2025),
    "2502.08978": ("ICLR blogpost track", "Workshop / blog track", 2024),
    # neurips.cc/virtual/2026/160449: poster in the XAI4Science workshop (checked 2026-10-19).
    "2601.08181": ("NeurIPS workshop (XAI4Science)", "Workshop / blog track", 2026),
    "2306.00541": ("JMLR", "Journal", 2024),
    "2109.01433": ("xAI", "Specialist conference", 2023),
    "2603.29946": ("ICLR 2026 workshop", "Workshop / blog track", 2026),
    "1902.05622": ("ICML", "Main ML conference", 2020),
    "10.1111/rssb.12377": ("JRSS-B", "Journal", 2020),
}
S2_VENUES = [
    ("Neural Information Processing Systems", "NeurIPS", "Main ML conference"),
    ("International Conference on Machine Learning", "ICML", "Main ML conference"),
    ("International Conference on Learning Representations", "ICLR", "Main ML conference"),
    ("International Conference on Artificial Intelligence and Statistics", "AISTATS", "Main ML conference"),
    ("Trans. Mach. Learn. Res.", "TMLR", "Journal"),
    ("Nature", "Nature", "Journal"),
    ("Journal of The Royal Statistical", "JRSS-B", "Journal"),
    ("xAI", "xAI", "Specialist conference"),
    ("arXiv.org", "arXiv only", "arXiv only"),
]


def venue_of(pid, s2):
    if pid in OVERRIDES:
        return OVERRIDES[pid]
    v = (s2 or {}).get("venue") or ""
    if not v:
        return ("arXiv only", "arXiv only", (s2 or {}).get("year"))
    for prefix, short, kind in S2_VENUES:
        if v.startswith(prefix):
            return (short, kind, s2.get("year"))
    raise ValueError(f"Unmapped Semantic Scholar venue '{v}' for {pid}; add it to S2_VENUES")


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="snapshot day, e.g. 2026-10-19")
    args = ap.parse_args()
    cit = json.loads((HERE / f"citation_snapshot_{args.day}.json").read_text(encoding="utf-8"))
    dl = json.loads((HERE / f"venue_deadlines_{args.day}.json").read_text(encoding="utf-8"))
    tpl = (HERE / "brief_template.html").read_text(encoding="utf-8")
    man = json.loads((ROOT / "papers" / "tabpfn" / "manifest.json").read_text(encoding="utf-8"))
    man = man["papers"] if isinstance(man, dict) else man
    by_title = {p["title"]: p for p in man}
    fetched = date.fromisoformat(cit["fetched_at"][:10])

    papers = []
    for r in cit["papers"]:
        pid = r["arxiv_id"] or r["doi"]
        s2 = r.get("s2")
        collection = r["group"].startswith("collection")
        if collection:
            m = by_title[r["label"]]
            authors = ast.literal_eval(m["authors"]) if isinstance(m["authors"], str) else m["authors"]
            surname = authors[0].split(",")[0].strip() if authors else "?"
            first_date = (s2 or {}).get("date") or m["date"].replace("/", "-")
            short = f"{surname}{' et al.' if len(authors) > 1 else ''} {first_date[:4]}: {r['label']}"
            group_short = "TabPFN " + r["group"].split(": ")[1]
        else:
            first_date = (s2 or {}).get("date") or f"{(s2 or {}).get('year', fetched.year)}-07-01"
            short, group_short = r["label"], r["group"]
        venue_short, kind, vyear = venue_of(pid, s2)
        age = max((fetched - date.fromisoformat(first_date[:10])).days / 365.25, 0.05)
        oa = r.get("openalex") or {}
        papers.append({"id": pid, "short": short, "group_short": group_short, "collection": collection,
                       "model_paper": pid in MODEL_PAPERS, "venue_short": venue_short, "kind": kind,
                       "year": vyear, "s2": None if s2 is None else s2["citations"],
                       "oa": oa.get("citations"), "age": round(age, 2)})
    log.info(f"{len(papers)} papers prepared; {sum(p['collection'] and not p['model_paper'] for p in papers)} comparable collection papers")

    def js(obj):
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    out = (tpl.replace("__CITATIONS_JSON__", js({"fetched_at": cit["fetched_at"], "papers": papers}))
              .replace("__DEADLINES_JSON__", js(dl))
              .replace("__BUILD_DATE__", date.today().isoformat())
              .replace("__FETCH_DATE__", cit["fetched_at"][:10])
              .replace("__FETCH_DAY__", args.day)
              .replace("__DEADLINE_DATE__", dl["checked_on"]))
    leftover = [t for t in ("__CITATIONS_JSON__", "__DEADLINES_JSON__", "__BUILD_DATE__", "__FETCH_DATE__",
                            "__FETCH_DAY__", "__DEADLINE_DATE__") if t in out]
    if leftover:
        raise RuntimeError(f"Unfilled template placeholders: {leftover}")
    dest = HERE / "supervisor_brief.html"
    dest.write_text(out, encoding="utf-8")
    log.info(f"Wrote {dest} ({dest.stat().st_size / 1024:.0f} KB) in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
