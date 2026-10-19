#!/usr/bin/env python3
"""Field-level counts over the core reading notes, for the review's synthesis section.

Classifies each core note by setting (simulation / real robot / both / offline), whether closed-loop
evidence exists, whether interpretations were validated by intervention, whether statistics were
reported, and which model families it studies. Classification is by keywords in the note fields,
so it is approximate; the per-paper labels are written out for checking.

Outputs lit_review/data/processed/notes_stats.json.

Usage (from robotics_foundation_models_explainability/), ~1 s, offline:
    .venv/Scripts/python.exe scripts/notes_stats.py
"""

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "lit_review" / "notes"
MODEL_PATTERNS = {"pi0.5": r"pi0\.5|pi-0\.5|π0\.5", "pi0 / pi0-FAST": r"\bpi0(?!\.5)|pi0-fast",
                  "OpenVLA / OFT": r"openvla", "GR00T": r"gr00t|groot", "SmolVLA": r"smolvla",
                  "Octo": r"\bocto\b", "Molmo family": r"molmo", "world-action models": r"cosmos|lingbot|fastwam|dit4dit",
                  "Diffusion Policy": r"diffusion policy"}


def first_word(s: str) -> str:
    return (re.match(r"\s*([A-Za-z]+)", str(s or "")) or [None, ""])[1].lower()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", stream=sys.stdout)
    works = [w for w in json.loads((ROOT / "lit_review" / "included.json").read_text(encoding="utf-8"))["works"]
             if w["tier"] == "core" and w["download"]]  # read in full; abstract-only core works have no note
    rows, setting, closed, causal, stats, models = [], Counter(), Counter(), Counter(), Counter(), Counter()
    for w in works:
        p = NOTES / f"{w['arxiv_id']}.json"
        if not p.exists():
            raise FileNotFoundError(f"core work without a reading note: {w['arxiv_id']}")
        n = json.loads(p.read_text(encoding="utf-8"))
        ev = n.get("evidence") or {}
        s = str(ev.get("setting", "")).lower()
        s_cls = ("both" if s.startswith("both") else "real robot" if s.startswith("real") else
                 "offline only" if s.startswith("offline") else "simulation" if s.startswith("sim") else "other")
        c_cls = "yes" if first_word(n.get("closed_loop")) in ("yes", "mixed", "mostly", "partly", "partially") else "no"
        v_cls = ("yes" if first_word(n.get("causal_validation")) in ("yes", "partial", "partially", "indirect",
                                                                       "limited") else "no")
        st_cls = first_word(ev.get("statistics")) or "unclear"
        found = [m for m, pat in MODEL_PATTERNS.items() if re.search(pat, " ".join(n.get("models") or []), re.I)]
        setting[s_cls] += 1
        closed[c_cls] += 1
        causal[v_cls] += 1
        stats[st_cls if st_cls in ("yes", "no", "partial") else "unclear"] += 1
        models.update(found)
        rows.append({"id": w["arxiv_id"], "setting": s_cls, "closed_loop": c_cls, "causal": v_cls, "stats": st_cls,
                     "models": found})
    out = {"n_core_notes": len(rows), "setting": dict(setting), "closed_loop": dict(closed),
           "causal_validation": dict(causal), "statistics": dict(stats), "models": dict(models.most_common()),
           "papers": rows}
    (ROOT / "lit_review" / "data" / "processed" / "notes_stats.json").write_text(json.dumps(out, indent=1),
                                                                                 encoding="utf-8")
    logging.info(json.dumps({k: v for k, v in out.items() if k != "papers"}, indent=1))


if __name__ == "__main__":
    main()
