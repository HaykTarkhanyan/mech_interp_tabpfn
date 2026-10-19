#!/usr/bin/env python3
"""Condense the per-paper reading notes into one Markdown digest per review section.

Reads lit_review/notes/<id>.json and lit_review/included.json; writes lit_review/notes/_digest.md with,
per paper: one-line summary, models and action head, setting and scale, the first findings, causal
validation, closed-loop status and the reader's critique. Also lists included downloads with no note,
so a missing note is visible rather than silently absent.

Usage (from robotics_foundation_models_explainability/), ~1 s, offline:
    .venv/Scripts/python.exe scripts/notes_digest.py [--sections probing,sae]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "lit_review" / "notes"


def clip(x, n: int) -> str:
    s = "; ".join(map(str, x)) if isinstance(x, list) else str(x or "")
    return s if len(s) <= n else s[:n - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="", help="comma-separated section keys; default all")
    ap.add_argument("--out", default=str(NOTES / "_digest.md"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", stream=sys.stdout)
    inc = json.loads((ROOT / "lit_review" / "included.json").read_text(encoding="utf-8"))
    want = set(filter(None, args.sections.split(",")))
    lines, missing = [], []
    for sec, title in inc["sections"].items():
        if want and sec not in want:
            continue
        group = [w for w in inc["works"] if w["section"] == sec and w["download"]]
        if not group:
            continue
        lines += [f"## {title} ({sec})", ""]
        for w in group:
            p = NOTES / f"{w['arxiv_id']}.json"
            if not p.exists():
                missing.append(w["arxiv_id"])
                lines.append(f"### {w['arxiv_id']} NO NOTE - {w['title']}\n")
                continue
            n = json.loads(p.read_text(encoding="utf-8"))
            ev = n.get("evidence") or {}
            lines += [f"### {w['arxiv_id']} {n.get('first_author', '')} {n.get('year', '')} [{w['tier']}] - {w['title']}",
                      f"- one line: {clip(n.get('one_line'), 300)}",
                      f"- models: {clip(n.get('models'), 120)} | head: {clip(n.get('action_head'), 80)} | "
                      f"methods: {clip(n.get('methods'), 160)}",
                      f"- setting: {clip(ev.get('setting'), 60)} | benchmarks: {clip(ev.get('benchmarks'), 120)} | "
                      f"scale: {clip(ev.get('scale'), 160)} | stats: {clip(ev.get('statistics'), 120)}",
                      f"- findings: {clip(n.get('key_findings'), 700)}",
                      f"- causal validation: {clip(n.get('causal_validation'), 260)}",
                      f"- closed loop: {clip(n.get('closed_loop'), 200)}",
                      f"- critique: {clip(n.get('critique'), 450)}",
                      f"- code: {clip(n.get('code'), 80)}", ""]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"Wrote {args.out} ({len(lines)} lines); {len(missing)} downloaded works without a note: {missing}")


if __name__ == "__main__":
    main()
