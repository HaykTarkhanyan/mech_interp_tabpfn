#!/usr/bin/env python3
"""Cost estimate for transfer_pilot_v2.py runs, from the measured timing runs (--timing results).

Wall time of one run = k_reality x t(reality task) + reps x (world build + k x sum over worlds of t(world task)),
with t measured on this laptop at 2 threads. Parallel runs split replicates across processes (2 threads each).
Writes results/transfer_pilot/cost_estimate.md.

Usage (from synthetic_data/, < 1 s):
    .venv/Scripts/python.exe scripts/transfer_cost.py
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "transfer_pilot"
BUILD_S_FALLBACK = 20.0   # timing_rf.json predates the build_seconds field; the log shows 18.4 s and 19.6 s
TIERS = [
    ("Tier 0: go/no-go", ["rf"], 4, 4, 30, 1),
    ("Tier 1: full pilot", ["rf", "mlp"], 8, 5, 30, 1),
    ("Tier 2: paper scale (x5 datasets, 3 realities)", ["rf", "mlp", "mlp"], 10, 5, 30, 5),
]


def per_world_seconds(res: dict) -> dict:
    out = {}
    for t in res["tasks"]:
        key = t["role"] if t["role"] != "builder" else t["world"]
        out.setdefault(key, []).append(t["seconds"]["total"])
    return {k: sum(v) / len(v) for k, v in out.items()}


def run_seconds(t: dict, build: float, reps: int, k: int, k_reality: int) -> float:
    per_rep_tasks = sum(v for name, v in t.items() if name != "reality")
    return k_reality * t["reality"] + reps * (build + k * per_rep_tasks)


def main() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "transfer_cost.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    timing, derived = {}, []
    rf = json.loads((OUT_DIR / "timing_rf.json").read_text(encoding="utf-8"))
    for fam in ("rf", "mlp"):
        path = OUT_DIR / f"timing_{fam}.json"
        if path.exists():
            res = json.loads(path.read_text(encoding="utf-8"))
            builds = res.get("build_seconds") or [BUILD_S_FALLBACK]
            timing[fam] = (per_world_seconds(res), sum(builds) / len(builds))
        else:
            # Builder and null worlds do not depend on the reality family; only reality and oracle tasks do, and their
            # data-level truth for an MLP is the exact enumeration measured in the rf run's mlp-target builder worlds.
            t_rf = per_world_seconds(rf)
            mlp_truth = [t["seconds"]["truth_data"] for t in rf["tasks"] if t["world"].endswith("|mlp")]
            rf_truth = [t["seconds"]["truth_data"] for t in rf["tasks"] if t["role"] in ("reality", "oracle")]
            shift = sum(mlp_truth) / len(mlp_truth) - sum(rf_truth) / len(rf_truth)
            t = dict(t_rf, reality=t_rf["reality"] + shift, oracle=t_rf["oracle"] + shift)
            timing[fam] = (t, BUILD_S_FALLBACK)
            derived.append(f"{fam}: no timing run; reality and oracle tasks derived from timing_rf.json "
                           f"(+{shift:.1f}s for exact MLP truths)")
        t = timing[fam][0]
        logging.info(f"{fam}: reality task {t['reality']:.1f}s, mean builder task "
                     f"{sum(v for k, v in t.items() if '|' in k) / 12:.1f}s, build {timing[fam][1]:.1f}s")
    lines = ["# Transfer pilot v2: cost estimate", "",
             "From the measured timing runs (one task per world, 2 threads, this laptop). Parallel = replicates split "
             "across processes of 2 threads each; ideal speed-up, so treat it as a lower bound.", ""]
    lines += [f"- Derived: {d}" for d in derived] + ([""] if derived else [])
    lines += [
             "| Tier | Realities | reps x k, k_reality | CPU time (2-thread hours) | Laptop, 1 process | Laptop, 3 processes |",
             "|---|---|---|---|---|---|"]
    for name, fams, reps, k, kr, mult in TIERS:
        secs = mult * sum(run_seconds(timing[f][0], timing[f][1], reps, k, kr) for f in fams)
        lines.append(f"| {name} | {', '.join(fams)} | {reps} x {k}, {kr} | {secs / 3600:.1f} | {secs / 3600:.1f} h | "
                     f"{secs / 3600 / 3:.1f} h |")
    out = OUT_DIR / "cost_estimate.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
