"""Pilot: layer scan of TabPFN v2 against the X1*X2 interaction, with null, floor, selectivity.

Run:  uv run python run_pilot.py            (measured 13.0 min on 2026-10-19 with the CPU shared
                                              with other processes, 293 passes at 2.65 s; an idle
                                              laptop should need ~5 min, estimate)
      uv run python run_pilot.py --smoke    (~2 min, wiring check: 2 layers, tiny null)
      add --device cuda on a GPU machine (RUNPOD.md)

What it does (task "interaction", gamma = 1.5; context of 256 rows; seed 509):
  1. behavior check: model ALEs vs the true ALEs for the additive, interaction and flat tasks
  2. for each column placement (same_pair: X1, X2 in columns 0, 1; diff_pair: columns 0, 2),
     on the same context rows and reference rows with columns permuted:
       - clean behavior B(g_D): ALE of X1, ALE of X2, second-order ALE of (X1, X2)
       - layer scan, mean ablation: all 6 feature-attention heads of layer L, all 6
         item-attention heads of layer L, all 768 MLP units of layer L, for L = 0..11
       - null: random sets of 6 feature-attention heads / 6 item-attention heads / 768 MLP
         units drawn across all layers (the same random sets for both placements)
       - noise floor with no ablation: new query sample, jittered grid, new context, all three
  3. zero-ablation contrast of the layer scan (same_pair only)

Raw results go to results/pilot.json (results/pilot_smoke.json with --smoke). The report is
built from that file by report.py; decision rules live in delta_b.py.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import model as M
from ale import ALEPlan, ALESpec, jitter_edges, l2_norm, relative_l2, uniform_edges
from common import RESULTS_DIR, SEED, setup, write_json
from delta_b import BEHAVIORS, deltas, prediction_diagnostics
from tasks import (
    HIGH,
    LOW,
    PLACEMENTS,
    TASKS,
    model_column_of,
    sample_labels,
    sample_semantic,
    to_model_columns,
    to_semantic_columns,
    true_logit_semantic,
)

SIZE = {"feat": M.N_HEADS, "item": M.N_HEADS, "mlp": M.D_MLP}


def rng_for(purpose: int) -> np.random.Generator:
    return np.random.default_rng([SEED, purpose])


def stratified_reference(n_per_cell: int, k2: int, rng: np.random.Generator) -> np.ndarray:
    """Reference rows (semantic order) with exactly n_per_cell rows in every (X1, X2) cell.

    Cells have equal probability under the uniform distribution, so this is proportional
    stratified sampling from P_X: the ALE estimand is unchanged and no cell is ever empty.
    X3 and N1 are not stratified.
    """
    e = uniform_edges(LOW, HIGH, k2)
    rows = []
    for a in range(k2):
        for b in range(k2):
            block = rng.uniform(LOW, HIGH, size=(n_per_cell, 4))
            block[:, 0] = rng.uniform(e[a], e[a + 1], n_per_cell)
            block[:, 1] = rng.uniform(e[b], e[b + 1], n_per_cell)
            rows.append(block)
    xs = np.vstack(rows)
    return xs[rng.permutation(len(xs))]


def make_specs(placement: str, e1: np.ndarray, e2: np.ndarray) -> list[ALESpec]:
    c1, c2 = model_column_of("X1", placement), model_column_of("X2", placement)
    return [
        ALESpec("ale_x1", (c1,), (e1,)),
        ALESpec("ale_x2", (c2,), (e1,)),
        ALESpec("ale_12", (c1, c2), (e2, e2)),
    ]


def ale_values(b: dict) -> dict:
    return {k: b[k].values for k in BEHAVIORS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--n-per-cell", type=int, default=8, help="reference rows per (X1, X2) cell")
    ap.add_argument("--k1", type=int, default=8, help="bins for first-order ALE")
    ap.add_argument("--k2", type=int, default=4, help="bins per axis for second-order ALE")
    ap.add_argument("--n-null", type=int, default=25, help="random sets per component type")
    ap.add_argument("--floor", type=str, default="3,3,3,7", help="resamples: query,grid,context,joint")
    ap.add_argument("--jitter", type=float, default=0.15, help="grid jitter, fraction of a bin width")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    name = "run_pilot_smoke" if args.smoke else "run_pilot"
    log = setup(name)
    t_start = time.perf_counter()
    layers = [0, 6] if args.smoke else list(range(M.N_LAYERS))
    n_null = 2 if args.smoke else args.n_null
    n_floor = dict(zip(("query", "grid", "context", "joint"), (1, 1, 1, 1) if args.smoke else map(int, args.floor.split(","))))
    spec = TASKS["interaction"]
    e1, e2 = uniform_edges(LOW, HIGH, args.k1), uniform_edges(LOW, HIGH, args.k2)

    # shared draws (semantic column order), identical for both placements
    ctx_s, y_ctx = sample_semantic(spec, args.n_ctx, rng_for(1))
    ref_s = stratified_reference(args.n_per_cell, args.k2, rng_for(2))
    true_ref = true_logit_semantic(spec, ref_s)
    rng_null = rng_for(3)
    null_sets = {site: [M.random_units(site, SIZE[site], rng_null) for _ in range(n_null)] for site in M.SITES}
    rng_floor = rng_for(4)
    floor_draws = []
    for source, n in n_floor.items():
        for _ in range(n):
            d = {"source": source}
            if source in ("context", "joint"):
                d["ctx"] = sample_semantic(spec, args.n_ctx, rng_floor)
            if source in ("query", "joint"):
                d["ref"] = stratified_reference(args.n_per_cell, args.k2, rng_floor)
            if source in ("grid", "joint"):
                d["e1"], d["e2"] = jitter_edges(e1, rng_floor, args.jitter), jitter_edges(e2, rng_floor, args.jitter)
            floor_draws.append(d)

    clf, net = M.load_v2(to_model_columns(ctx_s, "same_pair"), y_ctx, device=args.device)
    hooked = M.HookedTabPFN(net)
    ckpt = M.checkpoint_path()
    out: dict = {
        "meta": {
            "started": datetime.now().isoformat(timespec="seconds"),
            "smoke": args.smoke,
            "seed": SEED,
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "device": args.device,
            "checkpoint": {"file": M.CKPT_NAME, "repo": M.HF_REPO, "sha256": M.sha256(ckpt),
                           "n_params": sum(p.numel() for p in net.parameters())},
            "task": spec.to_dict(),
            "feature_distribution": f"iid Uniform({LOW}, {HIGH}); columns X1, X2, X3, N1",
            "placements": {k: [("X1", "X2", "X3", "N1")[i] for i in v] for k, v in PLACEMENTS.items()},
            "n_ctx": args.n_ctx,
            "n_ref": int(len(ref_s)),
            "n_per_cell": args.n_per_cell,
            "k1": args.k1,
            "k2": args.k2,
            "jitter": args.jitter,
            "n_null": n_null,
            "n_floor": n_floor,
            "layers": layers,
            "prediction_scale": "log-odds (logit_1 - logit_0, temperature 1)",
            "ablation_primary": "mean (per token column, context vs query rows)",
        },
        "behavior_check": {},
        "placements": {},
    }

    # job list: (placement, section, payload)
    jobs: list[tuple[str, str, dict]] = []
    for p in PLACEMENTS:
        for site in M.SITES:
            for layer in layers:
                jobs.append((p, "scan", {"site": site, "layer": layer, "kind": "mean"}))
        for site in M.SITES:
            for i, units in enumerate(null_sets[site]):
                jobs.append((p, "null", {"site": site, "i": i, "units": units}))
        for i, d in enumerate(floor_draws):
            jobs.append((p, "floor", {"i": i, **d}))
    for site in M.SITES:
        for layer in layers:
            jobs.append(("same_pair", "zero_scan", {"site": site, "layer": layer, "kind": "zero"}))
    n_passes = len(jobs) + 2 * len(PLACEMENTS) + len(TASKS)  # + clean and means per placement + behavior check
    log.info(f"{n_passes} forward passes planned ({len(jobs)} jobs); context {args.n_ctx}, "
             f"{len(ref_s)} reference rows -> {9 * len(ref_s)} query rows per pass")

    # 1. behavior check (same_pair placement, same context rows and reference rows)
    # same context X rows for every task; only the labels follow each task's own logit
    plan_bc = ALEPlan(to_model_columns(ref_s, "same_pair"), make_specs("same_pair", e1, e2))
    x_ctx_sp = to_model_columns(ctx_s, "same_pair")
    for ti, (tname, tspec) in enumerate(TASKS.items()):
        t = time.perf_counter()
        y_t = y_ctx if tname == "interaction" else sample_labels(tspec, ctx_s, rng_for(10 + ti))
        b_model = plan_bc.compute(hooked.logodds(x_ctx_sp, y_t, plan_bc.queries))
        b_true = plan_bc.compute(true_logit_semantic(tspec, to_semantic_columns(plan_bc.queries, "same_pair")))
        rec = {"spec": tspec.to_dict(), "model": ale_values(b_model), "truth": ale_values(b_true), "norm_model": {}, "norm_truth": {}, "rel_dist_to_truth": {}}
        for k in BEHAVIORS:
            rec["norm_model"][k] = l2_norm(b_model[k].values)
            rec["norm_truth"][k] = l2_norm(b_true[k].values)
            rec["rel_dist_to_truth"][k] = relative_l2(b_true[k].values, b_model[k].values) if rec["norm_truth"][k] > 1e-9 else None
        rec["seconds"] = time.perf_counter() - t
        out["behavior_check"][tname] = rec
        log.info(f"behavior check {tname}: done in {rec['seconds']:.1f} s")
    out["grid"] = {"e1": e1, "e2": e2}

    # 2. clean behavior and means per placement
    state = {}
    for p in PLACEMENTS:
        x_ctx = to_model_columns(ctx_s, p)
        plan = ALEPlan(to_model_columns(ref_s, p), make_specs(p, e1, e2))
        clean = hooked.logodds(x_ctx, y_ctx, plan.queries)
        b0 = plan.compute(clean)
        means = hooked.capture_means(x_ctx, y_ctx, plan.x_ref)
        state[p] = {"x_ctx": x_ctx, "plan": plan, "clean": clean, "b0": b0, "means": means}
        out["placements"][p] = {
            "clean": {"ale": ale_values(b0), "norm": {k: l2_norm(b0[k].values) for k in BEHAVIORS},
                      "acc_vs_bayes": float(np.mean((plan.reference_preds(clean) > 0) == (true_ref > 0)))},
            "scan": [], "null": [], "floor": [], "zero_scan": [],
        }
        log.info(f"{p}: clean behavior computed; ||ALE_12|| = {l2_norm(b0['ale_12'].values):.3f}")

    # 3. all ablation and floor jobs
    t_jobs = time.perf_counter()
    for j, (p, section, job) in enumerate(tqdm(jobs, desc="pilot", mininterval=5)):
        st = state[p]
        t = time.perf_counter()
        if section == "floor":
            x_ctx, y = (to_model_columns(job["ctx"][0], p), job["ctx"][1]) if "ctx" in job else (st["x_ctx"], y_ctx)
            ref = job.get("ref", ref_s)
            ee1, ee2 = job.get("e1", e1), job.get("e2", e2)
            plan = ALEPlan(to_model_columns(ref, p), make_specs(p, ee1, ee2))
            b = plan.compute(hooked.logodds(x_ctx, y, plan.queries))
            rec = {"source": job["source"], "i": job["i"], "delta": deltas(st["b0"], b)}
        else:
            units = job["units"] if section == "null" else M.layer_units(job["site"], job["layer"])
            kind = "zero" if section == "zero_scan" else "mean"
            preds = hooked.logodds(st["x_ctx"], y_ctx, st["plan"].queries, units=units, kind=kind, source=st["means"])
            b = st["plan"].compute(preds)
            rec = {"site": job["site"], "kind": kind, "delta": deltas(st["b0"], b),
                   "pred": prediction_diagnostics(st["plan"], st["clean"], preds, true_ref)}
            if section == "null":
                rec.update({"i": job["i"], "units": M.describe_units(units)})
            else:
                rec.update({"layer": job["layer"], "ale": ale_values(b)})
        rec["seconds"] = time.perf_counter() - t
        out["placements"][p][section].append(rec)
        if (j + 1) % 20 == 0 or j + 1 == len(jobs):
            el = time.perf_counter() - t_jobs
            log.info(f"{j + 1}/{len(jobs)} jobs, {el:.0f} s elapsed, ~{el / (j + 1) * (len(jobs) - j - 1):.0f} s left")

    out["meta"]["elapsed_s"] = time.perf_counter() - t_start
    out["meta"]["mean_seconds_per_job"] = (time.perf_counter() - t_jobs) / len(jobs)
    path = RESULTS_DIR / ("pilot_smoke.json" if args.smoke else "pilot.json")
    write_json(path, out)
    log.info(f"wrote {path.name}; {len(jobs)} jobs; elapsed {out['meta']['elapsed_s']:.0f} s "
             f"({out['meta']['mean_seconds_per_job']:.2f} s per job)")


if __name__ == "__main__":
    main()
