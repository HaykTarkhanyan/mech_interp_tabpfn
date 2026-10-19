"""Delta_B with the review's three fixes, and the pre-registered decision rules.

  Delta_B(C, D) = relative integrated L2 distance (ale.relative_l2) between the behavior of
                  g_D and of g_D with component set C ablated, on the same reference rows
                  and the same fixed grid (a paired comparison).

  (i)   null:        Delta_B of random component sets of matched size and type; a candidate's
                     null percentile = 100 * (#null below + 0.5 * #ties) / #null.
  (ii)  noise floor: Delta_B between the primary B and B recomputed with NO ablation under
                     resampling of the query sample, a jittered grid, a new context draw, and
                     all three at once ("joint"). FLOOR = the largest joint value.
  (iii) selectivity: S = Delta_target / max(Delta_nontarget, FLOOR_nontarget), with target =
                     second-order ALE of (X1, X2), non-target = first-order ALE of X2.

Decision rules (pre-registered 2026-10-19 in README.md and DECISIONS.md before the pilot ran):
  A candidate set PASSES iff Delta_target > FLOOR_target, its Delta_target null percentile is
  >= 95, and S > 1.
  H1 (feature attention carries the interaction, not item attention), per placement:
      no attention set passes -> inconclusive; else the passing attention set with the
      highest S decides: feature attention -> supported, item attention -> contradicted.
  H2 (same pair vs different pairs matters):
      no attention set passes in either placement -> inconclusive;
      identical sets of passing attention components -> not supported;
      the sets differ and at least one flipped set changes Delta_target by more than the
      larger of the two placements' FLOOR_target -> supported; otherwise inconclusive.

Not a script; imported by run_pilot.py and report.py.
"""

from __future__ import annotations

import numpy as np

from ale import ALEPlan, ALEResult, interp_to_grid, relative_l2

TARGET = "ale_12"
NONTARGET = "ale_x2"
BEHAVIORS = ("ale_x1", "ale_x2", "ale_12")


def deltas(ref: dict[str, ALEResult], new: dict[str, ALEResult]) -> dict[str, float]:
    """Relative L2 per behavior; `new` is interpolated onto `ref`'s grid if they differ."""
    out = {}
    for name in BEHAVIORS:
        r, n = ref[name], new[name]
        same = all(np.array_equal(a, b) for a, b in zip(r.edges, n.edges))
        vals = n.values if same else interp_to_grid(n, r.edges)
        out[name] = relative_l2(r.values, vals)
    return out


def prediction_diagnostics(plan: ALEPlan, clean: np.ndarray, ablated: np.ndarray, true_logit_ref: np.ndarray) -> dict:
    """Secondary, prediction-level diagnostics on the unmodified reference rows."""
    c, a = plan.reference_preds(clean), plan.reference_preds(ablated)
    bayes = true_logit_ref > 0
    return {
        "mean_abs_change_logodds": float(np.mean(np.abs(a - c))),
        "acc_vs_bayes_clean": float(np.mean((c > 0) == bayes)),
        "acc_vs_bayes_ablated": float(np.mean((a > 0) == bayes)),
    }


def null_percentile(value: float, null_values) -> float:
    nv = np.asarray(null_values, dtype=float)
    if nv.size == 0:
        raise ValueError("empty null distribution")
    return float(100.0 * (np.sum(nv < value) + 0.5 * np.sum(nv == value)) / nv.size)


def floor_of(floor_runs: list[dict], behavior: str) -> float:
    """FLOOR = the largest no-ablation Delta among the joint resamples.

    floor_runs: records {"source": query|grid|context|joint, "delta": {behavior: value}}.
    """
    joint = [r["delta"][behavior] for r in floor_runs if r["source"] == "joint"]
    if not joint:
        raise ValueError("no joint floor resamples")
    return float(max(joint))


def selectivity(d_target: float, d_nontarget: float, floor_nontarget: float) -> float:
    return float(d_target / max(d_nontarget, floor_nontarget))


def score_sets(placement: dict) -> list[dict]:
    """Add floors, null percentiles, selectivity and pass flags to every mean-ablation scan row."""
    floor_t = floor_of(placement["floor"], TARGET)
    floor_n = floor_of(placement["floor"], NONTARGET)
    nulls: dict[str, list[dict]] = {}
    for r in placement["null"]:
        nulls.setdefault(r["site"], []).append(r)
    rows = []
    for r in placement["scan"]:
        null = nulls[r["site"]]
        d_t, d_n = r["delta"][TARGET], r["delta"][NONTARGET]
        s = selectivity(d_t, d_n, floor_n)
        null_s = [selectivity(x["delta"][TARGET], x["delta"][NONTARGET], floor_n) for x in null]
        pct = null_percentile(d_t, [x["delta"][TARGET] for x in null])
        rows.append({
            **{k: r[k] for k in ("site", "layer")},
            "delta_target": d_t,
            "delta_nontarget": d_n,
            "delta_x1": r["delta"]["ale_x1"],
            "floor_target": floor_t,
            "floor_nontarget": floor_n,
            "above_floor": d_t > floor_t,
            "null_pct_target": pct,
            "null_p95_target": float(np.percentile([x["delta"][TARGET] for x in null], 95)),
            "selectivity": s,
            "null_pct_selectivity": null_percentile(s, null_s),
            "passes": bool(d_t > floor_t and pct >= 95.0 and s > 1.0),
            "pred": r["pred"],
        })
    return rows


def h1_verdict(rows: list[dict]) -> dict:
    passing = [r for r in rows if r["passes"] and r["site"] in ("feat", "item")]
    if not passing:
        return {"verdict": "inconclusive", "reason": "no attention set passes (floor, null p95 and S > 1)"}
    best = max(passing, key=lambda r: r["selectivity"])
    v = "supported" if best["site"] == "feat" else "contradicted"
    return {
        "verdict": v,
        "reason": f"highest-selectivity passing attention set is {best['site']} layer {best['layer']} (S = {best['selectivity']:.2f})",
        "passing": [f"{r['site']} L{r['layer']}" for r in passing],
    }


def h2_verdict(rows_same: list[dict], rows_diff: list[dict]) -> dict:
    def key(r):
        return (r["site"], r["layer"])

    att = ("feat", "item")
    p_same = {key(r) for r in rows_same if r["passes"] and r["site"] in att}
    p_diff = {key(r) for r in rows_diff if r["passes"] and r["site"] in att}
    if not p_same and not p_diff:
        return {"verdict": "inconclusive", "reason": "no attention set passes in either placement"}
    if p_same == p_diff:
        return {"verdict": "not supported", "reason": f"same passing attention sets in both placements: {sorted(p_same)}"}
    floor = max(rows_same[0]["floor_target"], rows_diff[0]["floor_target"])
    by_same, by_diff = {key(r): r for r in rows_same}, {key(r): r for r in rows_diff}
    flipped = sorted(p_same ^ p_diff)
    big = [k for k in flipped if abs(by_same[k]["delta_target"] - by_diff[k]["delta_target"]) > floor]
    detail = {"pass_same": sorted(p_same), "pass_diff": sorted(p_diff), "flipped": flipped, "flipped_beyond_floor": big}
    if big:
        return {"verdict": "supported", "reason": f"{len(big)} flipped set(s) differ by more than the floor {floor:.3f}", **detail}
    return {"verdict": "inconclusive", "reason": "passing sets differ, but every flip is within the floor", **detail}
