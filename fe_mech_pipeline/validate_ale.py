"""Check that ale.py recovers known effects of closed-form models. Plain asserts.

Run:  uv run python validate_ale.py      (~2 s on the laptop, numpy only)

Cases (iid Uniform(-2, 2) features, equal-width bins):
  1. linear f = 3 x1: first-order ALE is exactly 3 z (up to centring).
  2. additive f = sin(x1) + x2^2: first-order ALEs exact at the edges; second-order ALE is 0.
  3. pure interaction f = x1 x2: second-order ALE ~ x1 x2; first-order ALEs ~ 0.
  4. the pilot's task logit x1 + (x2^2 - 4/3) + 1.5 x1 x2 1{x3 > 0}: first-order ALEs
     recover h1 and h2, second-order ALE ~ 0.75 x1 x2 (the interaction averaged over x3).
  5. utilities: distance identities, bootstrap with identity rows, interpolation,
     empty bins raise.
"""

from __future__ import annotations

import time

import numpy as np

from ale import (
    ALEPlan,
    ALESpec,
    interp_to_grid,
    jitter_edges,
    l2_norm,
    relative_l2,
    uniform_edges,
)
from common import RESULTS_DIR, SEED, setup, write_json
from tasks import TASKS, true_logit_semantic

N = 50_000
K1, K2 = 8, 5


def make_plan(x: np.ndarray) -> ALEPlan:
    e1, e2 = uniform_edges(-2, 2, K1), uniform_edges(-2, 2, K2)
    specs = [
        ALESpec("ale_x1", (0,), (e1,)),
        ALESpec("ale_x2", (1,), (e1,)),
        ALESpec("ale_12", (0, 1), (e2, e2)),
    ]
    return ALEPlan(x, specs)


def run(f, x: np.ndarray) -> tuple[ALEPlan, np.ndarray, dict]:
    plan = make_plan(x)
    preds = f(plan.queries)
    return plan, preds, plan.compute(preds)


def main() -> None:
    log = setup("validate_ale")
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-2, 2, size=(N, 4))
    e1, e2 = uniform_edges(-2, 2, K1), uniform_edges(-2, 2, K2)
    z1a, z2a = np.meshgrid(e2, e2, indexing="ij")
    res: dict = {"n_rows": N, "k1": K1, "k2": K2, "checks": []}

    # 1. linear
    _, _, ale = run(lambda q: 3.0 * q[:, 0], x)
    err = relative_l2(3.0 * e1, ale["ale_x1"].values)
    log.info(f"[1] linear 3*x1: relative L2 error of ALE_x1 = {err:.2e}")
    assert err < 1e-12, err
    res["checks"].append({"case": "linear 3*x1", "metric": "rel L2 error of ALE_x1", "value": err, "tolerance": 1e-12})
    n = ale["ale_x1"].counts
    expected = 3.0 * e1 - np.sum(n * 3.0 * (e1[:-1] + e1[1:]) / 2) / n.sum()
    assert np.allclose(ale["ale_x1"].values, expected, atol=1e-12)
    assert l2_norm(ale["ale_12"].values) < 1e-12

    # 2. additive
    _, _, ale = run(lambda q: np.sin(q[:, 0]) + q[:, 1] ** 2, x)
    e_x1 = relative_l2(np.sin(e1), ale["ale_x1"].values)
    e_x2 = relative_l2(e1**2, ale["ale_x2"].values)
    n12 = l2_norm(ale["ale_12"].values)
    log.info(f"[2] additive: rel err x1 {e_x1:.2e}, x2 {e_x2:.2e}; ||ALE_12|| = {n12:.2e}")
    assert e_x1 < 1e-12 and e_x2 < 1e-12, (e_x1, e_x2)
    assert n12 < 1e-12, n12
    res["checks"] += [{"case": "additive sin(x1)+x2^2", "metric": "rel L2 error of ALE_x1", "value": e_x1, "tolerance": 1e-12},
                       {"case": "additive sin(x1)+x2^2", "metric": "rel L2 error of ALE_x2", "value": e_x2, "tolerance": 1e-12},
                       {"case": "additive sin(x1)+x2^2", "metric": "norm of ALE_12 (should be 0)", "value": n12, "tolerance": 1e-12}]

    # 3. pure interaction
    _, _, ale = run(lambda q: q[:, 0] * q[:, 1], x)
    e12 = relative_l2(z1a * z2a, ale["ale_12"].values)
    m1 = np.abs(ale["ale_x1"].values).max()
    m2 = np.abs(ale["ale_x2"].values).max()
    log.info(f"[3] x1*x2: rel err ALE_12 {e12:.3e}; max|ALE_x1| {m1:.3e}, max|ALE_x2| {m2:.3e}")
    assert e12 < 0.02, e12
    assert m1 < 0.05 and m2 < 0.05, (m1, m2)
    res["checks"] += [{"case": "pure interaction x1*x2", "metric": "rel L2 error of ALE_12", "value": e12, "tolerance": 0.02},
                       {"case": "pure interaction x1*x2", "metric": "max |ALE_x1| (should be ~0)", "value": float(m1), "tolerance": 0.05}]

    # 4. the pilot's task logit
    spec = TASKS["interaction"]
    plan, preds, ale = run(lambda q: true_logit_semantic(spec, q), x)
    e_x1 = relative_l2(e1, ale["ale_x1"].values)
    e_x2 = relative_l2(e1**2 - 4 / 3, ale["ale_x2"].values)
    e12 = relative_l2(0.75 * z1a * z2a, ale["ale_12"].values)
    log.info(f"[4] task logit: rel err x1 {e_x1:.3e}, x2 {e_x2:.3e}, 12 {e12:.3e}")
    assert e_x1 < 0.03 and e_x2 < 0.03 and e12 < 0.05, (e_x1, e_x2, e12)
    res["checks"] += [{"case": "pilot task logit", "metric": "rel L2 error of ALE_x1 vs x1", "value": e_x1, "tolerance": 0.03},
                       {"case": "pilot task logit", "metric": "rel L2 error of ALE_x2 vs x2^2-4/3", "value": e_x2, "tolerance": 0.03},
                       {"case": "pilot task logit", "metric": "rel L2 error of ALE_12 vs 0.75 x1 x2", "value": e12, "tolerance": 0.05}]
    # the flat variant: main effects vanish, interaction stays
    _, _, ale_flat = run(lambda q: true_logit_semantic(TASKS["flat"], q), x)
    r_main = l2_norm(ale_flat["ale_x2"].values) / l2_norm(ale_flat["ale_12"].values)
    log.info(f"[4b] flat task: ||ALE_x2|| / ||ALE_12|| = {r_main:.3e}")
    assert r_main < 0.05, r_main
    res["checks"].append({"case": "flat task logit", "metric": "||ALE_x2|| / ||ALE_12||", "value": r_main, "tolerance": 0.05})
    # the additive control: no interaction
    _, _, ale_add = run(lambda q: true_logit_semantic(TASKS["additive"], q), x)
    assert l2_norm(ale_add["ale_12"].values) < 1e-12

    # 5. utilities
    same = plan.compute(preds, rows=np.arange(plan.n_ref))
    assert all(np.array_equal(same[k].values, ale[k].values) for k in ale)
    boot = plan.compute(preds, rows=np.random.default_rng(SEED).integers(0, plan.n_ref, plan.n_ref))
    log.info(f"[5] bootstrap resample moves ALE_12 by rel {relative_l2(ale['ale_12'].values, boot['ale_12'].values):.3e}")
    v = ale["ale_12"].values
    assert relative_l2(v, v) == 0.0
    assert abs(relative_l2(v, 2 * v) - 1.0) < 1e-12
    assert relative_l2(v, v + 5.0) < 1e-12  # constant offsets are ignored
    assert np.allclose(interp_to_grid(ale["ale_x1"], [e1]), ale["ale_x1"].values)
    assert np.allclose(interp_to_grid(ale["ale_12"], [e2, e2]), v)
    ej = jitter_edges(e1, np.random.default_rng(SEED), 0.3)
    xj = ALEPlan(x, [ALESpec("ale_x1", (0,), (ej,))])
    lin = xj.compute(3.0 * xj.queries[:, 0])["ale_x1"]
    back = interp_to_grid(lin, [e1])
    assert relative_l2(3.0 * e1, back) < 1e-12  # linear survives grid jitter exactly
    tiny = make_plan(x[:5])
    try:
        tiny.compute(np.zeros(tiny.queries.shape[0]))
    except ValueError as exc:
        assert "empty" in str(exc), exc
        log.info(f"[5] empty bins raise as intended: {exc}")
    else:
        raise AssertionError("empty bins did not raise")

    res["elapsed_s"] = time.perf_counter() - t0
    res["all_passed"] = True
    write_json(RESULTS_DIR / "validate_ale.json", res)
    log.info(f"ALL ALE CHECKS PASSED in {res['elapsed_s']:.2f} s; wrote results/validate_ale.json")


if __name__ == "__main__":
    main()
