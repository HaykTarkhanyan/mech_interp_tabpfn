#!/usr/bin/env python3
"""Realism-tractability frontier: how much of a real regression task can target families with exact explanations
capture? (Exploration, 2026-10-27; ChatGPT's "Gap 2" in chatgpt_conference_submission_plan.md.)

For each dataset, five families fitted to the real labels, scored by held-out R^2 against the real labels (3-fold):
- linear            RidgeCV on standardized features: exact SHAP in closed form;
- additive splines  the transfer pilot's "additive" target (quantile-knot splines + RidgeCV);
- additive boosted  HistGradientBoosting with interaction_cst="no_interactions": a sum of one-feature functions;
- pairwise boosted  HistGradientBoosting with interaction_cst="pairwise": every leaf depends on at most two features;
- full boosted      HistGradientBoosting without constraints (the reference for "how much signal there is").
Exact interventional SHAP of an order-2 model needs only coalitions of size <= 2 (pairwise_truth below), so the
pairwise family keeps exact ground truth at d^2/2 background averages instead of 2^d. The script checks that
identity against shap's exact enumeration and checks that third-order Moebius terms of the fitted pairwise model
vanish, then times exact SHAP per explained point for each family.

Usage (from synthetic_data/):
    .venv/Scripts/python.exe scripts/frontier_probe.py            # ~8 min on 2 threads (estimate); OpenML downloads
Writes results/frontier/frontier.json; scripts/frontier_report.py builds the HTML report from it.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import SplineTransformer, StandardScaler  # noqa: E402

SEED = 509
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "frontier"
# name -> OpenML data id (looked up 2026-10-27 through the OpenML API: active, all-numeric regression sets)
OPENML = {"wine_quality": 287, "cpu_act": 197, "kin8nm": 189, "house_16H": 574, "airfoil_self_noise": 44957,
          "concrete_compressive_strength": 44959, "energy_efficiency": 43918, "elevators": 216}
FAMILIES = ["linear", "additive splines", "additive boosted", "pairwise boosted", "full boosted"]


def make_model(family: str):
    hgb = dict(max_iter=500, learning_rate=0.1, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
               random_state=SEED)
    return {
        "linear": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "additive splines": lambda: make_pipeline(SplineTransformer(n_knots=6, degree=3, knots="quantile"),
                                                  RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "additive boosted": lambda: HistGradientBoostingRegressor(interaction_cst="no_interactions", **hgb),
        "pairwise boosted": lambda: HistGradientBoostingRegressor(interaction_cst="pairwise", **hgb),
        "full boosted": lambda: HistGradientBoostingRegressor(**hgb),
    }[family]()


# ---------------------------------------------------------------- exact interventional SHAP for order <= 2
def coalition_values(f, x_e: np.ndarray, bg: np.ndarray, subsets: list) -> dict:
    """v(S) = mean_b f(x_S, b_rest) for every point in x_e and every S in subsets."""
    n_e, n_b, d = len(x_e), len(bg), x_e.shape[1]
    out = {}
    for s in subsets:
        z = np.repeat(bg[None, :, :], n_e, axis=0)
        if s:
            idx = list(s)
            z[:, :, idx] = x_e[:, None, idx]
        out[s] = f(z.reshape(-1, d)).reshape(n_e, n_b).mean(1)
    return out


def pairwise_truth(f, x_e: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Exact interventional SHAP of a model whose terms each depend on at most two features.

    Moebius terms of the game v(S) = E_b f(x_S, b_rest) above order 2 vanish for such a model, so
    phi_i = m_i + 1/2 sum_k m_ik with m_i = v(i) - v(0) and m_ik = v(ik) - v(i) - v(k) + v(0).
    """
    d = x_e.shape[1]
    pairs = list(itertools.combinations(range(d), 2))
    v = coalition_values(f, x_e, bg, [()] + [(i,) for i in range(d)] + pairs)
    phi = np.stack([v[(i,)] - v[()] for i in range(d)], axis=1)
    for i, k in pairs:
        m = v[(i, k)] - v[(i,)] - v[(k,)] + v[()]
        phi[:, i] += m / 2
        phi[:, k] += m / 2
    return phi


def additive_truth(f, x_e: np.ndarray, bg: np.ndarray) -> np.ndarray:
    d = x_e.shape[1]
    v = coalition_values(f, x_e, bg, [()] + [(i,) for i in range(d)])
    return np.stack([v[(i,)] - v[()] for i in range(d)], axis=1)


def max_third_order(f, x_e: np.ndarray, bg: np.ndarray, triples: list) -> float:
    """Largest |third-order Moebius term| over the given triples; zero (up to rounding) for an order-2 model."""
    subsets = {()}
    for t in triples:
        for r in range(1, 4):
            subsets.update(itertools.combinations(t, r))
    v = coalition_values(f, x_e, bg, sorted(subsets, key=len))
    worst = 0.0
    for i, j, k in triples:
        m = (v[(i, j, k)] - v[(i, j)] - v[(i, k)] - v[(j, k)] + v[(i,)] + v[(j,)] + v[(k,)] - v[()])
        worst = max(worst, float(np.abs(m).max()))
    return worst


# ---------------------------------------------------------------- data
def load(name: str) -> tuple:
    if name == "california_housing":
        d = fetch_california_housing()
        return d.data.astype(float), d.target.astype(float)
    if name == "diabetes":
        d = load_diabetes()
        return d.data.astype(float), d.target.astype(float)
    d = fetch_openml(data_id=OPENML[name], as_frame=True, parser="auto")
    x = d.data
    if any(str(t) in ("category", "object") for t in x.dtypes):
        raise ValueError(f"{name}: non-numeric feature columns {list(x.select_dtypes(exclude='number').columns)}")
    return x.to_numpy(dtype=float), np.asarray(d.target, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["california_housing", "diabetes", *OPENML])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--n-explain", type=int, default=20)
    ap.add_argument("--n-bg", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=20000, help="subsample larger data sets")
    args = ap.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "frontier_probe.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("frontier")
    started = time.time()
    rng = np.random.default_rng(SEED)
    results = []
    for di, name in enumerate(args.datasets):
        x, y = load(name)
        keep = np.isfinite(x).all(1) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if len(x) > args.max_rows:
            idx = rng.choice(len(x), args.max_rows, replace=False)
            x, y = x[idx], y[idx]
        rec = {"dataset": name, "n": int(len(x)), "d": int(x.shape[1]), "families": {}}
        fitted = {}
        for fam in FAMILIES:
            r2, secs = [], []
            for k, (tr, te) in enumerate(KFold(args.folds, shuffle=True, random_state=SEED).split(x)):
                t0 = time.time()
                m = make_model(fam).fit(x[tr], y[tr])
                secs.append(time.time() - t0)
                pred = m.predict(x[te])
                r2.append(float(1 - np.mean((pred - y[te]) ** 2) / np.var(y[te])))
                if k == 0:
                    fitted[fam] = (m, tr, te)
            rec["families"][fam] = {"r2_mean": float(np.mean(r2)), "r2_sd": float(np.std(r2)), "r2_folds": r2,
                                    "fit_seconds_mean": float(np.mean(secs))}
        # exact SHAP of each tractable family on the first fold's model, timed per explained point
        m_pair, tr, te = fitted["pairwise boosted"]
        bg, x_e = x[rng.choice(tr, args.n_bg, replace=False)], x[te[: args.n_explain]]
        for fam, fn in [("additive boosted", additive_truth), ("pairwise boosted", pairwise_truth)]:
            t0 = time.time()
            phi = fn(fitted[fam][0].predict, x_e, bg)
            rec["families"][fam]["exact_shap_seconds_per_point"] = (time.time() - t0) / len(x_e)
            gap = float(np.abs(phi.sum(1) - (fitted[fam][0].predict(x_e) - fitted[fam][0].predict(bg).mean())).max())
            rec["families"][fam]["efficiency_gap"] = gap
        d = x.shape[1]
        triples = [tuple(sorted(rng.choice(d, 3, replace=False))) for _ in range(5)]
        rec["pairwise_max_third_order_moebius"] = max_third_order(m_pair.predict, x_e[:5], bg[:100], triples)
        rec["target_sd"] = float(np.std(y))
        if d <= 10:   # 2^d coalitions: affordable cross-check against shap's exact enumeration
            ex = shap.explainers.Exact(m_pair.predict, shap.maskers.Independent(bg[:100], max_samples=100))(x_e[:3]).values
            rec["pairwise_vs_shap_exact_max_abs"] = float(np.abs(pairwise_truth(m_pair.predict, x_e[:3], bg[:100]) - ex).max())
        results.append(rec)
        fam_r2 = ", ".join(f"{f} {rec['families'][f]['r2_mean']:.3f}" for f in FAMILIES)
        el = time.time() - started
        log.info(f"{di + 1}/{len(args.datasets)} {name} (n={rec['n']}, d={d}): {fam_r2}; third-order "
                 f"{rec['pairwise_max_third_order_moebius']:.1e}; {el:.0f}s elapsed, "
                 f"~{el / (di + 1) * (len(args.datasets) - di - 1):.0f}s left")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "frontier.json"
    out.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "args": vars(args),
                               "families": FAMILIES, "openml_ids": OPENML,
                               "versions": {"sklearn": sklearn.__version__, "shap": shap.__version__,
                                            "numpy": np.__version__},
                               "runtime_s": round(time.time() - started, 1), "datasets": results}, indent=1),
                   encoding="utf-8")
    log.info(f"wrote {out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
