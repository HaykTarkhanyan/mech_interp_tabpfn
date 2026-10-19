#!/usr/bin/env python3
"""Transfer pilot v2: does an explainer ranking measured in a data-fitted synthetic world reproduce the ranking
in the world the data came from? (DECISIONS.md #22, redesign #24.)

Design (nested known-truth, after Advani et al. 2019):
- Reality: the real feature rows of California housing (standardized; every world's values are snapped to a
  2^-12 grid, see below) and a known true function fitted to the real target: a random forest
  (`--reality rf`) or an MLP (`--reality mlp`, a family no builder can choose), plus noise with the fit's
  residual sd. Reality's reference ranking is the mean over --k-reality tasks.
- A builder sees one sample D of 2,000 rows and builds worlds from it: feature generator (independent columns,
  Gaussian with D's covariance, row bootstrap, ARF) x target fitted on D (additive splines, random forest, MLP).
  The bootstrap world trains on one part of D and explains points from the other.
- Comparators: the oracle (K fresh reality tasks, the agreement task noise alone allows) and a generic null
  world that never sees D (Gaussian features with pairwise correlation 0.5 and a nonlinear additive target, as
  in XAI-BENCH).
- Every world's ranking is the mean over K tasks. A task trains a gradient-boosting model g on 1,000 rows,
  explains 30 points with nine explainers, and scores each against interventional SHAP of g (model level,
  primary) and of the world's true function (data level), both with a 1,000-row background.

Ground truth: interventional TreeSHAP for trees. shap rounds split thresholds to float32, so a threshold between
two nearly equal values can send a point the other way than scikit-learn does; on raw California rows TreeSHAP
missed exact enumeration by up to 2.2e-2. Every feature value in every world is therefore snapped to a 2^-12
grid (standardized units, a 1e-4 sd change), on which all values and split midpoints are exact in float32 and
TreeSHAP matched exact enumeration to 3.3e-8 on 8 models. The code still raises if any row loses additivity by
more than 1e-6 or two rows differ from exact enumeration by more than 1e-6. Additive functions use  the exact
identity phi_j = mean_b f(b with b_j = x_j) - mean_b f(b), and exact enumeration through
model.predict (500-row background) for MLPs.

Usage (from synthetic_data/; threads capped at 2):
    .venv/Scripts/python.exe scripts/transfer_pilot_v2.py --timing --tag timing_rf      # one task per world type
    .venv/Scripts/python.exe scripts/transfer_pilot_v2.py --reality rf --reps 8 --k 5 --k-reality 30 --tag v2_rf
Runtime: measured by --timing; see results/transfer_pilot/cost_estimate.md.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
from arfpy import arf as arfpy_arf  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from sklearn.datasets import fetch_california_housing  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import SplineTransformer  # noqa: E402
from tqdm import tqdm  # noqa: E402

SEED = 509
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "transfer_pilot"
CONFIG = {
    "dataset": "california_housing",
    "grid": 2.0 ** -12,         # all feature values are multiples of this (see the module docstring)
    "n_D": 2000,                # rows the builder sees
    "bootstrap_train_share": 0.7,
    "n_train": 1000,            # training rows of the explained model g, in every world
    "n_explain": 30,
    "n_bg_explainer": 50,
    "n_bg_truth": 1000,
    "n_bg_truth_exact": 500,    # MLP truths, enumerated through predict
    "tree_additivity_tol": 1e-6,
    "tree_crosscheck_tol": 1e-6,
    "kernel_nsamples": 200,
    "permutation_evals": 500,
    "partition_evals": 500,
    "lime_samples": 2000,
    "null_rho": 0.5,
}
EXPLAINERS = ["TreeSHAP-int (bg 50)", "TreeSHAP-path", "Saabas", "KernelSHAP (kmeans 10)", "Permutation SHAP",
              "Owen (correlation tree)", "Mean-baseline SHAP", "LIME", "Occlusion (mean)"]
GENERATORS = ["independent", "gaussian", "bootstrap", "arf"]
TARGETS = ["additive", "forest", "mlp"]


@dataclass
class World:
    name: str
    sample: Callable[[int, np.random.Generator, str], np.ndarray]   # (n, rng, role) with role "train"/"explain"
    f: object
    sigma: float
    truth: str          # "tree", "additive" or "exact"


def setup_logging(tag: str) -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "transfer_pilot_v2.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.info(f"run {tag}")


def snap(x: np.ndarray) -> np.ndarray:
    """Round to the float32-exact grid, so split thresholds (midpoints) are exact in float32 as well."""
    return np.round(np.asarray(x, dtype=float) / CONFIG["grid"]) * CONFIG["grid"]


# ---------------------------------------------------------------- models and targets
def forest(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=30, max_depth=8, min_samples_leaf=20, random_state=seed, n_jobs=1)


def mlp(seed: int) -> MLPRegressor:
    return MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, early_stopping=True, random_state=seed)


def additive(seed: int):
    # quantile knots: evenly spaced knots leave heavy-tailed features almost linear
    return make_pipeline(SplineTransformer(n_knots=6, degree=3, knots="quantile"), RidgeCV(alphas=np.logspace(-3, 3, 13)))


TARGET_MAKERS = {"additive": (additive, "additive"), "forest": (forest, "tree"), "mlp": (mlp, "exact")}


class NullTarget:
    """XAI-BENCH-style nonlinear additive function: sin, abs, square, exp, zero, cycling over features."""
    funcs = [np.sin, np.abs, np.square, lambda z: np.exp(z / 2), lambda z: 0.0 * z]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return sum(self.funcs[j % len(self.funcs)](x[:, j]) for j in range(x.shape[1]))


# ---------------------------------------------------------------- ground truth
def tree_truth(model, x_e: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Interventional TreeSHAP on grid-snapped data, checked for additivity and against exact enumeration."""
    e = shap.TreeExplainer(model, data=shap.maskers.Independent(bg, max_samples=len(bg)),
                           feature_perturbation="interventional")
    phi = e.shap_values(x_e, check_additivity=False)
    gap = float(np.abs(phi.sum(1) + e.expected_value - model.predict(x_e)).max())
    if gap > CONFIG["tree_additivity_tol"]:
        raise RuntimeError(f"interventional TreeSHAP lost additivity by {gap:.2e}; is the data on the grid?")
    sub = bg[:100]
    tree_sub = shap.TreeExplainer(model, data=shap.maskers.Independent(sub, max_samples=100),
                                  feature_perturbation="interventional").shap_values(x_e[:2], check_additivity=False)
    exact_sub = shap.explainers.Exact(model.predict, shap.maskers.Independent(sub, max_samples=100))(x_e[:2]).values
    diff = float(np.abs(tree_sub - exact_sub).max())
    if diff > CONFIG["tree_crosscheck_tol"]:
        raise RuntimeError(f"interventional TreeSHAP differs from exact enumeration by {diff:.2e}")
    return phi


def additive_truth(model, x_e: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Exact interventional SHAP of an additive function: phi_j = mean_b f(b, b_j := x_j) - mean_b f(b)."""
    base = model.predict(bg).mean()
    phi = np.empty_like(x_e, dtype=float)
    for j in range(x_e.shape[1]):
        z = np.repeat(bg[None, :, :], len(x_e), axis=0)
        z[:, :, j] = x_e[:, j][:, None]
        phi[:, j] = model.predict(z.reshape(-1, x_e.shape[1])).reshape(len(x_e), len(bg)).mean(1) - base
    return phi


def exact_truth(model, x_e: np.ndarray, bg: np.ndarray) -> np.ndarray:
    return shap.explainers.Exact(model.predict, shap.maskers.Independent(bg, max_samples=len(bg)))(x_e).values


def truth(world: World, model, kind: str, x_e: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if kind == "exact":
        return exact_truth(model, x_e, world.sample(CONFIG["n_bg_truth_exact"], rng, "explain"))
    bg = world.sample(CONFIG["n_bg_truth"], rng, "explain")
    return tree_truth(model, x_e, bg) if kind == "tree" else additive_truth(model, x_e, bg)


# ---------------------------------------------------------------- explainers (all applied to g)
def explain_all(g, x_e: np.ndarray, bg: np.ndarray, x_tr: np.ndarray, seed: int) -> tuple:
    out, secs = {}, {}

    def timed(name, fn):
        t = time.time()
        out[name] = np.asarray(fn(), dtype=float)
        secs[name] = round(time.time() - t, 3)

    timed("TreeSHAP-int (bg 50)", lambda: shap.TreeExplainer(
        g, data=shap.maskers.Independent(bg, max_samples=len(bg)), feature_perturbation="interventional"
    ).shap_values(x_e, check_additivity=False))
    path = shap.TreeExplainer(g, feature_perturbation="tree_path_dependent")
    timed("TreeSHAP-path", lambda: path.shap_values(x_e))
    timed("Saabas", lambda: path.shap_values(x_e, approximate=True))

    def kernel():
        np.random.seed(seed)  # KernelExplainer ignores seed= and samples with numpy's global state
        return shap.KernelExplainer(g.predict, shap.kmeans(bg, 10)).shap_values(
            x_e, nsamples=CONFIG["kernel_nsamples"], silent=True)
    timed("KernelSHAP (kmeans 10)", kernel)
    timed("Permutation SHAP", lambda: shap.explainers.Permutation(
        g.predict, shap.maskers.Independent(bg, max_samples=len(bg)), seed=seed
    )(x_e, max_evals=CONFIG["permutation_evals"], silent=True).values)
    timed("Owen (correlation tree)", lambda: shap.explainers.Partition(
        g.predict, shap.maskers.Partition(bg, max_samples=len(bg), clustering="correlation")
    )(x_e, max_evals=CONFIG["partition_evals"], silent=True).values)
    timed("Mean-baseline SHAP", lambda: shap.TreeExplainer(
        g, data=shap.maskers.Independent(bg.mean(0, keepdims=True), max_samples=1), feature_perturbation="interventional"
    ).shap_values(x_e, check_additivity=False))

    def lime():
        lx = LimeTabularExplainer(x_tr, mode="regression", discretize_continuous=False, random_state=seed)
        # LIME fits slopes on standardized features; contribution = slope x standardized deviation of x
        rows = []
        for x in x_e:
            w = dict(lx.explain_instance(x, g.predict, num_features=x_e.shape[1], num_samples=CONFIG["lime_samples"]).as_map()[1])
            rows.append([w.get(j, 0.0) * (x[j] - lx.scaler.mean_[j]) / lx.scaler.scale_[j] for j in range(x_e.shape[1])])
        return rows
    timed("LIME", lime)

    def occlusion():
        base, mean = g.predict(x_e), bg.mean(0)
        cols = []
        for j in range(x_e.shape[1]):
            z = x_e.copy()
            z[:, j] = mean[j]
            cols.append(base - g.predict(z))
        return np.column_stack(cols)
    timed("Occlusion (mean)", occlusion)
    return out, secs


def rel_error(phi: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(phi - ref, axis=1).mean() / np.linalg.norm(ref, axis=1).mean())


def run_task(world: World, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    x_tr = world.sample(CONFIG["n_train"], rng, "train")
    y_tr = world.f.predict(x_tr) + rng.normal(0.0, world.sigma, len(x_tr))
    g = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  random_state=seed).fit(x_tr, y_tr)
    x_e = world.sample(CONFIG["n_explain"], rng, "explain")
    x_val = world.sample(500, rng, "explain")
    f_val = world.f.predict(x_val)
    r2 = float(1 - np.mean((g.predict(x_val) - f_val) ** 2) / np.var(f_val))
    t1 = time.time()
    truth_model = truth(world, g, "tree", x_e, rng)
    t2 = time.time()
    truth_data = truth(world, world.f, world.truth, x_e, rng)
    t3 = time.time()
    phis, secs = explain_all(g, x_e, world.sample(CONFIG["n_bg_explainer"], rng, "explain"), x_tr, seed)
    for name, phi in phis.items():
        if phi.shape != truth_model.shape:
            raise ValueError(f"{name}: attribution shape {phi.shape} != {truth_model.shape}")
    return {
        "world": world.name, "seed": seed, "r2_g_vs_f": r2,
        "err_model": {k: rel_error(v, truth_model) for k, v in phis.items()},
        "err_data": {k: rel_error(v, truth_data) for k, v in phis.items()},
        "model_gap": rel_error(truth_model, truth_data),
        "seconds": {"fit": round(t1 - t0, 3), "truth_model": round(t2 - t1, 3), "truth_data": round(t3 - t2, 3),
                    "explainers": secs, "total": round(time.time() - t0, 3)},
    }


# ---------------------------------------------------------------- worlds
def load_reality(family: str) -> tuple:
    data = fetch_california_housing()
    x = (data.data - data.data.mean(0)) / data.data.std(0)
    maker = {"rf": (forest, "tree"), "mlp": (mlp, "exact")}[family]
    f = maker[0](SEED).fit(x, data.target)
    sigma = float(np.std(data.target - f.predict(x)))
    r2 = float(np.var(f.predict(x)) / (np.var(f.predict(x)) + sigma ** 2))
    world = World(f"reality ({family})", lambda n, r, role: snap(x[r.integers(0, len(x), n)]), f, sigma, maker[1])
    return world, list(data.feature_names), r2


def null_world(d: int, signal_r2: float) -> World:
    cov = np.full((d, d), CONFIG["null_rho"]) + (1 - CONFIG["null_rho"]) * np.eye(d)
    f = NullTarget()
    probe = np.random.default_rng(SEED).multivariate_normal(np.zeros(d), cov, 20000)
    sigma = float(np.sqrt(np.var(f.predict(probe)) * (1 - signal_r2) / signal_r2))  # same signal-to-noise as reality
    return World("null (generic, ignores D)", lambda n, r, role: snap(r.multivariate_normal(np.zeros(d), cov, n)), f, sigma, "additive")


def builder_worlds(x_d: np.ndarray, y_d: np.ndarray, names: list, seed: int) -> list:
    n_tr = int(CONFIG["bootstrap_train_share"] * len(x_d))
    pools = {"train": x_d[:n_tr], "explain": x_d[n_tr:]}
    mean, cov = x_d.mean(0), np.cov(x_d, rowvar=False)
    arf_model = arfpy_arf.arf(x=pd.DataFrame(x_d, columns=names), num_trees=30, random_state=seed, verbose=False)
    arf_model.forde()
    gens = {
        "independent": lambda n, r, role: snap(np.column_stack(
            [x_d[r.integers(0, len(x_d), n), j] for j in range(x_d.shape[1])])),
        "gaussian": lambda n, r, role: snap(r.multivariate_normal(mean, cov, n)),
        "bootstrap": lambda n, r, role: snap(pools[role][r.integers(0, len(pools[role]), n)]),
        "arf": lambda n, r, role: snap(arf_model.forge(n=n, random_state=int(r.integers(1 << 31))).to_numpy(dtype=float)),
    }
    worlds = []
    for t_name in TARGETS:
        make, kind = TARGET_MAKERS[t_name]
        f = make(seed).fit(x_d, y_d)
        sigma = float(np.std(y_d - f.predict(x_d)))
        for g_name in GENERATORS:
            worlds.append(World(f"{g_name}|{t_name}", gens[g_name], f, sigma, kind))
    return worlds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reality", choices=["rf", "mlp"], default="rf")
    ap.add_argument("--reps", type=int, default=8, help="samples D a builder could have seen")
    ap.add_argument("--k", type=int, default=5, help="tasks averaged per world (builders, oracle, null)")
    ap.add_argument("--k-reality", type=int, default=30, help="tasks averaged into reality's reference ranking")
    ap.add_argument("--timing", action="store_true", help="one task per world type, one rep: measures the cost")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.timing:
        args.reps, args.k, args.k_reality = 1, 1, 1
    setup_logging(args.tag)
    start = time.time()
    reality, names, signal_r2 = load_reality(args.reality)
    null = null_world(len(names), signal_r2)
    logging.info(f"{reality.name}: noise sd {reality.sigma:.3f}, signal R2 {signal_r2:.3f}; null noise sd {null.sigma:.3f}")
    n_worlds = len(GENERATORS) * len(TARGETS) + 2
    n_total = args.k_reality + args.reps * n_worlds * args.k
    bar = tqdm(total=n_total, desc="tasks", file=sys.stdout)
    tasks, build_seconds = [], []
    for k in range(args.k_reality):
        tasks.append({"rep": None, "role": "reality", **run_task(reality, SEED + k)})
        bar.update()
    for rep in range(args.reps):
        rr = np.random.default_rng(SEED + 10_000 + rep)
        x_d = reality.sample(CONFIG["n_D"], rr, "train")
        y_d = reality.f.predict(x_d) + rr.normal(0.0, reality.sigma, len(x_d))
        t = time.time()
        worlds = builder_worlds(x_d, y_d, names, SEED + rep)
        build_seconds.append(round(time.time() - t, 1))
        logging.info(f"rep {rep}: built {len(worlds)} worlds from D in {build_seconds[-1]}s")
        for world, role in [(reality, "oracle"), (null, "null")] + [(w, "builder") for w in worlds]:
            for k in range(args.k):
                tasks.append({"rep": rep, "role": role, **run_task(world, SEED + 100_000 + rep * 1000 + k)})
                bar.update()
        elapsed = time.time() - start
        logging.info(f"rep {rep + 1}/{args.reps}: {elapsed:.0f}s elapsed, ~{elapsed / bar.n * (n_total - bar.n):.0f}s left")
    bar.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.tag}.json"
    out.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "reality": args.reality, "reps": args.reps,
        "k": args.k, "k_reality": args.k_reality, "timing_mode": args.timing, "config": CONFIG,
        "features": names, "explainers": EXPLAINERS, "generators": GENERATORS, "targets": TARGETS,
        "reality_noise_sd": reality.sigma, "reality_signal_r2": signal_r2, "null_noise_sd": null.sigma,
        "versions": {"shap": shap.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "pandas": pd.__version__},
        "runtime_s": round(time.time() - start, 1), "build_seconds": build_seconds, "tasks": tasks}, indent=1),
        encoding="utf-8")
    logging.info(f"Wrote {out}: {len(tasks)} tasks in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
