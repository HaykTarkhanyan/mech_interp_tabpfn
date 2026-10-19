#!/usr/bin/env python3
"""Transfer pilot: does an explainer ranking measured on a data-fitted synthetic world reproduce the ranking in
the world the data came from?

Nested known-truth design after Advani et al. (2019), applied to explainers (DECISIONS.md #22).
- Reality keeps the real feature rows of a dataset and a known true function fitted to its real target, plus
  calibrated noise, so every explainer's error is computable there. Its explainer ranking is averaged over
  several task replicates: that is the ranking a benchmark should reproduce.
- A benchmark builder sees one sample D from reality (as an analyst sees one real dataset), builds its own
  world from D (a feature generator plus a target model fitted on D) and ranks the same explainers inside it.
- The oracle builder is reality itself with a fresh sample: its agreement with reality is the ceiling set by
  task noise alone.

Explanation estimand: interventional SHAP. Each explainer is applied to a gradient-boosting model g trained in
the world, and scored against the interventional SHAP values of the world's true function (data level) and of
g itself (model level). Ground truth is computed by exact enumeration through model.predict and checked for
additivity; shap's interventional TreeSHAP is only recorded as a diagnostic (treeshap_int_gap), because it lost
additivity on California rows. Ranking agreement and regret are computed from the saved JSON by
transfer_report.py.

Usage (from synthetic_data/; threads capped at 2 so the laptop stays usable):
    .venv/Scripts/python.exe scripts/transfer_pilot.py --reps 2 --tag smoke    # ~2 min (guess)
    .venv/Scripts/python.exe scripts/transfer_pilot.py --reps 10 --tag pilot   # ~10 min (guess)
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
import shap  # noqa: E402
from sklearn.datasets import fetch_california_housing  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import SplineTransformer  # noqa: E402
from tqdm import tqdm  # noqa: E402

SEED = 509
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "transfer_pilot"
CONFIG = {
    "dataset": "california_housing",
    "n_D": 1000,                  # rows the builder sees
    "n_train": 2000,              # training rows for the explained model g, in any world
    "n_explain": 50,              # points explained per task
    "n_background_truth": 100,    # background rows for the true interventional SHAP values
    "n_background_explainer": 50, # an explainer's own background is a separate, smaller draw
    "reality_task_reps": 5,       # tasks averaged into reality's ranking
    "kernel_nsamples": 200,
}
EXPLAINERS = ["Interventional SHAP (bg 50)", "Interventional SHAP (bg 10)", "TreeSHAP-path", "Saabas",
              "KernelSHAP (kmeans 10)", "Mean-baseline SHAP"]
ADDITIVITY_TOL = 1e-6


@dataclass
class World:
    name: str
    sample_x: Callable[[int, np.random.Generator], np.ndarray]
    f: object          # fitted model with .predict: the world's true function
    sigma: float       # noise sd added to f in this world


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "transfer_pilot.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    logging.getLogger("shap").setLevel(logging.WARNING)  # KernelExplainer logs internals at INFO, per point


def interventional_shap(model, x_explain: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Interventional SHAP by full enumeration through model.predict (2^d coalitions x background rows).

    Exact by construction, so it serves as ground truth. shap's interventional TreeSHAP is not used for this:
    on California rows (many repeated values) it lost additivity by up to 0.12 in shap 0.54.1, while it was
    exact on Gaussian data. Raises if additivity fails here.
    """
    # The masker is sized to the background; given raw data, shap silently subsamples to 100 rows.
    masker = shap.maskers.Independent(background, max_samples=len(background))
    exp = shap.explainers.Exact(model.predict, masker)(x_explain)
    gap = float(np.abs(exp.values.sum(1) + exp.base_values - model.predict(x_explain)).max())
    if gap > ADDITIVITY_TOL:
        raise RuntimeError(f"exact interventional SHAP lost additivity: max gap {gap:.2e}")
    return exp.values


def treeshap_interventional_gap(g, x_explain: np.ndarray, bg: np.ndarray) -> float:
    """Diagnostic only: largest additivity violation of shap's interventional TreeSHAP for g."""
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    e = shap.TreeExplainer(g, data=masker, feature_perturbation="interventional")
    phi = e.shap_values(x_explain, check_additivity=False)
    return float(np.abs(phi.sum(1) + e.expected_value - g.predict(x_explain)).max())


def run_explainers(g, x_explain: np.ndarray, bg: np.ndarray, rng: np.random.Generator) -> dict:
    path = shap.TreeExplainer(g, feature_perturbation="tree_path_dependent")
    kernel = shap.KernelExplainer(g.predict, shap.kmeans(bg, 10), seed=int(rng.integers(1 << 31)))
    return {
        "Interventional SHAP (bg 50)": interventional_shap(g, x_explain, bg),
        "Interventional SHAP (bg 10)": interventional_shap(g, x_explain, bg[:10]),
        "TreeSHAP-path": path.shap_values(x_explain),
        "Saabas": path.shap_values(x_explain, approximate=True),
        "KernelSHAP (kmeans 10)": np.asarray(kernel.shap_values(x_explain, nsamples=CONFIG["kernel_nsamples"], silent=True)),
        "Mean-baseline SHAP": interventional_shap(g, x_explain, bg.mean(axis=0, keepdims=True)),
    }


def rel_error(phi: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(phi - truth, axis=1).mean() / np.linalg.norm(truth, axis=1).mean())


def run_task(world: World, seed: int) -> dict:
    """Train g in `world`, explain it six ways, score against data-level and model-level interventional SHAP."""
    rng = np.random.default_rng(seed)
    x_tr = world.sample_x(CONFIG["n_train"], rng)
    y_tr = world.f.predict(x_tr) + rng.normal(0.0, world.sigma, len(x_tr))
    g = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  random_state=seed).fit(x_tr, y_tr)
    x_e = world.sample_x(CONFIG["n_explain"], rng)
    bg_truth = world.sample_x(CONFIG["n_background_truth"], rng)
    truth_data = interventional_shap(world.f, x_e, bg_truth)
    truth_model = interventional_shap(g, x_e, bg_truth)
    bg = world.sample_x(CONFIG["n_background_explainer"], rng)
    phis = run_explainers(g, x_e, bg, rng)
    for name, phi in phis.items():
        if phi.shape != truth_data.shape:
            raise ValueError(f"{name}: attribution shape {phi.shape} != truth shape {truth_data.shape}")
    return {
        "world": world.name, "seed": seed,
        "err_data": {k: rel_error(v, truth_data) for k, v in phis.items()},
        "err_model": {k: rel_error(v, truth_model) for k, v in phis.items()},
        "model_gap": rel_error(truth_model, truth_data),  # how far g's true SHAP is from f's: model error
        "treeshap_int_gap": treeshap_interventional_gap(g, x_e, bg),
    }


def load_reality():
    data = fetch_california_housing()
    x = (data.data - data.data.mean(0)) / data.data.std(0)
    y = data.target
    f_r = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=SEED,
                                n_jobs=1).fit(x, y)
    sigma = float(np.std(y - f_r.predict(x)))
    world = World("reality", lambda n, r: x[r.integers(0, len(x), n)], f_r, sigma)
    return world, list(data.feature_names)


def make_builder_worlds(x_d: np.ndarray, y_d: np.ndarray) -> list:
    """All generator x target combinations a builder could make from the one sample D."""
    mean, cov = x_d.mean(0), np.cov(x_d, rowvar=False)
    generators = {
        "independent": lambda n, r: np.column_stack([x_d[r.integers(0, len(x_d), n), j] for j in range(x_d.shape[1])]),
        "gaussian": lambda n, r: r.multivariate_normal(mean, cov, n),
        "bootstrap": lambda n, r: x_d[r.integers(0, len(x_d), n)],
    }
    targets = {
        "linear": LinearRegression(),
        "additive": make_pipeline(SplineTransformer(n_knots=6, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "forest": RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=SEED, n_jobs=1),
    }
    worlds = []
    for t_name, model in targets.items():
        model.fit(x_d, y_d)
        sigma = float(np.std(y_d - model.predict(x_d)))
        for g_name, gen in generators.items():
            worlds.append(World(f"{g_name}|{t_name}", gen, model, sigma))
    return worlds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=10, help="number of samples D a builder could have seen")
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    setup_logging()
    start = time.time()
    reality, features = load_reality()
    logging.info(f"reality: {CONFIG['dataset']}, {len(features)} features, noise sd {reality.sigma:.3f}")

    tasks = []
    n_total = CONFIG["reality_task_reps"] + args.reps * 10
    bar = tqdm(total=n_total, desc="tasks", file=sys.stdout)
    for k in range(CONFIG["reality_task_reps"]):
        tasks.append({"rep": None, "role": "reality", **run_task(reality, SEED + k)})
        bar.update()
    for rep in range(args.reps):
        rr = np.random.default_rng(SEED + 1000 + rep)
        x_d = reality.sample_x(CONFIG["n_D"], rr)
        y_d = reality.f.predict(x_d) + rr.normal(0.0, reality.sigma, len(x_d))
        tasks.append({"rep": rep, "role": "oracle", **run_task(reality, SEED + 2000 + rep)})
        bar.update()
        for world in make_builder_worlds(x_d, y_d):
            tasks.append({"rep": rep, "role": "builder", **run_task(world, SEED + 3000 + rep)})
            bar.update()
        elapsed = time.time() - start
        logging.info(f"rep {rep + 1}/{args.reps} done, {elapsed:.0f}s elapsed, "
                     f"~{elapsed / bar.n * (n_total - bar.n):.0f}s left")
    bar.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.tag}.json"
    out.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "reps": args.reps, "config": CONFIG,
        "features": features, "explainers": EXPLAINERS, "reality_noise_sd": reality.sigma,
        "versions": {"shap": shap.__version__, "numpy": np.__version__},
        "runtime_s": round(time.time() - start, 1), "tasks": tasks}, indent=1), encoding="utf-8")
    logging.info(f"Wrote {out}: {len(tasks)} tasks in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
