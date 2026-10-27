#!/usr/bin/env python3
"""Transfer pilot v3: the v2 question (DECISIONS #22) under the *conditional* (observational) Shapley estimand, where
the data distribution enters the definition of the ground truth (DECISIONS #26; Q20 option (a), 2026-10-28).

Why: Tier 0 of v2 showed that for interventional SHAP every world, even a generic one, reproduces reality's ranking.
Interventional SHAP uses the data only through the background sample; conditional SHAP,
v(S)(x) = E[f(x_S, X_rest) | X_S = x_S], depends on the joint distribution itself, so if benchmark realism matters
anywhere it should matter here.

Design (nested known truth, as v2, with known densities so that conditionals are exact):
- Reality (--reality-family): a Gaussian copula with the real California marginals (default; tied values
  dequantized so that conditionals are exact) or a Gaussian mixture (--reality-components), fitted to the
  standardized features, and v2's true function (a random forest fitted to the real target) plus noise with the
  fit's residual sd.
- Builders see D (2,000 rows from reality) and build worlds whose conditionals are exact: independent columns
  (conditionals = marginals), Gaussian with D's moments, Gaussian mixtures with 3 and 8 components, and a Gaussian
  copula, all fitted to D; targets fitted to D: additive splines and random forest (v2's families). The builder of
  reality's own family is a same-family control that can match reality by construction. Bootstrap and ARF worlds are
  left out: their conditionals are not available in closed form (ARF needs conditional forging, Q9).
- Comparators: the oracle (fresh reality tasks) and a generic null (Gaussian, correlation 0.5, XAI-BENCH-style
  additive target, as v2).
- A task trains a gradient-boosting model g on 1,000 rows, explains --n-explain points with seven explainers and
  scores them against conditional SHAP of g (model level, primary) and of the world's true function (data level).
  Ground truth: all 2^d coalitions, each conditional expectation a Monte Carlo mean over --m-truth draws from the
  world's exact conditional distribution. Reality tasks also record the error against *interventional* SHAP of g
  (exact TreeSHAP on the 2^-12 grid, as v2), for the estimand-sensitivity table (Q13).
- Explainers: interventional TreeSHAP (50-row background), path-dependent TreeSHAP, KernelSHAP (interventional),
  Gaussian-conditional SHAP (Aas et al. 2021: a Gaussian fitted to 200 rows of the world), mixture-conditional SHAP
  (a 3-component mixture fitted to the same rows), LIME, occlusion. The two conditional explainers are written here:
  Python has no maintained package for conditional Shapley values (shapr is R).

Usage (from synthetic_data/; threads capped at 2):
    .venv/Scripts/python.exe scripts/transfer_pilot_v3_conditional.py --timing --tag v3_timing   # ~6 min (estimate)
    .venv/Scripts/python.exe scripts/transfer_pilot_v3_conditional.py --reps 2 --k 3 --k-reality 10 --tag v3_probe
Writes results/transfer_pilot/<tag>.json in v2's format (transfer_report_v2.py reads it), checkpointed after every
task. Every task is seeded, so rerunning an interrupted run with the same arguments resumes it.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_pilot_v2 as tp  # noqa: E402

SEED = 509
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "transfer_pilot"
CONFIG = {"dataset": "california_housing (Gaussian-mixture stand-in)", "grid": tp.CONFIG["grid"], "n_D": 2000,
          "n_train": 1000, "n_bg_explainer": 50, "n_fit_conditional_explainer": 200, "m_explainer": 50,
          "n_bg_interventional_truth": 1000, "kernel_nsamples": 200, "lime_samples": 2000, "null_rho": 0.5}
EXPLAINERS = ["TreeSHAP-int (bg 50)", "TreeSHAP-path", "KernelSHAP (kmeans 10)", "Gaussian-conditional SHAP",
              "Mixture-conditional SHAP (3 comp.)", "LIME", "Occlusion (mean)"]
GENERATORS = ["independent", "gaussian", "mixture3", "mixture8", "copula"]
TARGETS = ["additive", "forest"]


# ---------------------------------------------------------------- distributions with exact conditionals
class Mixture:
    """Gaussian mixture with exact conditional sampling; K = 1 is a Gaussian."""

    def __init__(self, weights: np.ndarray, means: np.ndarray, covs: np.ndarray):
        self.w, self.mu, self.cov = np.asarray(weights, float), np.asarray(means, float), np.asarray(covs, float)
        self.k, self.d = self.mu.shape
        self._cache = {}

    @classmethod
    def fit(cls, x: np.ndarray, k: int, seed: int) -> "Mixture":
        if k == 1:
            return cls(np.ones(1), x.mean(0, keepdims=True), np.cov(x, rowvar=False)[None])
        gm = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=seed, max_iter=500).fit(x)
        if not gm.converged_:
            raise RuntimeError(f"Gaussian mixture with {k} components did not converge")
        return cls(gm.weights_, gm.means_, gm.covariances_)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(self.k, size=n, p=self.w)
        z = rng.standard_normal((n, self.d))
        chol = np.linalg.cholesky(self.cov)
        return self.mu[comp] + np.einsum("nij,nj->ni", chol[comp], z)

    def _parts(self, s: tuple) -> tuple:
        if s not in self._cache:
            s_idx = np.array(s, dtype=int)
            r_idx = np.array([j for j in range(self.d) if j not in s], dtype=int)
            a, l_c, inv_ss, logdet = [], [], [], []
            for k in range(self.k):
                c = self.cov[k]
                c_ss, c_rs, c_rr = c[np.ix_(s_idx, s_idx)], c[np.ix_(r_idx, s_idx)], c[np.ix_(r_idx, r_idx)]
                inv = np.linalg.inv(c_ss)
                a.append(c_rs @ inv)
                l_c.append(np.linalg.cholesky(c_rr - c_rs @ inv @ c_rs.T + 1e-10 * np.eye(len(r_idx))))
                inv_ss.append(inv)
                logdet.append(np.linalg.slogdet(c_ss)[1])
            self._cache[s] = (s_idx, r_idx, np.array(a), np.array(l_c), np.array(inv_ss), np.array(logdet))
        return self._cache[s]

    def cond_sample(self, s: tuple, x_s: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
        """Draws [n, m, d] from p(x | x_S = x_s) for each of the n rows of x_s (s non-empty, not all features)."""
        s_idx, r_idx, a, l_c, inv_ss, logdet = self._parts(s)
        n = len(x_s)
        diff = x_s[:, None, :] - self.mu[None, :, s_idx]                                  # [n, K, |S|]
        maha = np.einsum("nki,kij,nkj->nk", diff, inv_ss, diff)
        logp = np.log(self.w)[None] - 0.5 * (maha + logdet[None])
        post = np.exp(logp - logp.max(1, keepdims=True))
        post /= post.sum(1, keepdims=True)
        comp = (rng.random((n, m, 1)) > np.cumsum(post, 1)[:, None, :]).sum(-1)          # [n, m]
        comp = np.minimum(comp, self.k - 1)
        mean_r = self.mu[None, :, r_idx] + np.einsum("kri,nki->nkr", a, diff)             # [n, K, |R|]
        z = rng.standard_normal((n, m, len(r_idx)))
        out = np.empty((n, m, self.d))
        out[:, :, s_idx] = x_s[:, None, :]
        nn = np.arange(n)[:, None]
        out[:, :, r_idx] = mean_r[nn, comp] + np.einsum("nmrq,nmq->nmr", l_c[comp], z)
        return out


class GaussianCopula:
    """The pool's own marginals (piecewise-linear quantile functions) joined by a Gaussian copula fitted to normal
    scores. Conditionals are exact: map x_S to latent normal scores, draw the rest from the latent Gaussian
    conditional, map back through the marginal quantile functions."""

    def __init__(self, pool: np.ndarray, seed: int = SEED):
        self.d = pool.shape[1]
        n = len(pool)
        # Dequantize tied columns (California's HouseAge has 52 values): jitter within half the gap between
        # neighbouring distinct values. Ties would map one x to an interval of latent scores, and conditioning on a
        # single point would not be the exact conditional; without ties the marginal maps are one-to-one.
        rng = np.random.default_rng(seed)
        pool = np.array(pool, dtype=float)
        for j in range(self.d):
            u = np.unique(pool[:, j])
            if len(u) < n:
                gap = np.min(np.diff(u))
                pool[:, j] += rng.uniform(-gap / 2, gap / 2, n)
        self.sorted = np.sort(pool, axis=0)
        self.p = (np.arange(1, n + 1) - 0.5) / n
        z = norm.ppf(self.p)[np.argsort(np.argsort(pool, axis=0), axis=0)]
        self.latent = Mixture(np.ones(1), np.zeros((1, self.d)), np.corrcoef(z, rowvar=False)[None])

    def _to_x(self, z: np.ndarray, cols) -> np.ndarray:
        u = norm.cdf(z)
        return np.column_stack([np.interp(u[..., i].ravel(), self.p, self.sorted[:, j]) for i, j in enumerate(cols)]
                               ).reshape(z.shape)

    def _to_z(self, x: np.ndarray, cols) -> np.ndarray:
        u = np.column_stack([np.interp(x[:, i], self.sorted[:, j], self.p) for i, j in enumerate(cols)])
        return norm.ppf(np.clip(u, self.p[0], self.p[-1]))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._to_x(self.latent.sample(n, rng), range(self.d))

    def cond_sample(self, s: tuple, x_s: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
        z = self.latent.cond_sample(s, self._to_z(x_s, s), m, rng)        # [n, m, d], latent
        r = [j for j in range(self.d) if j not in s]
        out = np.empty_like(z)
        out[:, :, list(s)] = x_s[:, None, :]
        out[:, :, r] = self._to_x(z[:, :, r], r)
        return out


class Independent:
    """Columns resampled independently from a pool: every conditional is the product of marginals."""

    def __init__(self, pool: np.ndarray):
        self.pool, self.d = pool, pool.shape[1]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.column_stack([self.pool[rng.integers(0, len(self.pool), n), j] for j in range(self.d)])

    def cond_sample(self, s: tuple, x_s: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
        n = len(x_s)
        out = self.sample(n * m, rng).reshape(n, m, self.d)
        out[:, :, list(s)] = x_s[:, None, :]
        return out


# ---------------------------------------------------------------- conditional Shapley values
def conditional_shap(predict: Callable, x_e: np.ndarray, dist, m: int, rng: np.random.Generator) -> np.ndarray:
    """phi_i(x) = sum over S not containing i of |S|!(d-|S|-1)!/d! [v(S + i) - v(S)], with v(S) = E[f | X_S = x_S]
    estimated by m draws from dist's conditional distribution; v(empty) and v(all) by the mean and f(x)."""
    n, d = x_e.shape
    v = np.empty((1 << d, n))
    base = predict(tp.snap(dist.sample(n * m, rng))).reshape(n, m).mean(1)
    v[0] = base
    v[(1 << d) - 1] = predict(x_e)
    for mask in range(1, (1 << d) - 1):
        s = tuple(j for j in range(d) if mask >> j & 1)
        z = tp.snap(dist.cond_sample(s, x_e[:, list(s)], m, rng))
        v[mask] = predict(z.reshape(-1, d)).reshape(n, m).mean(1)
    w = [math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d) for k in range(d)]
    phi = np.zeros((n, d))
    for mask in range(1 << d):
        size = bin(mask).count("1")
        for i in range(d):
            if not mask >> i & 1:
                phi[:, i] += w[size] * (v[mask | 1 << i] - v[mask])
    return phi


# ---------------------------------------------------------------- worlds and tasks
@dataclass
class World:
    name: str
    dist: object
    f: object
    sigma: float


def explain_all(g, x_e: np.ndarray, world: World, x_tr: np.ndarray, seed: int, rng: np.random.Generator) -> tuple:
    out, secs = {}, {}
    bg = tp.snap(world.dist.sample(CONFIG["n_bg_explainer"], rng))
    fit_rows = tp.snap(world.dist.sample(CONFIG["n_fit_conditional_explainer"], rng))

    def timed(name, fn):
        t = time.time()
        out[name] = np.asarray(fn(), dtype=float)
        secs[name] = round(time.time() - t, 3)

    timed("TreeSHAP-int (bg 50)", lambda: shap.TreeExplainer(
        g, data=shap.maskers.Independent(bg, max_samples=len(bg)), feature_perturbation="interventional"
    ).shap_values(x_e, check_additivity=False))
    timed("TreeSHAP-path", lambda: shap.TreeExplainer(g, feature_perturbation="tree_path_dependent").shap_values(x_e))

    def kernel():
        np.random.seed(seed)  # KernelExplainer ignores seed= and samples with numpy's global state
        return shap.KernelExplainer(g.predict, shap.kmeans(bg, 10)).shap_values(
            x_e, nsamples=CONFIG["kernel_nsamples"], silent=True)
    timed("KernelSHAP (kmeans 10)", kernel)
    timed("Gaussian-conditional SHAP", lambda: conditional_shap(
        g.predict, x_e, Mixture.fit(fit_rows, 1, seed), CONFIG["m_explainer"], rng))
    timed("Mixture-conditional SHAP (3 comp.)", lambda: conditional_shap(
        g.predict, x_e, Mixture.fit(fit_rows, 3, seed), CONFIG["m_explainer"], rng))

    def lime():
        lx = LimeTabularExplainer(x_tr, mode="regression", discretize_continuous=False, random_state=seed)
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


def run_task(world: World, seed: int, args, interventional: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    x_tr = tp.snap(world.dist.sample(CONFIG["n_train"], rng))
    y_tr = world.f.predict(x_tr) + rng.normal(0.0, world.sigma, len(x_tr))
    g = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  random_state=seed).fit(x_tr, y_tr)
    x_e = tp.snap(world.dist.sample(args.n_explain, rng))
    t1 = time.time()
    truth_model = conditional_shap(g.predict, x_e, world.dist, args.m_truth, rng)
    t2 = time.time()
    truth_data = conditional_shap(world.f.predict, x_e, world.dist, args.m_truth, rng)
    t3 = time.time()
    phis, secs = explain_all(g, x_e, world, x_tr, seed, rng)
    rec = {"world": world.name, "seed": seed,
           "err_model": {k: tp.rel_error(v, truth_model) for k, v in phis.items()},
           "err_data": {k: tp.rel_error(v, truth_data) for k, v in phis.items()},
           "model_gap": tp.rel_error(truth_model, truth_data)}
    if interventional:
        bg = tp.snap(world.dist.sample(CONFIG["n_bg_interventional_truth"], rng))
        truth_int = tp.tree_truth(g, x_e, bg)
        rec["err_model_interventional"] = {k: tp.rel_error(v, truth_int) for k, v in phis.items()}
        rec["estimand_gap"] = tp.rel_error(truth_int, truth_model)
        # Monte Carlo noise floor of the conditional truth: a second, independent estimate of the same quantity
        truth_b = conditional_shap(g.predict, x_e, world.dist, args.m_truth, np.random.default_rng(seed + 7_777_777))
        rec["truth_mc_noise"] = tp.rel_error(truth_b, truth_model)
    rec["seconds"] = {"fit": round(t1 - t0, 3), "truth_model": round(t2 - t1, 3), "truth_data": round(t3 - t2, 3),
                      "explainers": secs, "total": round(time.time() - t0, 3)}
    return rec


def null_world(d: int, signal_r2: float) -> World:
    cov = np.full((d, d), CONFIG["null_rho"]) + (1 - CONFIG["null_rho"]) * np.eye(d)
    f = tp.NullTarget()
    probe = np.random.default_rng(SEED).multivariate_normal(np.zeros(d), cov, 20000)
    sigma = float(np.sqrt(np.var(f.predict(probe)) * (1 - signal_r2) / signal_r2))
    return World("null (generic, ignores D)", Mixture(np.ones(1), np.zeros((1, d)), cov[None]), f, sigma)


def builder_worlds(x_d: np.ndarray, y_d: np.ndarray, seed: int) -> list:
    gens = {"independent": Independent(x_d), "gaussian": Mixture.fit(x_d, 1, seed),
            "mixture3": Mixture.fit(x_d, 3, seed), "mixture8": Mixture.fit(x_d, 8, seed),
            "copula": GaussianCopula(x_d)}
    worlds = []
    for t_name in TARGETS:
        make, _ = tp.TARGET_MAKERS[t_name]
        f = make(seed).fit(x_d, y_d)
        sigma = float(np.std(y_d - f.predict(x_d)))
        for g_name in GENERATORS:
            worlds.append(World(f"{g_name}|{t_name}", gens[g_name], f, sigma))
    return worlds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--k-reality", type=int, default=10)
    ap.add_argument("--n-explain", type=int, default=20)
    ap.add_argument("--m-truth", type=int, default=500, help="Monte Carlo draws per coalition for the ground truth")
    ap.add_argument("--reality-family", choices=["copula", "mixture"], default="copula",
                    help="copula: the real marginals with a Gaussian copula; mixture: a Gaussian mixture")
    ap.add_argument("--reality-components", type=int, default=8, help="for --reality-family mixture")
    ap.add_argument("--timing", action="store_true", help="one task per world type, one rep")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.timing:
        args.reps, args.k, args.k_reality = 1, 1, 1
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "transfer_pilot_v3.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.info(f"run {args.tag}")
    start = time.time()
    v2_reality, names, signal_r2 = tp.load_reality("rf")
    raw = tp.fetch_california_housing().data
    x_real = (raw - raw.mean(0)) / raw.std(0)   # the same standardization as v2's reality
    if args.reality_family == "copula":
        label, dist = "Gaussian copula with the real marginals", GaussianCopula(x_real)
    else:
        label, dist = f"{args.reality_components}-component mixture", Mixture.fit(x_real, args.reality_components, SEED)
    reality = World(f"reality (rf, {label})", dist, v2_reality.f, v2_reality.sigma)
    null = null_world(len(names), signal_r2)
    logging.info(f"{reality.name} fitted to {len(x_real)} rows; noise sd {reality.sigma:.3f}")
    per_rep = (len(GENERATORS) * len(TARGETS) + 2) * args.k
    n_total = args.k_reality + args.reps * per_rep
    tasks, build_seconds, prior_s = [], [], 0.0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.tag}.json"
    if out.exists():   # resume an interrupted run: tasks are seeded, so the finished ones need not rerun
        old = json.loads(out.read_text(encoding="utf-8"))
        if old["complete"]:
            raise SystemExit(f"{out} is already complete; use a new --tag")
        if old["args"] != vars(args):
            raise SystemExit(f"{out} was started with other arguments {old['args']}; use a new --tag")
        tasks, build_seconds, prior_s = old["tasks"], old["build_seconds"], old["runtime_s"]
        logging.info(f"resuming {out}: {len(tasks)} of {n_total} tasks already done")
    done = len(tasks)
    bar = tqdm(total=n_total, initial=done, desc="tasks", file=sys.stdout)

    def write(complete: bool) -> None:
        out.write_text(json.dumps({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": complete, "estimand": "conditional",
            "seed": SEED, "reality": reality.name, "reps": args.reps,
            "k": args.k, "k_reality": args.k_reality, "timing_mode": args.timing, "args": vars(args), "config": CONFIG,
            "features": names, "explainers": EXPLAINERS, "generators": GENERATORS, "targets": TARGETS,
            "reality_noise_sd": reality.sigma, "reality_signal_r2": signal_r2, "null_noise_sd": null.sigma,
            "versions": {"shap": shap.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
            "runtime_s": round(prior_s + time.time() - start, 1), "build_seconds": build_seconds, "tasks": tasks},
            indent=1), encoding="utf-8")

    def add(task: dict) -> None:
        tasks.append(task)
        bar.update()
        write(complete=len(tasks) == n_total)

    for k in range(done, args.k_reality):
        add({"rep": None, "role": "reality", **run_task(reality, SEED + k, args, interventional=True)})
    for rep in range(args.reps):
        first = args.k_reality + rep * per_rep   # index of this replicate's first task
        if done >= first + per_rep:
            continue
        rr = np.random.default_rng(SEED + 10_000 + rep)
        x_d = tp.snap(reality.dist.sample(CONFIG["n_D"], rr))
        y_d = reality.f.predict(x_d) + rr.normal(0.0, reality.sigma, len(x_d))
        t = time.time()
        worlds = builder_worlds(x_d, y_d, SEED + rep)
        if len(build_seconds) == rep:
            build_seconds.append(round(time.time() - t, 1))
        jobs = [(world, role, k) for world, role in [(reality, "oracle"), (null, "null")] + [(w, "builder") for w in worlds]
                for k in range(args.k)]
        for i, (world, role, k) in enumerate(jobs):
            if first + i >= done:
                add({"rep": rep, "role": role, **run_task(world, SEED + 100_000 + rep * 1000 + k, args)})
        elapsed = time.time() - start
        left = elapsed / (bar.n - done) * (n_total - bar.n)
        logging.info(f"rep {rep + 1}/{args.reps}: {elapsed:.0f}s elapsed this session, ~{left:.0f}s left")
    bar.close()
    logging.info(f"wrote {out}: {len(tasks)} tasks, {time.time() - start:.0f}s this session")


if __name__ == "__main__":
    main()
