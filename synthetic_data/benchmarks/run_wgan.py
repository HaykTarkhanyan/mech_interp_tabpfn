"""Athey, Imbens, Metzger & Munro (2024), "Using WGANs for the design of Monte Carlo simulations": simulate from the
trained WGANs shipped in github.com/gsbDBI/ds-wgan (trained_models/) and rerun a small estimator comparison.

The repository ships the weights but not the code that trained them (that is in a Colab notebook). The weight
shapes fix the layout: G_0 maps 12 noise + t to 4 continuous covariates (age, education, re74, re75) and 4 binary
ones as one-hot pairs (black, hispanic, married, nodegree); G_1 maps 1 noise + t + those 8 covariates to re78.
Two things the weights do not fix, both chosen here by how well generated data match the real data:
- the scaling statistics (mean, sd) the DataWrapper took from its training frame: the raw data, or a resample
  balanced on t (the package docs suggest balancing when t is unbalanced);
- the order of the 9 context variables of G_1 (two candidates).
Selection rule, fixed before looking at estimator results: the configuration with the smallest mean absolute
standardized difference between real and generated means of every variable within t = 0 and t = 1.

Then, for each LaLonde sample (PSID, CPS controls with the experimental treated units, as in the paper):
draw t from its empirical share, X | t from G_0, y(0) and y(1) from G_1; the true ATE is the mean of
y(1) - y(0) over 100,000 draws. --reps simulated data sets of the real sample size are each given to
seven estimators (the RealCause causallib wrappers used in run_realcause.py), scored by bias and RMSE against
the true ATE and the true ATT (the LaLonde literature's usual target).

Usage (from synthetic_data/benchmarks/):
    .venv/Scripts/python.exe run_wgan.py                 # ~4 min (estimate)
Writes results/wgan.json.
"""

import common  # noqa: F401  (thread caps first)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(common.SRC / "ds-wgan"))
warnings.filterwarnings("ignore")

import wgan  # noqa: E402

from run_realcause import estimate, estimators  # noqa: E402  (same estimators as the RealCause run)

WG = common.SRC / "ds-wgan"
CONT0, CAT0 = ["age", "education", "re74", "re75"], ["black", "hispanic", "married", "nodegree"]
CONTEXT1 = {"t + continuous + binary": ["t"] + CONT0 + CAT0,
            "t + covariates in data order": ["t", "age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]}
VARS = CONT0 + CAT0 + ["re78"]


def balanced(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    p = df["t"].mean()
    w = (1 - p) * df["t"] + p * (1 - df["t"])
    return df.sample(2 * len(df), weights=w, replace=True, random_state=seed)


def load_models(tag: str, frame0: pd.DataFrame, frame1: pd.DataFrame, context1: list) -> tuple:
    dw0 = wgan.DataWrapper(frame0, CONT0, CAT0, ["t"], continuous_lower_bounds={"re74": 0, "re75": 0})
    dw1 = wgan.DataWrapper(frame1, ["re78"], [], context1, continuous_lower_bounds={"re78": 0})
    g0, g1 = wgan.Generator(wgan.Specifications(dw0)), wgan.Generator(wgan.Specifications(dw1))
    g0.load_state_dict(torch.load(WG / "trained_models" / f"G_0_{tag}.pth", map_location="cpu"))
    g1.load_state_dict(torch.load(WG / "trained_models" / f"G_1_{tag}.pth", map_location="cpu"))
    g0.eval()
    g1.eval()
    return dw0, g0, dw1, g1


def simulate(models: tuple, n: int, p_treat: float, rng: np.random.Generator, both_arms: bool = False) -> pd.DataFrame:
    dw0, g0, dw1, g1 = models
    torch.manual_seed(int(rng.integers(1 << 31)))
    np.random.seed(int(rng.integers(1 << 31)))   # apply_generator samples a template row with pandas' global RNG
    with torch.no_grad():
        df = pd.DataFrame({"t": rng.binomial(1, p_treat, n).astype(float)})
        df = dw0.apply_generator(g0, df)
        if not both_arms:
            return dw1.apply_generator(g1, df)
        y0 = dw1.apply_generator(g1, df.assign(t=0.0))["re78"].to_numpy()
        y1 = dw1.apply_generator(g1, df.assign(t=1.0))["re78"].to_numpy()
    out = df.copy()
    out["y0"], out["y1"] = y0, y1
    out["re78"] = np.where(out["t"] == 1, y1, y0)
    return out


def fit_score(real: pd.DataFrame, fake: pd.DataFrame) -> float:
    diffs = []
    for t in (0, 1):
        r, f = real[real["t"] == t], fake[fake["t"] == t]
        for v in VARS:
            diffs.append(abs(f[v].mean() - r[v].mean()) / (real[v].std() + 1e-9))
    return float(np.mean(diffs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n-truth", type=int, default=100000)
    ap.add_argument("--tag", default="wgan")
    args = ap.parse_args()
    log = common.setup(args.tag)
    started = time.time()
    rng = np.random.default_rng(common.SEED)
    out = {"source": "github.com/gsbDBI/ds-wgan", "commit": "ee7fa679dfda852cdb04cef211f2370599e657b6",
           "args": vars(args), "datasets": {}}
    for tag in ["psid", "cps"]:
        real = pd.read_feather(WG / "data" / "original_data" / f"{tag}.feather")
        n, p = len(real), float(real["t"].mean())
        configs = {}
        for scale0 in ["raw", "balanced"]:
            frame0 = real if scale0 == "raw" else balanced(real, common.SEED)
            for c_name, context1 in CONTEXT1.items():
                for scale1 in ["raw", "balanced"]:
                    frame1 = real if scale1 == "raw" else balanced(real, common.SEED)
                    models = load_models(tag, frame0, frame1, context1)
                    fake = simulate(models, 50000, p, np.random.default_rng(common.SEED))
                    configs[(scale0, c_name, scale1)] = (fit_score(real, fake), models, fake)
                    log.info(f"{tag}: X|t scaling {scale0}, Y context {c_name}, Y scaling {scale1}: "
                             f"mean |std. diff| {configs[(scale0, c_name, scale1)][0]:.4f}")
        best = min(configs, key=lambda k: configs[k][0])
        _, models, fake = configs[best]
        means = []
        for t in (0, 1):
            for v in VARS:
                means.append({"t": t, "variable": v, "real": float(real.loc[real["t"] == t, v].mean()),
                              "generated": float(fake.loc[fake["t"] == t, v].mean())})
        truth = simulate(models, args.n_truth, p, rng, both_arms=True)
        true_ate = float((truth["y1"] - truth["y0"]).mean())
        true_att = float((truth["y1"] - truth["y0"])[truth["t"] == 1].mean())
        log.info(f"{tag}: chose {best}; true ATE {true_ate:.1f}, ATT {true_att:.1f}; real naive difference "
                 f"{real.loc[real.t == 1, 're78'].mean() - real.loc[real.t == 0, 're78'].mean():.1f}")
        rows = []
        for r in range(args.reps):
            d = simulate(models, n, p, rng)
            w, t, y = d[CONT0 + CAT0], d["t"].astype(int), d["re78"]
            for name, make in estimators().items():
                ate, att, _ = estimate(make, w, t, y)
                rows.append({"rep": r, "estimator": name, "ate_est": ate, "att_est": att})
            if (r + 1) % 5 == 0:
                el = time.time() - started
                log.info(f"{tag}: rep {r + 1}/{args.reps}, {el:.0f}s elapsed")
        out["datasets"][tag] = {
            "n": n, "p_treat": p, "chosen_config": list(best),
            "config_scores": [{"x_scaling": k[0], "y_context": k[1], "y_scaling": k[2], "mean_abs_std_diff": v[0]}
                              for k, v in configs.items()],
            "means_by_t": means, "true_ate": true_ate, "true_att": true_att, "rows": rows}
    out["versions"] = common.pkg_versions("torch", "causallib", "scikit-learn", "pandas")
    common.save(args.tag, out, started)


if __name__ == "__main__":
    main()
