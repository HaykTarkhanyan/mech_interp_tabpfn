"""Credence (Parikh et al. 2022): fit the Credence-v2 generator (github.com/amzn/credence-to-causal-estimation,
credence-v2/) to LaLonde data with a user-chosen treatment effect, then check what it produced.

For each data set (PSID controls with the NSW treated, 6.9% treated; the NSW experiment, 41.6% treated) and each
target effect f (0, as in the paper's real-data runs, and a constant +1,000 USD): fit the three conVAEs
(T; X | T; Y(0), Y(1) | X, T) with upstream defaults except --max-epochs, sample, and record
- the generated treated share (sample() thresholds the T decoder at 0.5 instead of drawing T, so a rare
  treatment can vanish; then no estimator can be scored, and the run says so);
- how close the generated individual effects Y(1) - Y(0) are to f (the constraint the method promises);
- generated vs real means of every variable within t = 0 and t = 1;
- seven estimators (the RealCause causallib wrappers, run_realcause.py) on --reps generated data sets of the real
  size, scored against each data set's true ATE.

Two settings per data set:
- upstream's own (raw dollars, f = 0, --raw-epochs): recorded to show whether it trains at all; in the smoke test
  on 2026-10-27 every generated effect was NaN after 5 epochs (the effect constraint is weighted 1e20 and earnings
  enter in dollars, so fp32 gradients overflow);
- the main arms: numeric columns z-scored before fitting, the target effect given in z units (f / sd(re78)), and
  samples mapped back to dollars.

Upstream facts this run works around or records (see patches/credence-to-causal-estimation.patch):
- Trainers are hard-coded to gpus=1, precision=16; patched to CPU, fp32, no progress bar or checkpoints.
- Credence.fit(batch_size=...) is never passed on: every conVAE trains with its own default batch size 10.
- The variable-bound term adds 1e20 * max(indicator) to the loss; an indicator has no gradient, so bounds do not
  shape training. Earnings can therefore be generated below 0; the share is recorded.
- Data are not standardized (earnings enter the MSE in dollars).

Usage (from synthetic_data/benchmarks/):
    .venv/Scripts/python.exe run_credence.py             # ~8 min (estimate)
Writes results/credence.json.
"""

import common  # noqa: F401  (thread caps first)

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytorch_lightning as pl  # noqa: E402

CRED = common.SRC / "credence-to-causal-estimation" / "credence-v2" / "src"
sys.path.insert(0, str(CRED / "credence"))   # credence/__init__.py does "import autoencoder"
sys.path.insert(0, str(CRED))
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from credence import Credence  # noqa: E402

from run_realcause import estimate, estimators  # noqa: E402

CAT = ["black", "hispanic", "married", "nodegree", "t"]
NUM = ["age", "education", "re74", "re75", "re78"]
EFFECTS = {"f = 0": 0.0, "f = +1000": 1000.0}


def run_arm(ds: str, real: pd.DataFrame, effect: str, f_val: float, args, log, started: float,
            standardize: bool = True, max_epochs: int | None = None, reps: int | None = None) -> dict:
    """Fit Credence to one data set with one target effect, sample, and score the estimators.

    standardize=True fits on z-scored numeric columns (target effect divided by sd(re78)) and maps samples back to
    dollars; standardize=False is upstream's raw-dollar setting.
    """
    arm = f"{ds}, {effect}" + ("" if standardize else ", raw dollars (upstream)")
    max_epochs = args.max_epochs if max_epochs is None else max_epochs
    reps = args.reps if reps is None else reps
    mu = real[NUM].mean() if standardize else pd.Series(0.0, index=NUM)
    sd = real[NUM].std() if standardize else pd.Series(1.0, index=NUM)
    fit_data = real.copy()
    fit_data[NUM] = (real[NUM] - mu) / sd
    pl.seed_everything(common.SEED, workers=True)
    t0 = time.time()
    cr = Credence(data=fit_data, post_treatment_var=["re78"], treatment_var=["t"], categorical_var=CAT,
                  numerical_var=NUM)
    cr.fit(treatment_effect_fn=lambda x, v=f_val / sd["re78"]: v, max_epochs=max_epochs)
    t_fit = time.time() - t0

    def draw(n: int) -> pd.DataFrame:
        d = cr.sample(num_samples=n).rename(columns={0: "y0", 1: "y1"})
        for c in [c for c in NUM if c != "re78"]:
            d[c] = d[c] * sd[c] + mu[c]
        d["y0"], d["y1"] = d["y0"] * sd["re78"] + mu["re78"], d["y1"] * sd["re78"] + mu["re78"]
        return d

    big = draw(50000)
    ite = (big["y1"] - big["y0"]).to_numpy()
    if not np.isfinite(ite).all():
        log.error(f"{arm}: {np.mean(~np.isfinite(ite)):.0%} of generated effects are not finite (training diverged)")
    big["re78"] = np.where(big["t"] == 1, big["y1"], big["y0"])
    means = [{"t": t, "variable": v, "real": float(real.loc[real.t == t, v].mean()),
              "generated": float(big.loc[big.t == t, v].mean()) if (big.t == t).any() else None}
             for t in (0, 1) for v in [c for c in CAT + NUM if c != "t"]]
    log.info(f"{arm}: fit {t_fit:.0f}s; generated ITE mean {ite.mean():.2f}, sd {ite.std():.2f} "
             f"(target {f_val}); treated share {big.t.mean():.3f} vs real {real.t.mean():.3f}")
    collapsed = big["t"].nunique() < 2
    diverged = not np.isfinite(ite).all()
    if collapsed:
        log.error(f"{arm}: every generated unit has t = {big['t'].iloc[0]:.0f} (real treated share "
                  f"{real['t'].mean():.3f}); no estimator can be scored on this generator")
    rows = []
    rng = np.random.default_rng(common.SEED)
    for r in range(0 if (collapsed or diverged) else reps):
        pl.seed_everything(int(rng.integers(1 << 31)), workers=True)
        d = draw(len(real))
        d["re78"] = np.where(d["t"] == 1, d["y1"], d["y0"])
        true_ate = float((d["y1"] - d["y0"]).mean())
        w, t, y = d[[c for c in CAT + NUM if c not in ("t", "re78")]], d["t"].astype(int), d["re78"]
        if t.nunique() < 2:
            raise RuntimeError(f"{arm}: generated data set {r} has a single treatment arm although the "
                               f"50,000-row sample had both")
        true_att = float((d["y1"] - d["y0"])[d["t"] == 1].mean())
        for name, make in estimators().items():
            ate, att, _ = estimate(make, w, t, y)
            rows.append({"rep": r, "estimator": name, "ate_true": true_ate, "att_true": true_att,
                         "ate_est": ate, "att_est": att})
    log.info(f"{arm}: {len(rows) // len(estimators())} estimator reps done, {time.time() - started:.0f}s elapsed")
    return {
        "data": ds, "n": int(len(real)), "target_effect": f_val, "standardized": standardize,
        "max_epochs": max_epochs, "seconds_fit": round(t_fit, 1), "treatment_collapsed": bool(collapsed),
        "diverged": bool(diverged), "ite_mean": float(np.nanmean(ite)) if not np.isnan(ite).all() else None,
        "ite_sd": float(np.nanstd(ite)) if not np.isnan(ite).all() else None,
        "ite_abs_dev_from_target_mean": float(np.nanmean(np.abs(ite - f_val))) if not np.isnan(ite).all() else None,
        "share_negative_earnings": {v: float((big[v] < 0).mean()) for v in ["re74", "re75", "y0", "y1"]},
        "treated_share_generated": float(big["t"].mean()), "treated_share_real": float(real["t"].mean()),
        "means_by_t": means, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-epochs", type=int, default=30, help="upstream default 100")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--raw-epochs", type=int, default=5, help="epochs of the upstream raw-dollar check")
    ap.add_argument("--datasets", nargs="+", default=["psid", "experimental"],
                    help="ds-wgan copies of LaLonde: psid (6.9%% treated) or experimental (NSW, 41.6%%)")
    ap.add_argument("--tag", default="credence")
    args = ap.parse_args()
    log = common.setup(args.tag)
    started = time.time()
    out = {"source": "github.com/amzn/credence-to-causal-estimation (credence-v2)",
           "commit": "195cbf6febf13d53dcb1a864b80fd39794dfe584", "args": vars(args), "arms": {}}
    for ds in args.datasets:
        real = pd.read_feather(common.SRC / "ds-wgan" / "data" / "original_data" / f"{ds}.feather")[CAT + NUM]
        # upstream's own setting (raw dollars), short: does it train at all?
        out["arms"][f"{ds}, f = 0, raw dollars (upstream)"] = run_arm(
            ds, real, "f = 0", 0.0, args, log, started, standardize=False, max_epochs=args.raw_epochs, reps=0)
        for effect, f_val in EFFECTS.items():
            out["arms"][f"{ds}, {effect}"] = run_arm(ds, real, effect, f_val, args, log, started)
    out["versions"] = common.pkg_versions("torch", "pytorch-lightning", "causallib", "pandas")
    common.save(args.tag, out, started)


if __name__ == "__main__":
    main()
