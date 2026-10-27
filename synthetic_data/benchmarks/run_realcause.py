"""RealCause (Neal et al. 2020): two cheap runs on what the repository ships, no generator training.

A. Re-analysis of causal-predictive-analysis.csv (1,568 fitted estimators): within each (dataset, meta-estimator,
   model class) group, the Spearman correlation between predictive error (CV RMSE for outcome models, -F1 / -AP /
   -balanced accuracy for propensity models) and causal error (ATE RMSE, PEHE), then the median over groups, as in
   experiments/uai_analysis.py (Tables 4-5). That script reads intermediate CSVs that are not in the repo and uses
   pandas' removed Series.append, so the same logic is restated here (Spearman only).
B. RealCause's own estimator wrappers (causal_estimators/, causallib) on --n-samples of the 100 pre-sampled datasets
   per realistic dataset (realcause_datasets/, true potential outcomes included): ATE bias and RMSE against each
   sample's true ATE (mean of ite), and PEHE for the outcome-model estimators.

Usage (from synthetic_data/benchmarks/):
    .venv/Scripts/python.exe run_realcause.py              # ~3 min (estimate)
Writes results/realcause.json.
"""

import common  # noqa: F401  (thread caps first)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402

RC = common.SRC / "realcause"
sys.path.insert(0, str(RC))
warnings.filterwarnings("ignore")

from causal_estimators.ipw_estimator import IPWEstimator  # noqa: E402
from causal_estimators.standardization_estimator import (StandardizationEstimator,  # noqa: E402
                                                         StratifiedStandardizationEstimator)

DATASETS = {"lalonde_psid": "re78", "lalonde_cps": "re78", "twins": "mortality"}


def group_spearman(df: pd.DataFrame, group_cols: list, pred_col: str, causal_col: str) -> pd.DataFrame:
    """uai_analysis.get_correlation: negate sklearn's 'higher is better' scores so both sides are errors."""
    out = []
    for key, g in df.groupby(group_cols):
        g = g[[pred_col, causal_col]].dropna()
        if len(g) <= 1:
            continue
        rho = spearmanr(-g[pred_col], g[causal_col]).correlation
        out.append({**dict(zip(group_cols, key)), "pred_score": pred_col, "causal_score": causal_col,
                    "n": len(g), "spearman": float(rho)})
    return pd.DataFrame(out)


def part_a() -> dict:
    df = pd.read_csv(RC / "causal-predictive-analysis.csv")
    stand = df[df["meta-estimator"].isin(["standardization", "stratified_standardization"])]
    ipw = df[df["meta-estimator"].isin(["ipw", "ipw_trimeps.01"])]
    outcome = pd.concat([group_spearman(stand, ["dataset", "meta-estimator", "outcome_model"],
                                        "mean_test_neg_root_mean_squared_error", c) for c in ["ate_rmse", "mean_pehe"]])
    prop = pd.concat([group_spearman(ipw, ["dataset", "meta-estimator", "prop_score_model"], s, "ate_rmse")
                      for s in ["mean_test_f1", "mean_test_average_precision", "mean_test_balanced_accuracy"]])
    return {
        "outcome_median_spearman": outcome.groupby(["dataset", "causal_score"])["spearman"].median().reset_index().to_dict("records"),
        "propensity_median_spearman": prop.groupby("dataset")["spearman"].median().reset_index().to_dict("records"),
        "outcome_groups": outcome.to_dict("records"), "propensity_groups": prop.to_dict("records"),
        "paper_tables_4_5": {"outcome_vs_ate_rmse": {"lalonde_psid": 0.77, "lalonde_cps": 0.81, "twins": 0.81},
                             "outcome_vs_pehe": {"lalonde_psid": 0.92, "lalonde_cps": 0.80, "twins": 0.91},
                             "propensity_vs_ate_rmse": {"lalonde_psid": 0.22, "lalonde_cps": 0.51, "twins": 0.38}},
    }


def estimators() -> dict:
    return {
        "difference in means": None,
        "S-learner: linear": lambda: StandardizationEstimator(outcome_model=LinearRegression()),
        "T-learner: linear": lambda: StratifiedStandardizationEstimator(outcome_models=LinearRegression()),
        "T-learner: gradient boosting": lambda: StratifiedStandardizationEstimator(
            outcome_models=GradientBoostingRegressor(random_state=common.SEED)),
        "T-learner: random forest": lambda: StratifiedStandardizationEstimator(
            outcome_models=RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=common.SEED, n_jobs=1)),
        "IPW: logistic": lambda: IPWEstimator(prop_score_model=LogisticRegression(max_iter=1000)),
        "IPW: logistic, trimmed 0.01": lambda: IPWEstimator(prop_score_model=LogisticRegression(max_iter=1000), trim_weights=True),
    }


def part_b(n_samples: int, log) -> list:
    rows = []
    for ds in DATASETS:
        for s in range(n_samples):
            d = pd.read_csv(RC / "realcause_datasets" / f"{ds}_sample{s}.csv")
            w = d.drop(columns=["t", "y", "y0", "y1", "ite"])
            t, y, ite = d["t"].astype(int), d["y"], d["ite"].to_numpy()
            true_ate, true_att = float(ite.mean()), float(ite[t.to_numpy() == 1].mean())
            for name, make in estimators().items():
                t0 = time.time()
                pehe = None
                if make is None:
                    est = float(y[t == 1].mean() - y[t == 0].mean())
                else:
                    e = make()
                    e.fit(w, t, y)
                    est = float(e.estimate_ate())
                    if hasattr(e, "estimate_ite"):
                        pehe = float(np.sqrt(np.mean((np.asarray(e.estimate_ite(), dtype=float) - ite) ** 2)))
                rows.append({"dataset": ds, "sample": s, "estimator": name, "ate_true": true_ate,
                             "att_true": true_att, "treated_share": float(t.mean()), "ate_est": est,
                             "pehe": pehe, "seconds": round(time.time() - t0, 2)})
        log.info(f"{ds}: {n_samples} samples done")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-samples", type=int, default=20, help="of the 100 pre-sampled datasets per realistic dataset")
    ap.add_argument("--tag", default="realcause")
    args = ap.parse_args()
    log = common.setup(args.tag)
    started = time.time()
    a = part_a()
    for r in a["outcome_median_spearman"] + a["propensity_median_spearman"]:
        log.info(f"median Spearman {r}")
    b = part_b(args.n_samples, log)
    common.save(args.tag, {"source": "github.com/bradyneal/realcause", "commit": "7411cc26f10d750aa011b25f2a46cb572d8b8540",
                           "args": vars(args), "outcomes": DATASETS,
                           "versions": common.pkg_versions("causallib", "scikit-learn", "pandas", "numpy"),
                           "part_a": a, "part_b": b}, started)


if __name__ == "__main__":
    main()
