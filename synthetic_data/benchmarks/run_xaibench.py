"""XAI-BENCH (Liu et al., NeurIPS 2021 D&B): the upstream Experiment over a dataset x correlation grid, plus one
addition: every explainer is also scored against the interventional Shapley reference.

XAI-BENCH's ground-truth Shapley (custom_explainers/ground_truth_shap.py) is the *conditional* (observational) one:
coalition values are means over 20,000 draws from the Gaussian conditional distribution. Its "brutekernelshap"
explainer is exact *interventional* Shapley over 1,000 training rows. At rho > 0 the two estimands differ, so part
of every interventional explainer's "error" is the estimand gap, not explainer failure (Chen et al. 2020). This run
reports the upstream metrics and, from the same explanations, each explainer's MSE against brutekernelshap.

Changes from upstream (patches/xai-bench.patch): L2X (TensorFlow 2.5) and BreakDown (blist) are not installed and
are skipped; removed numpy aliases (np.int, np.bool) replaced by the builtins; SHAPR's weighted mean evaluates the
model once per coalition on all reference rows instead of once per row (same arithmetic). ROAR metrics (which
retrain models) are left out. Validation points: --n-val (25) instead of 100.

Usage (from synthetic_data/benchmarks/):
    .venv/Scripts/python.exe run_xaibench.py                                   # ~13 min (estimate from the smoke run)
    .venv/Scripts/python.exe run_xaibench.py --datasets gaussianLinear --rhos 0.5 --tag xaibench_smoke
Writes results/<tag>.json.
"""

import common  # noqa: F401  (thread caps first)

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(common.SRC / "xai-bench"))
warnings.filterwarnings("ignore")

from src import datasets, experiments, explainer, metric, model, parse_utils  # noqa: E402

DATASETS = ["gaussianLinear", "gaussianNonLinearAdditive", "gaussianPiecewiseConstant"]
MODELS = ["lr", "dtree", "mlp"]
EXPLAINERS = [("random", {}), ("shap", {}), ("shapr", {"sigma": 0.4}), ("brutekernelshap", {}),
              ("maple", {"n_estimators": 200, "min_samples_leaf": 10}), ("lime", {"kernel_width": 0.5})]
METRICS = ["faithfulness", "monotonicity", "shapley", "shapley_corr", "infidelity"]
REFERENCE = "brutekernelshap"


def to_float(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()) if v.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.5, 0.9])
    ap.add_argument("--n-val", type=int, default=25)
    ap.add_argument("--conditional", default="observational", choices=["observational", "interventional"],
                    help="reference expectation of the faithfulness and monotonicity metrics (upstream default)")
    ap.add_argument("--seed", type=int, default=7, help="upstream README default")
    ap.add_argument("--tag", default="xaibench")
    args = ap.parse_args()
    log = common.setup(args.tag)
    started = time.time()
    os.chdir(common.workdir("xaibench"))
    rows, perfs, gaps = [], [], []
    conditions = [(d, r) for d in args.datasets for r in args.rhos]
    for i, (data_name, rho) in enumerate(conditions):
        parse_utils.set_global_seed(args.seed)
        data = datasets.Data(data_name, "regression", mu="np.zeros(5)", dim=5, rho=rho,
                             weight="np.array([4, 3, 2, 1, 0])", noise=0.01, num_train_samples=1000,
                             num_val_samples=args.n_val)
        exp = experiments.Experiment(
            data, [model.Model(m, "regression") for m in MODELS],
            [explainer.Explainer(n, **kw) for n, kw in EXPLAINERS],
            [metric.Metric(m, conditional=args.conditional) for m in METRICS])
        res = exp.get_results()
        for m in MODELS:
            perfs.append({"data": data_name, "rho": rho, "model": m, "mse_train_test": res["model_perfs"][m]})
            gt_cond = exp.ground_truth_explanations[m][1]
            ref = exp.explanations[m][REFERENCE][0]
            gaps.append({"data": data_name, "rho": rho, "model": m,
                         "mse_conditional_gt_vs_interventional": float(np.mean((gt_cond - ref) ** 2)),
                         "mean_sq_conditional_gt": float(np.mean(gt_cond ** 2))})
            for e, _ in EXPLAINERS:
                fw = exp.explanations[m][e][0]
                for k, v in res["models"][m][e].items():
                    rows.append({"data": data_name, "rho": rho, "model": m, "explainer": e, "metric": k,
                                 "value": to_float(v)})
                rows.append({"data": data_name, "rho": rho, "model": m, "explainer": e,
                             "metric": "mse_vs_interventional", "value": float(np.mean((fw - ref) ** 2))})
        el = time.time() - started
        log.info(f"{i + 1}/{len(conditions)} {data_name} rho={rho}: {el:.0f}s elapsed, "
                 f"~{el / (i + 1) * (len(conditions) - i - 1):.0f}s left")
    common.save(args.tag, {
        "source": "github.com/abacusai/xai-bench", "commit": "f0431a7c1628a8b66fd322ce86ddcba0d81201d5",
        "args": vars(args), "explainers": [e for e, _ in EXPLAINERS], "metrics": METRICS + ["mse_vs_interventional"],
        "notes": {"shapley": "MSE against XAI-BENCH's conditional ground-truth Shapley (lower is better)",
                  "mse_vs_interventional": f"MSE against {REFERENCE} (exact interventional, 1,000 rows)",
                  "faithfulness": "higher is better", "monotonicity": "higher is better",
                  "infidelity": "lower is better", "shapley_corr": "higher is better"},
        "versions": common.pkg_versions("shap", "lime", "scikit-learn", "numpy", "pandas"),
        "model_perfs": perfs, "estimand_gap": gaps, "rows": rows}, started)


if __name__ == "__main__":
    main()
