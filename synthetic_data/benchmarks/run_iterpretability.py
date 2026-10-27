"""ITErpretability (Crabbe et al., NeurIPS 2022 D&B): the predictive-sensitivity experiment on the real Twins
covariates, with a random-attribution baseline and the prognostic score computed both ways.

The loop is upstream's PredictiveSensitivity.run (src/iterpretability/experiments.py) on the same components (linear
simulator, the five catenets learners with upstream hyperparameters, captum explainers, attribution_accuracy), run
from here so that it can also record:
- the chance level: attribution_accuracy of uniformly random attributions on the same points;
- prognostic accuracy as upstream computes it and at top-k: get_prognostic_features() returns the tuple from np.where,
  so attribution_accuracy takes len(tuple) = 1 as the number of important features (a top-1 hit rate).
Upstream defaults kept: num_important_features 8 for Twins (the paper's rule floor(0.2 d) gives 7), noise sd 0.1,
n_iter 1000, 80/20 split by row order. Each learner is seeded before fitting (upstream seeds once per scale).
Smaller than upstream: one seed (upstream 30), --explainer-limit test rows explained (upstream 1,000), --scales.

Usage (from synthetic_data/benchmarks/):
    .venv/Scripts/python.exe run_iterpretability.py                       # ~15 min (estimate from the smoke run)
    .venv/Scripts/python.exe run_iterpretability.py --scales 1 --tag iterp_smoke
Writes results/<tag>.json. Twins downloads (bitbucket, ~1 MB) into the clone's catenets/datasets/data/.
"""

import common  # noqa: F401  (thread caps first)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(common.SRC / "ITErpretability"))
warnings.filterwarnings("ignore")

import catenets.models as cate_models  # noqa: E402
from src.iterpretability.datasets.data_loader import load  # noqa: E402
from src.iterpretability.explain import Explainer  # noqa: E402
from src.iterpretability.synthetic_simulate import SyntheticSimulatorLinear  # noqa: E402
from src.iterpretability.utils import attribution_accuracy, compute_pehe, enable_reproducible_results  # noqa: E402


def learners(d: int, binary: bool, n_iter: int) -> dict:
    """Upstream PredictiveSensitivity.run, verbatim hyperparameters."""
    common_kw = dict(binary_y=binary, n_iter=n_iter, batch_size=1024, batch_norm=False, nonlin="relu")
    return {
        "TLearner": cate_models.torch.TLearner(d, n_layers_out=2, n_units_out=100, **common_kw),
        "SLearner": cate_models.torch.SLearner(d, n_layers_out=2, n_units_out=100, **common_kw),
        "TARNet": cate_models.torch.TARNet(d, n_layers_r=1, n_layers_out=1, n_units_out=100, n_units_r=100, **common_kw),
        "DRLearner": cate_models.torch.DRLearner(d, n_layers_out=2, n_units_out=100, **common_kw),
        "XLearner": cate_models.torch.XLearner(d, n_layers_out=2, n_units_out=100, **common_kw),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="twins")
    ap.add_argument("--n-important", type=int, default=8, help="upstream run_experiments.py default for twins")
    ap.add_argument("--scales", nargs="+", type=float, default=[0.01, 0.5, 2.0], help="upstream 1e-3 .. 2 (6 values)")
    ap.add_argument("--seed", type=int, default=1, help="upstream seed_list starts at 1")
    ap.add_argument("--explainer-limit", type=int, default=50)
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--explainers", nargs="+",
                    default=["feature_ablation", "feature_permutation", "integrated_gradients", "shapley_value_sampling"])
    ap.add_argument("--tag", default="iterpretability")
    args = ap.parse_args()
    log = common.setup(args.tag)
    started = time.time()
    x_train_raw, x_test_raw = load(args.dataset, train_ratio=0.8)
    d = x_train_raw.shape[1]
    sim = SyntheticSimulatorLinear(x_train_raw, num_important_features=args.n_important, random_feature_selection=True,
                                   seed=args.seed)
    pred_f, prog_tuple = sim.get_predictive_features(), sim.get_prognostic_features()
    prog_f = np.asarray(prog_tuple[0])
    log.info(f"{args.dataset}: d={d}, train {len(x_train_raw)}, test {len(x_test_raw)}; predictive {len(pred_f)}, "
             f"prognostic {len(prog_f)} (upstream passes a {type(prog_tuple).__name__} of length {len(prog_tuple)})")
    rows, chance = [], []
    rng = np.random.default_rng(common.SEED)
    for i, scale in enumerate(args.scales):
        x_tr, w_tr, y_tr, _, _, _ = sim.simulate_dataset(x_train_raw, predictive_scale=scale)
        x_te, _, _, _, _, _ = sim.simulate_dataset(x_test_raw, predictive_scale=scale)
        cate_test = sim.te(x_te)
        x_e = x_te[: args.explainer_limit]
        rand = np.abs(rng.normal(size=x_e.shape))
        chance.append({"scale": scale, "pred_acc_random": float(attribution_accuracy(pred_f, rand)),
                       "all_acc_random": float(attribution_accuracy(sim.get_all_important_features(), rand)),
                       "prog_acc_random_upstream": float(attribution_accuracy(prog_tuple, rand)),
                       "prog_acc_random_topk": float(attribution_accuracy(prog_f, rand))})
        for name, lrn in learners(d, len(np.unique(y_tr)) == 2, args.n_iter).items():
            enable_reproducible_results(args.seed)
            t0 = time.time()
            lrn.fit(X=x_tr, y=y_tr, w=w_tr)
            t_fit = time.time() - t0
            expl = Explainer(lrn, feature_names=list(range(d)), explainer_list=args.explainers).explain(x_e)
            t_expl = time.time() - t0 - t_fit
            pehe = float(compute_pehe(cate_true=cate_test, cate_pred=lrn.predict(X=x_te)))
            for e_name, a in expl.items():
                a = np.abs(a)
                rows.append({"scale": scale, "learner": name, "explainer": e_name,
                             "pred_acc": float(attribution_accuracy(pred_f, a)),
                             "all_acc": float(attribution_accuracy(sim.get_all_important_features(), a)),
                             "prog_acc_upstream": float(attribution_accuracy(prog_tuple, a)),
                             "prog_acc_topk": float(attribution_accuracy(prog_f, a)),
                             "pehe": pehe, "pehe_normalized": float(pehe / np.sqrt(np.var(cate_test))),
                             "cate_sd": float(np.std(cate_test)),
                             "seconds_fit": round(t_fit, 1), "seconds_explain": round(t_expl, 1)})
            ig = [r for r in rows if r["learner"] == name and r["scale"] == scale and r["explainer"] == "integrated_gradients"]
            log.info(f"scale {scale} {name}: fit {t_fit:.0f}s, explain {t_expl:.0f}s, PEHE {pehe:.3f}"
                     + (f", IG pred acc {ig[0]['pred_acc']:.3f}" if ig else ""))
        el = time.time() - started
        log.info(f"{i + 1}/{len(args.scales)} scales, {el:.0f}s elapsed, ~{el / (i + 1) * (len(args.scales) - i - 1):.0f}s left")
    common.save(args.tag, {
        "source": "github.com/vanderschaarlab/ITErpretability", "commit": "8121b9f78c766e0a4a41c4c03e6378a6b8f5c20d",
        "args": vars(args), "d": int(d), "n_predictive": int(len(pred_f)), "n_prognostic": int(len(prog_f)),
        "chance_pred_acc_code_rule": float(len(pred_f) / d),
        "chance_pred_acc_paper_rule": float(2 * int(0.2 * d) / d),
        "random_baseline": chance,
        "versions": common.pkg_versions("torch", "captum", "numpy", "pandas"), "rows": rows}, started)


if __name__ == "__main__":
    main()
