"""OpenXAI (Agarwal et al., NeurIPS 2022 D&B): pretrained models, seven explainers and OpenXAI's own metrics.

A small run of the upstream pipeline (generate_explanations.py, then evaluate_metrics.py) with these changes:
- the first --n-test (50) test points instead of 1,000; PGI and PGU (AUC over k = 1 .. 25% of the features,
  100 perturbations each, as upstream) on the first --n-pg (10) of them, because they cost ~0.08 s per point
  per k; stability metrics (RIS, RRS, ROS) on the first --n-stab (10) points, each re-explained at
  --n-perturb (10) same-class perturbations kept from upstream's 1,000 candidates, instead of 100;
- captum's KernelSHAP evaluates its 500 samples in one batch (perturbations_per_eval) instead of one model call
  each; checked on 4 Adult/LR and 4 German/ANN points: same attributions to 2.4e-7 and 2.8e-6;
- metric parallelism off (upstream n_jobs -1 takes every core);
- datasets (Harvard Dataverse) and pretrained models download into _data/openxai/.
Ground-truth metrics exist only for the logistic regression ("lr"): the reference is its coefficient vector.

Usage (from synthetic_data/benchmarks/):
    .venv/Scripts/python.exe run_openxai.py --models lr --tag openxai_lr     # ~10 min (estimate)
    .venv/Scripts/python.exe run_openxai.py --models ann --tag openxai_ann   # ~13 min (estimate)
    .venv/Scripts/python.exe run_openxai.py --datasets german --models lr --tag openxai_smoke
Writes results/<tag>.json.
"""

import common  # noqa: F401  (thread caps first)

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(common.SRC / "OpenXAI"))
warnings.filterwarnings("ignore")

from openxai.dataloader import ReturnLoaders, ReturnTrainTestX  # noqa: E402
from openxai.evaluator import Evaluator, ground_truth_metrics, prediction_metrics, stability_metrics  # noqa: E402
from openxai.experiment_utils import fill_param_dict, invalid_model_metric_combination, load_config  # noqa: E402
from openxai.explainer import Explainer  # noqa: E402
from openxai.explainers.perturbation_methods import get_perturb_method  # noqa: E402
from openxai.explainers.catalog.shap_explainer.shap_explainer_captum import SHAPExplainerC  # noqa: E402
from openxai.model import LoadModel  # noqa: E402

DATASETS = ["adult", "compas", "gaussian", "german", "gmsc", "heart", "heloc", "pima"]
METHODS = ["control", "grad", "ig", "itg", "sg", "shap", "lime"]
METRICS = ["PRA", "RC", "FA", "RA", "SA", "SRA", "PGU", "PGI", "RIS", "RRS", "ROS"]


def _batched_shap(self, x, label=None):
    """Upstream SHAPExplainerC.get_explanations with all KernelSHAP samples in one forward call."""
    label = self.model(x.float()).argmax(dim=-1) if label is None else label
    if self.seed is not None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
    return torch.FloatTensor(self.explainer.attribute(x.float(), target=label, n_samples=self.n_samples,
                                                      perturbations_per_eval=self.n_samples))


SHAPExplainerC.get_explanations = _batched_shap


def metric_params(config: dict, metric: str, expl: np.ndarray, x_test, preds, meta, explainer, data_name, args) -> dict:
    """Same parameters as upstream evaluate_metrics._construct_param_dict, minus parallelism and with smaller n."""
    if metric in ground_truth_metrics:
        p = dict(config["evaluators"]["ground_truth_metrics"], explanations=expl)
        if metric in ["FA", "RA", "SA", "SRA"]:
            p["predictions"] = preds
        else:
            del p["k"], p["AUC"]
        return p
    if metric in prediction_metrics:
        p = dict(config["evaluators"]["prediction_metrics"], inputs=x_test[: args.n_pg],
                 explanations=expl[: args.n_pg], feature_metadata=meta, n_jobs=None)
        p["perturb_method"] = get_perturb_method(p.pop("std"), data_name)
        return p
    p = dict(config["evaluators"]["stability_metrics"], inputs=x_test[: args.n_stab], explainer=explainer,
             feature_metadata=meta, n_jobs=None, n_perturbations=args.n_perturb)
    p["perturb_method"] = get_perturb_method(p.pop("std"), data_name)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=["lr", "ann"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--n-pg", type=int, default=10)
    ap.add_argument("--n-stab", type=int, default=10)
    ap.add_argument("--n-perturb", type=int, default=10)
    ap.add_argument("--tag", default="openxai")
    args = ap.parse_args()
    log = common.setup(args.tag)
    started = time.time()
    config = load_config(str(common.SRC / "OpenXAI" / "experiment_config.json"))
    os.chdir(common.workdir("openxai"))   # upstream downloads to ./data and ./models relative to the cwd
    rows, models_info = [], []
    n_jobs_total = len(args.datasets) * len(args.models)
    for i, data_name in enumerate(args.datasets):
        x_train, x_test_all, meta = ReturnTrainTestX(data_name, float_tensor=True, return_feature_metadata=True)
        x_test = x_test_all[: args.n_test]
        for model_name in args.models:
            torch.manual_seed(common.SEED)
            np.random.seed(common.SEED)
            model = LoadModel(data_name, model_name, pretrained=True)
            _, test_loader = ReturnLoaders(data_name)
            y_all = torch.as_tensor(np.asarray(test_loader.dataset.targets), dtype=torch.long)
            with torch.no_grad():
                acc = float((model(x_test_all).argmax(-1) == y_all).float().mean())
                preds = model(x_test).argmax(dim=-1)
            models_info.append({"data": data_name, "model": model_name, "n_features": int(x_test.shape[1]),
                                "n_train": int(x_train.shape[0]), "n_test_all": int(x_test_all.shape[0]),
                                "test_accuracy": acc})
            log.info(f"{data_name}/{model_name}: d={x_test.shape[1]}, test accuracy {acc:.4f}")
            for method in args.methods:
                t = time.time()
                p_expl = fill_param_dict(method, dict(config["explainers"][method]), x_train)
                explainer = Explainer(method, model, p_expl)
                expl = explainer.get_explanations(x_test, preds).detach().numpy()
                t_expl = time.time() - t
                for metric in args.metrics:
                    if invalid_model_metric_combination(model_name, metric):
                        continue
                    t = time.time()
                    ev = Evaluator(model, metric=metric)
                    score, mean_score = ev.evaluate(**metric_params(config, metric, expl, x_test, preds, meta,
                                                                    explainer, data_name, args))
                    score = np.asarray(score, dtype=float)
                    rows.append({"data": data_name, "model": model_name, "method": method, "metric": metric,
                                 "mean": float(mean_score), "se": float(np.std(score) / np.sqrt(score.size)),
                                 "n": int(score.size), "seconds_metric": round(time.time() - t, 2),
                                 "seconds_explain": round(t_expl, 2)})
                log.info(f"  {method}: explained in {t_expl:.1f}s; {len(rows)} metric rows so far")
            done = i * len(args.models) + args.models.index(model_name) + 1
            el = time.time() - started
            log.info(f"{done}/{n_jobs_total} model-data pairs, {el:.0f}s elapsed, ~{el / done * (n_jobs_total - done):.0f}s left")
    common.save(args.tag, {"source": "github.com/AI4LIFE-GROUP/OpenXAI", "args": vars(args),
                           "config_upstream": {k: config[k] for k in ("explainers", "evaluators")},
                           "versions": common.pkg_versions("torch", "captum", "shap", "lime", "numpy"),
                           "models": models_info, "rows": rows}, started)


if __name__ == "__main__":
    main()
