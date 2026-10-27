"""Miro-Nicolau et al. (2026), tabular arm: rerun the released notebook (github.com/miquelmn/fb2ts, tabular.ipynb)
on CPU, with both ground-truth definitions and a random baseline.

What the paper and the released code say differ:
- ground truth: the paper (Eq. 5) lowers feature i by one discrete level, phi_i = f(x) - f(x_i - 1 level); the
  notebook sets it to zero, phi_i = f(x) - f(x_i := 0);
- label weights: the paper gives w = [0.5, 0.25, 0.125]; attr_functions.ssin uses 1/2, 1/4, 1/6;
- data: the paper describes a discretized grid (about 4 million samples); aixi-dataset/bin/daixi.py draws
  continuous uniform(0, 1) values (50,000 rows) and never seeds them.
This run puts the features on a grid of --levels levels (so a one-level step exists), uses the code's weights and
label (ssin > 0.6), the notebook's MLP and training recipe, and scores Gradient, DeepLIFT, Integrated Gradients,
KernelSHAP and LIME against both ground truths with the notebook's own SIM and KL code (AUC-Borji only with --auc:
the paper does not use it for tabular data, and its loops take ~30 min at 500 points). LORE (xailib) is left out.
A uniform random attribution gives the chance level of each metric.

Usage (from synthetic_data/benchmarks/):
    .venv/Scripts/python.exe run_fb2ts_tabular.py          # ~7 min on 2 threads (5 min training + explanations)
    .venv/Scripts/python.exe run_fb2ts_tabular.py --auc    # 36 min (measured 2026-10-27); made results/fb2ts_tabular.json
Writes results/fb2ts_tabular.json.
"""

import common  # noqa: F401  (thread caps first)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from captum import attr  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from scipy import stats  # noqa: E402
from torch import nn  # noqa: E402

nb = json.loads((common.SRC / "fb2ts" / "tabular.ipynb").read_text(encoding="utf-8"))
METRIC_CELL = "".join(nb["cells"][24]["source"])   # kl, AUC_Borji, sim exactly as released
_ns = {"np": np, "torch": torch, "stats": stats, "np_rand": np.random}
exec("from sklearn.preprocessing import MinMaxScaler\n" + METRIC_CELL.replace("np.trapz", "np.trapezoid"), _ns)
METRICS = _ns["metrics"]                             # {"kl": ..., "auc": ..., "sim": ...}
WEIGHTS_CODE = (1 / 2, 1 / 4, 1 / 6)


def ssin(x: np.ndarray, w=WEIGHTS_CODE) -> np.ndarray:
    return sum(w[j] * np.sin(np.pi * x[:, j] / 2) for j in range(3))


def train(x_tr, y_tr, x_te, y_te, epochs: int, log) -> tuple:
    """The notebook's cell 17: full-batch Adam 1e-4, StepLR(150, 0.85), keep the best test accuracy."""
    torch.manual_seed(common.SEED)
    net = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.85)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, t0 = 0.0, None, time.time()
    for epoch in range(epochs):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(x_tr), y_tr)
        loss.backward()
        opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            acc = float(((net(x_te) > 0).float() == y_te).float().mean())
        if acc > best:
            best, best_state = acc, copy.deepcopy(net.state_dict())
        if (epoch + 1) % 500 == 0:
            el = time.time() - t0
            log.info(f"epoch {epoch + 1}/{epochs}: loss {loss.item():.4f}, best test acc {best:.4f}, "
                     f"{el:.0f}s elapsed, ~{el / (epoch + 1) * (epochs - epoch - 1):.0f}s left")
    net.load_state_dict(best_state)
    net.eval()
    return net, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, default=100, help="grid levels per feature")
    ap.add_argument("--n", type=int, default=50000, help="rows drawn from the grid (daixi.py default)")
    ap.add_argument("--epochs", type=int, default=2500, help="notebook 5000; StepLR(150, 0.85) leaves lr < 1e-5 by 2500")
    ap.add_argument("--n-explain", type=int, default=500)
    ap.add_argument("--kernel-samples", type=int, default=200, help="notebook 200; paper Table 4 says 25")
    ap.add_argument("--auc", action="store_true", help="also AUC-Borji: pure-Python loops, ~30 min at 500 points")
    ap.add_argument("--tag", default="fb2ts_tabular")
    args = ap.parse_args()
    log = common.setup(args.tag)
    metrics = METRICS if args.auc else {k: v for k, v in METRICS.items() if k != "auc"}
    started = time.time()
    rng = np.random.default_rng(common.SEED)
    step = 1.0 / (args.levels - 1)
    x = rng.integers(0, args.levels, size=(args.n, 3)) * step
    y = (ssin(x) > 0.6).astype(np.float32)
    n_tr = int(0.8 * args.n)
    x_tr, x_te, y_tr, y_te = x[:n_tr], x[n_tr:], y[:n_tr], y[n_tr:]
    # the notebook balances the training classes by subsampling the larger one
    pos, neg = np.flatnonzero(y_tr == 1), np.flatnonzero(y_tr == 0)
    m = min(len(pos), len(neg))
    keep = rng.permutation(np.concatenate([rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)]))
    log.info(f"grid {args.levels}^3, step {step:.4f}; train {len(keep)} balanced of {n_tr}, positive share "
             f"{y.mean():.3f}")
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    net, best_acc = train(t(x_tr[keep]), t(y_tr[keep]).reshape(-1, 1), t(x_te), t(y_te).reshape(-1, 1), args.epochs, log)
    log.info(f"best test accuracy {best_acc:.4f}")

    # explain test points that are not on the lowest level of any feature (the one-level step needs a level below)
    idx = np.flatnonzero((x_te > 0).all(1))[: args.n_explain]
    xe = t(x_te[idx])
    with torch.no_grad():
        out = net(xe)[:, 0]
        gt = {"code: f(x) - f(x_i := 0)": torch.zeros_like(xe), "paper Eq. 5: f(x) - f(x_i - 1 level)": torch.zeros_like(xe)}
        for i in range(3):
            z = xe.clone()
            z[:, i] = 0.0
            gt["code: f(x) - f(x_i := 0)"][:, i] = out - net(z)[:, 0]
            z = xe.clone()
            z[:, i] -= step
            gt["paper Eq. 5: f(x) - f(x_i - 1 level)"][:, i] = out - net(z)[:, 0]
        correct = ((out > 0).float().numpy() == y_te[idx])
    lime_ex = LimeTabularExplainer(x_tr[keep], discretize_continuous=True, mode="regression", random_state=common.SEED)

    def lime_one(row):
        e = lime_ex.explain_instance(row, lambda a: net(t(a)).detach().numpy()[:, 0], num_features=3, top_labels=1)
        w = dict(next(iter(e.as_map().values())))
        return [w[j] for j in range(3)]

    gen = torch.Generator().manual_seed(common.SEED)
    methods = {
        "Gradient": lambda: attr.Saliency(net).attribute(xe, target=0),
        "DeepLIFT": lambda: attr.DeepLift(net).attribute(xe, target=0),
        "Integrated Gradients": lambda: attr.IntegratedGradients(net).attribute(xe, target=0),
        "KernelSHAP": lambda: torch.cat([attr.KernelShap(net).attribute(xe[k:k + 1], target=0, n_samples=args.kernel_samples)
                                         for k in range(len(xe))]),
        "LIME": lambda: t([lime_one(r) for r in x_te[idx]]),
        "Random (uniform)": lambda: torch.rand(xe.shape, generator=gen),
    }
    torch.manual_seed(common.SEED)
    np.random.seed(common.SEED)   # AUC_Borji samples with numpy's global state
    rows = []
    for name, fn in methods.items():
        t0 = time.time()
        a = fn().detach().numpy()
        secs = time.time() - t0
        for gt_name, g in gt.items():
            g = g.numpy()
            vals = {mk: [float(mf(g[k], a[k])) for k in range(len(a)) if correct[k]] for mk, mf in metrics.items()}
            rows.append({"method": name, "ground_truth": gt_name, "seconds": round(secs, 2),
                         **{f"{mk}_mean": float(np.nanmean(v)) for mk, v in vals.items()},
                         **{f"{mk}_sd": float(np.nanstd(v)) for mk, v in vals.items()}, "n": int(correct.sum())})
        log.info(f"{name}: {secs:.1f}s; SIM vs code GT {rows[-2]['sim_mean']:.3f}, vs paper GT {rows[-1]['sim_mean']:.3f}")
    # how different are the two ground truths from each other?
    g0, g1 = (g.numpy() for g in gt.values())
    gt_agree = {mk: float(np.nanmean([mf(g0[k], g1[k]) for k in range(len(g0)) if correct[k]])) for mk, mf in metrics.items()}
    order_agree = float(np.mean([(np.argsort(g0[k]) == np.argsort(g1[k])).all() for k in range(len(g0)) if correct[k]]))
    log.info(f"paper GT scored against code GT: {gt_agree}; identical feature order in {order_agree:.3f} of points")
    common.save(args.tag, {
        "source": "github.com/miquelmn/fb2ts (tabular.ipynb, attr_functions.py); aixi-dataset/bin/daixi.py",
        "commit": "d49d3a7905883ec6e2861a46488ad0b28cf6619f", "args": vars(args),
        "weights_used": WEIGHTS_CODE, "weights_paper": [0.5, 0.25, 0.125], "best_test_accuracy": best_acc,
        "n_explained": int(len(idx)), "n_correct": int(correct.sum()),
        "paper_table7_sim": {"Gradient": 0.980, "LORE": 0.784, "DeepLIFT": 0.636, "Integrated Gradients": 0.642,
                             "KernelSHAP": 0.601, "LIME": 0.560},
        "ground_truths_agreement": gt_agree, "ground_truths_same_order_share": order_agree,
        "versions": common.pkg_versions("torch", "captum", "lime", "numpy"), "rows": rows}, started)


if __name__ == "__main__":
    sys.exit(main())
