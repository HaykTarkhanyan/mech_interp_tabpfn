"""How much does OpenXAI's per-split scaling change its pretrained models' test accuracy?

openxai/dataloader.py fits a fresh MinMaxScaler on each split, so the test features are scaled with the test set's
own minimum and maximum. This scores every pretrained model on the test split scaled both ways: as OpenXAI does
(test statistics) and as intended (training statistics).

Usage (from synthetic_data/benchmarks/, after run_openxai.py has downloaded the data and models):
    .venv/Scripts/python.exe probe_openxai_scaler.py       # ~10 s
Writes results/openxai_scaler_probe.json.
"""

import common  # noqa: F401  (thread caps first)

import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

sys.path.insert(0, str(common.SRC / "OpenXAI"))
from openxai.dataloader import labels  # noqa: E402
from openxai.model import LoadModel  # noqa: E402

DATASETS = ["adult", "compas", "gaussian", "german", "gmsc", "heart", "heloc", "pima"]


def main() -> None:
    log = common.setup("openxai_scaler_probe")
    started = time.time()
    os.chdir(common.workdir("openxai"))
    rows = []
    for ds in DATASETS:
        tr = pd.read_csv(f"./data/{ds}/{ds}-train.csv")
        te = pd.read_csv(f"./data/{ds}/{ds}-test.csv")
        y = te[labels[ds]].to_numpy()
        x_tr, x_te = tr.drop(columns=labels[ds]), te.drop(columns=labels[ds])
        variants = {"test-split scaler (OpenXAI)": MinMaxScaler().fit(x_te).transform(x_te),
                    "training-split scaler": MinMaxScaler().fit(x_tr).transform(x_te)}
        for model_name in ["lr", "ann"]:
            model = LoadModel(ds, model_name, pretrained=True)
            for v, x in variants.items():
                with torch.no_grad():
                    acc = float((model(torch.FloatTensor(x)).argmax(-1).numpy() == y).mean())
                rows.append({"data": ds, "model": model_name, "scaling": v, "test_accuracy": acc})
        diff = np.abs(variants["test-split scaler (OpenXAI)"] - variants["training-split scaler"]).max()
        log.info(f"{ds}: max |difference| between the two scalings {diff:.3f}; "
                 + ", ".join(f"{r['model']} {r['scaling'][:4]} {r['test_accuracy']:.4f}" for r in rows[-4:]))
    common.save("openxai_scaler_probe", {"source": "OpenXAI a182886, data and models from run_openxai.py",
                                         "rows": rows}, started)


if __name__ == "__main__":
    main()
