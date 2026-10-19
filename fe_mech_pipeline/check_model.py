"""Load TabPFN v2 explicitly, verify the hooked wrapper, and time one forward pass.

Run:  uv run python check_model.py      (31-42 s measured on the laptop; the first run also
                                          downloads the 29 MB checkpoint, +13 s)
      add --device cuda on a GPU machine

Checks, each must pass or the script raises:
  1. the checkpoint is the original v2 file; logs path, size, sha256, parameter count
  2. the sklearn wrapper (pinned preprocessing) hands the model exactly our columns, in
     our order, and its raw logits equal our direct forward pass
  3. hooks registered but inactive leave the output bitwise unchanged
  4. patching activations captured from the same pass back in is bitwise unchanged
  5. mean ablation does change the output (the hooks are live)
  6. timing of one forward pass at the pilot's size, used to scale the pilot

Writes results/check_model.json.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import RESULTS_DIR, SEED, setup, write_json
from model import (
    CKPT_NAME,
    HF_REPO,
    HookedTabPFN,
    checkpoint_path,
    layer_units,
    load_v2,
    pinned_inference_config,
    sha256,
)
from tasks import TASKS, sample_semantic, to_model_columns

N_CTX = 256  # the pilot's context size
N_Q_TIMING = 9 * 128  # the pilot's query batch: 128 reference rows x 9 ALE blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()
    log = setup("check_model")
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    spec = TASKS["interaction"]
    xs, y = sample_semantic(spec, N_CTX, rng)
    x_ctx = to_model_columns(xs, "same_pair")
    xq_s, _ = sample_semantic(spec, 256, rng)
    x_q = to_model_columns(xq_s, "same_pair")
    out: dict = {"n_ctx": N_CTX, "device": args.device}

    # 1. load
    try:
        clf, model = load_v2(x_ctx, y, device=args.device)
    except Exception:
        log.exception(f"loading {CKPT_NAME} from {HF_REPO} failed")
        raise
    path = checkpoint_path()
    n_params = sum(p.numel() for p in model.parameters())
    digest = sha256(path)
    log.info(f"checkpoint {path} ({path.stat().st_size / 1e6:.1f} MB), sha256 {digest}")
    log.info(f"architecture {type(model).__name__}, parameters {n_params:,}")
    out["checkpoint"] = {"file": CKPT_NAME, "repo": HF_REPO, "path": str(path), "sha256": digest,
                         "bytes": path.stat().st_size, "n_params": n_params,
                         "architecture": type(model).__name__}
    cfg = clf.get_inference_config()
    ens = clf.ensemble_configs_
    if len(ens) != 1:
        raise RuntimeError(f"{len(ens)} ensemble members, expected 1")
    e = ens[0]
    active = {
        "preprocess_config": str(e.preprocess_config),
        "add_fingerprint_feature": e.add_fingerprint_feature,
        "polynomial_features": e.polynomial_features,
        "feature_shift_decoder": e.feature_shift_decoder,
        "feature_shift_count": e.feature_shift_count,
        "class_permutation": None if e.class_permutation is None else e.class_permutation.tolist(),
        "outlier_removal_std": e.outlier_removal_std,
        "softmax_temperature": clf.softmax_temperature_,
    }
    log.info(f"active ensemble member: {active}")
    out["wrapper_config"] = {"overrides": {k: str(v) for k, v in pinned_inference_config().items()},
                             "ensemble_member": active,
                             "resolved_FEATURE_SHIFT_METHOD": str(cfg.FEATURE_SHIFT_METHOD),
                             "resolved_FINGERPRINT_FEATURE": cfg.FINGERPRINT_FEATURE}

    # 2. wrapper vs direct
    seen = []

    def grab(_m, args, kwargs):
        x = args[0] if args else kwargs["x"]
        yy = args[1] if len(args) > 1 else kwargs.get("y")
        seen.append((x.detach().clone(), None if yy is None else yy.detach().clone()))

    h = model.register_forward_pre_hook(grab, with_kwargs=True)
    wrapper_logits = clf.predict_raw_logits(x_q)[0]
    h.remove()
    if len(seen) != 1:
        raise RuntimeError(f"wrapper made {len(seen)} model calls, expected 1")
    x_seen, y_seen = seen[0]
    x_seen = x_seen[:, 0, :].double().numpy()
    expect = np.vstack([x_ctx, x_q])
    col_err = float(np.abs(x_seen - expect).max())
    y_err = float(np.abs(y_seen.double().numpy().ravel() - y).max())
    log.info(f"wrapper passes {x_seen.shape} to the model; max |input - our columns| = {col_err:.2e}, labels {y_err:.1e}")
    if x_seen.shape != expect.shape or col_err > 1e-6 or y_err > 0:
        raise RuntimeError("the wrapper changes the columns or labels the model sees")
    hooked = HookedTabPFN(model)
    direct = hooked.logits(x_ctx, y, x_q)
    w_err = float(np.abs(direct - wrapper_logits).max())
    log.info(f"max |wrapper raw logits - direct logits| = {w_err:.2e}")
    if w_err > 1e-4:
        raise RuntimeError(f"direct forward differs from the wrapper by {w_err}")
    out["wrapper_equivalence"] = {"max_abs_input_diff": col_err, "max_abs_logit_diff": w_err}

    # 3. inactive hooks are a no-op
    with_hooks = hooked.logodds(x_ctx, y, x_q)
    hooked.remove_hooks()
    bare = hooked.logodds(x_ctx, y, x_q)  # same forward, no hooks attached at all
    identical = bool(np.array_equal(with_hooks, bare))
    log.info(f"hooks inactive vs hooks removed: bitwise identical = {identical}")
    if not identical:
        raise RuntimeError(f"inactive hooks change output by {np.abs(with_hooks - bare).max()}")
    hooked = HookedTabPFN(model)
    again = hooked.logodds(x_ctx, y, x_q)
    if not np.array_equal(again, bare):
        raise RuntimeError("forward pass is not deterministic")

    # 4. patching in the same pass's activations is a no-op
    full = hooked.capture_full(x_ctx, y, x_q)
    units = {**layer_units("feat", 3), **layer_units("item", 7), **layer_units("mlp", 5),
             ("feat", 11): np.array([0, 4]), ("mlp", 0): np.arange(0, 768, 7)}
    patched = hooked.logodds(x_ctx, y, x_q, units=units, kind="patch", source=full)
    p_ok = bool(np.array_equal(patched, bare))
    log.info(f"self-patch of 5 sites: bitwise identical = {p_ok}")
    if not p_ok:
        raise RuntimeError(f"self-patching changes output by {np.abs(patched - bare).max()}")

    # 5. ablation is live
    means = hooked.capture_means(x_ctx, y, x_q)
    effects = {}
    for site in ("feat", "item", "mlp"):
        for kind in ("mean", "zero"):
            ab = hooked.logodds(x_ctx, y, x_q, units=layer_units(site, 0), kind=kind, source=means)
            effects[f"{site}_L0_{kind}"] = float(np.abs(ab - bare).mean())
    log.info(f"mean |change in log-odds| when ablating layer 0: {effects}")
    if min(effects.values()) <= 0:
        raise RuntimeError("an ablation had no effect at all")
    out["hook_checks"] = {"inactive_hooks_bitwise_identical": identical, "deterministic": True,
                          "self_patch_bitwise_identical": p_ok, "layer0_ablation_mean_abs_change": effects}

    # 6. timing at the pilot's size
    xt_s, _ = sample_semantic(spec, N_Q_TIMING, rng)
    x_t = to_model_columns(xt_s, "same_pair")
    times = []
    for i in range(3):
        t = time.perf_counter()
        hooked.logodds(x_ctx, y, x_t, units=layer_units("feat", 5), kind="mean", source=means)
        times.append(time.perf_counter() - t)
        log.info(f"timing pass {i + 1}/3: {times[-1]:.2f} s for {N_CTX} context + {N_Q_TIMING} query rows")
    out["timing"] = {"n_ctx": N_CTX, "n_query": N_Q_TIMING, "seconds": times,
                     "torch_threads": torch.get_num_threads()}

    out["elapsed_s"] = time.perf_counter() - t0
    write_json(RESULTS_DIR / "check_model.json", out)
    log.info(f"ALL MODEL CHECKS PASSED; wrote results/check_model.json; elapsed {out['elapsed_s']:.1f} s")


if __name__ == "__main__":
    main()
