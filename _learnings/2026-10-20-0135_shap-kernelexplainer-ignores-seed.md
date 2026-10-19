# shap.KernelExplainer ignores seed= and logs its internals at INFO

**Symptom.** `shap.KernelExplainer(f, bg, seed=123)` accepted the argument, yet two runs gave attributions
that differed by up to 0.028 (found by the independent code review, 2026-10-20). Separately, with logging at
INFO, the run log filled with lines like `np.sum(w_aug) = np.float64(8.0)` and `phi = array([...])`, once per
explained point (100 such lines from one task).

**Cause.** In shap 0.54.1 the signature is `KernelExplainer.__init__(self, model, data, feature_names=None,
link='identity', **kwargs)`; `seed` lands in `**kwargs` and is never read. Sampling uses numpy's global
`np.random.choice` / `np.random.permutation`. The internals are logged on the `shap` logger at INFO.

**Fix.**
- `np.random.seed(task_seed)` immediately before each `kernel.shap_values(...)` call.
- `logging.getLogger("shap").setLevel(logging.WARNING)` after configuring logging.
- `shap.explainers.Permutation` does take `seed=`; `Partition` (Owen values) is deterministic.

**Consequences.** Check an explainer's actual signature before trusting a seed argument; silent `**kwargs`
swallow it.
