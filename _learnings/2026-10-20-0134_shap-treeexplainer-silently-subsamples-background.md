# shap.TreeExplainer silently cuts a raw background above 100 rows down to 100

**Symptom.** Passing a 200-row numpy array as `data=` to `shap.TreeExplainer(..., feature_perturbation="interventional")`
only logs a warning:

```
WARNING  Background dataset has 200 samples but max_samples=100. Subsampling to 100 samples for SHAP value
computation. To use all samples, set max_samples=200 when initializing the masker.
```

The SHAP values are then computed against a random 100-row subset, so a "200-row ground truth" is not one.

**Fix.** Pass a masker sized to the background:
`shap.maskers.Independent(bg, max_samples=len(bg))` as `data=` (TreeExplainer) or as the masker (Exact,
Permutation). Checked: with the masker, `explainer.data.shape` is `(200, 4)` and values sum to the prediction.

**Consequences.** The background size that defines an interventional SHAP target must be set explicitly
everywhere. Two independent 100-row truths differed by 0.09 relative error on California rows, as large as
the explainer gaps being ranked, which is why the pilot now uses 1,000-row truths.
