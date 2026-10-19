# shap's interventional TreeSHAP is wrong on data with nearly equal values; a float32-exact grid fixes it

**Symptom.** On California-housing rows, `shap.TreeExplainer(model, data=masker, feature_perturbation="interventional")`
lost additivity and disagreed with exact enumeration (`shap.explainers.Exact` through `model.predict`):

```
gap per fitted model, |TreeSHAP - exact|, 30 rows x 100 background rows (shap 0.54.1, sklearn 1.9.1):
raw rows        up to 1.0e-1 on single rows; a constant -4.6e-4 offset on every row of one task
jitter 1e-6     1e-03 6e-03 3e-03 1e-04 2e-02 4e-04 2e-03 4e-03   (worst 2.25e-02)
grid 2^-12      1e-08 3e-08 1e-08 3e-08 2e-08 2e-08 1e-08 2e-08   (worst 3.31e-08)
grid, 1,000 background rows: additivity gap 2.6e-08, vs exact 2.9e-09
```

On Gaussian data it was exact. shap's own `model.predict` traversal agreed with sklearn on all 500 test rows.

**Cause.** sklearn sets a split threshold to the float64 midpoint of two float32 feature values. When two
values are nearly equal, that threshold is not representable in float32, and shap's interventional path
compares in a precision where a point, or a background row, at that value goes the other way. One such
background row shifts every explained row by the same amount, which is the constant offset seen above.
Jitter only moves points across the boundary; ties were not the full story.

**Fix.** Snap every feature value to a grid that is exact in float32 together with its midpoints:
`np.round(x / 2**-12) * 2**-12` (standardized units, max |x| 119, so 19 mantissa bits). Then TreeSHAP is
exact at TreeSHAP speed. `synthetic_data/scripts/transfer_pilot_v2.py` does this in every world and raises
if additivity fails by more than 1e-6.

**Consequences.** Never use interventional TreeSHAP as ground truth on raw real data; check additivity and
compare a few rows with exact enumeration. Worth reporting upstream to shap with this reproduction.
