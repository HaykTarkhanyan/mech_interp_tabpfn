# arfpy on PyPI (0.1.1, 2023) is broken on current numpy and pandas; use the fork pinned by commit

**Symptom.** With numpy 2.5.4 and pandas 3.0.6:

```
arf.arf(x=X, ...)   AttributeError: module 'numpy' has no attribute 'in1d'
model.forge(n=500)  KeyError: 0      (self.factor_cols[j], positional Series access)
```

**Cause.** `np.in1d` was removed in numpy 2.4 and positional `Series[int]` access in pandas 3. Upstream
`main` (bips-hb/arfpy, pushed 2026-06-01) already fixes both, but no release followed.

**Also found while fixing (upstream issues, all reproduced):**
- Issue #4, "probabilities do not sum to 1": `forde()` zeroes and drops single-observation leaves, so per-tree
  coverage sums fall below 1 and `forge()` fails in `np.random.choice`. Reproduced with `min_node_size=1` on the
  diabetes data (smallest per-tree sum 0.898). The maintainer had asked for a reproducible example.
- `forde(oob=True)` always failed (`self.x` does not exist), and used sklearn's private
  `_generate_unsampled_indices`, whose signature gained `sample_weight` in scikit-learn 1.9. The public
  `estimators_samples_` (sklearn >= 1.4) works in 1.8 and 1.9.
- Nothing was seedable: fit, shuffles and `forge` used numpy's global state.

**Fix.** Fork at github.com/HaykTarkhanyan/arfpy (clone `Desktop/arfpy`): branch `fix/leaf-coverage-and-oob`
and stacked `feature/random-state`, 27 tests pass. `synthetic_data/pyproject.toml` pins
`arfpy @ git+https://github.com/HaykTarkhanyan/arfpy@da710119eece1924406bab868c2c6fdee55222f4`.

**Consequences.** When upstream releases, switch back to a PyPI pin. Upstream's CONTRIBUTING asks for an
issue before a feature PR (e.g. likelihood and conditional sampling, issue #5).
