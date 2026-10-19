# One task per world gives explainer rankings that are mostly noise; compare K-task means

**Symptom.** In the transfer pilot's smoke run (`synthetic_data/results/transfer_pilot/smoke.json`), the oracle,
a fresh sample of reality itself, agreed with reality's 5-task ranking at Kendall tau 0.20, while most builder
worlds scored higher (0.60 to 0.73 at model level).

**Cause.** Explainer errors sit close together and vary a lot between tasks (a task = one trained model and
30 to 50 explained points):

```
reality, data level, mean relative error (sd over 5 tasks)
TreeSHAP-path 0.290 (0.063)   Interventional bg50 0.307 (0.066)   KernelSHAP 0.328 (0.051)
Saabas 0.340 (0.071)          Interventional bg10 0.399 (0.130)   Mean-baseline 0.458 (0.055)
paired difference of the top two over 5 tasks: mean -0.013, sd 0.10, t about -0.3
```

The oracle compared one noisy task with a 5-task mean, so a world with more stable rankings beat reality.

**Fix (pilot v2, DECISIONS.md #24).** Every world's ranking is the mean over K tasks, the oracle uses the same K,
reality's reference uses 30 tasks, and the primary metric counts only pairs whose order reality resolves
(paired 95% interval excludes zero). Explainers that differ only in a setting (background size) were dropped,
because their order holds in any world and inflates agreement.

**Consequences.** Any benchmark that ranks explainers from one trained model and a few dozen points (XAI-BENCH
trains on 1,000 rows and explains 100) may be reporting noise; this is itself a possible finding.
