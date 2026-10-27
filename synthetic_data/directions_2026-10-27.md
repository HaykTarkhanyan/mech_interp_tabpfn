# Directions from the 2026-10-27 session (exploration, not decisions)

Written while you were away. The evidence lives in two pages:

- `benchmarks/report.html`: the benchmarks of the papers in `papers/`, run here.
- `results/frontier/frontier_report.html`: the realism-tractability probe.

Every number below comes from those runs. Open questions continue the session's Q numbering.

## 1. The ground truth is an estimand choice, and rankings flip with it

This is the strongest pattern of the day. It showed up in four of the benchmarks I ran, each time hidden behind
"the ground truth":

| Benchmark | Ground truth used | The alternative | What happens |
|---|---|---|---|
| OpenXAI (LR) | coefficient vector (a slope) | per-point contributions (what SHAP estimates) | slope methods score 1.000 by algebra; KernelSHAP's rank correlation is -0.09 to 0.86 |
| XAI-BENCH | conditional (observational) Shapley | interventional Shapley (what SHAP, KernelSHAP, MAPLE estimate) | at rho > 0 the interventional methods' "error" is the estimand gap (see report) |
| Miro-Nicolau et al. | paper: one-level finite difference | released code: set the feature to 0 | Gradient 0.995 vs DeepLIFT / IG / KernelSHAP 0.62-0.64 under the first (reproduces Table 7); 0.644 vs 0.92-0.95 under the second; random 0.47-0.49 |
| Looking Glass (explainer page) | coefficient magnitudes | contributions | same as OpenXAI |

**Why it matters for the thesis.** A data-fitted benchmark can publish several exact estimands for the same
world: interventional SHAP, a baseline attribution, and, for additive and pairwise targets, the functional
components themselves. It can then report how much an explainer ranking depends on the choice. "Is this ranking
stable across reasonable estimands?" is a question none of these benchmarks asks, and it is cheap for us. The
pilot already computes interventional truth for g and f (DECISIONS #24).

**Open:** Q13: add an estimand-sensitivity table (ranking under 2-3 exact estimands) to the pilot or to the
first paper?

## 2. Data-fitted simulators agree where the data speak and disagree where they extrapolate

RealCause, the WGAN package and Credence all fit LaLonde PSID (Credence then generated no treated units, so only
two worlds remain). Final runs:

| World | True ATE | True ATT |
|---|---|---|
| RealCause, mean of 20 shipped samples | about -13,300 USD | about +1,930 USD |
| WGAN, 100,000 draws | about -4,270 USD | about +1,260 USD |

The ATEs differ by a factor of three. The ATTs differ by 35-50%: closer, but not close. The ATE depends on what
treatment would do to high earners, a region with almost no treated rows, so each generator's extrapolation
decides it. The ATT is better supported by the data, yet the two generators still disagree noticeably.

**Estimator rankings.** By ATT RMSE the two worlds agree at Kendall tau 0.62, against -0.52 for the ATE. The same
estimator is best for the ATT in both worlds (the gradient-boosted T-learner), and the same two are worst. This
rests on 7 estimators in one data set, so treat it as an illustration, not a measurement.

**For the transfer pilot:** builders' worlds and reality may disagree mainly off the support of D. Restricting
explained points and truth backgrounds to the support of the real rows (or reporting agreement on- vs
off-support) would separate "the generator is unrealistic" from "the question has no data-determined answer".

**Open:** Q14: add an on-support / off-support split to the pilot's agreement metric?

## 3. Chance levels are high and rarely stated

| Benchmark | Metric | Chance level |
|---|---|---|
| fb2ts (3 features) | SIM | about 0.52-0.58 for a uniform random attribution, against the paper's 0.60-0.64 for three methods |
| ITErpretability | predictive accuracy | 0.41 with the code's own setting for Twins (16 of 39 features), 0.36 with the paper's rule |
| OpenXAI | FA | about 0.17 for the random control |

Our pilot already carries a null world and a random-orientation null of 0.5 (DECISIONS #24); keep both in every
figure.

## 4. Realism-tractability frontier (explored, `scripts/frontier_probe.py`)

**Question:** how much of a real task can target families with exact SHAP capture? Families: linear, additive
(splines, boosted), pairwise boosted (`HistGradientBoostingRegressor(interaction_cst="pairwise")`), unconstrained
boosting. Ten regression datasets.

**New tool:** exact interventional SHAP for any order-2 model, from coalitions of size at most two
(`pairwise_truth`). It is tested against shap's exact enumeration in `tests/test_truth.py`, and on every fitted
model the script checks that third-order Moebius terms vanish.

**Results** (`results/frontier/frontier.json`, 4 min): held-out R^2 against the real labels, 3-fold.

| Dataset | Linear | Additive (boosted) | Pairwise | Full | Pairwise / full |
|---|---|---|---|---|---|
| california | 0.50 | 0.75 | 0.84 | 0.84 | 0.99 |
| airfoil | 0.51 | 0.54 | 0.90 | 0.93 | 0.97 |
| house_16H | 0.25 | 0.45 | 0.57 | 0.64 | 0.88 |
| wine_quality | 0.29 | 0.30 | 0.41 | 0.44 | 0.92 |
| kin8nm | 0.41 | 0.38 | 0.52 | 0.81 | 0.64 |
| cpu_act, concrete, energy, elevators | 0.60-0.91 | 0.78-0.99 | 0.84-1.00 | 0.88-1.00 | 0.95-1.00 |

- A pairwise target keeps at least 88% of full boosting's R^2 on 9 of 10 data sets. Its exact SHAP costs 14 ms to
  1.4 s per explained point with a 200-row background.
- An additive target can be far off (airfoil 0.58 of full, kin8nm 0.47).
- kin8nm (robot-arm kinematics) is the case where only higher-order targets are realistic.
- Exactness checks: against shap's exact enumeration (d <= 10) the formula matches to 2.7e-13; attributions sum to
  f(x) - E f within 1e-9; third-order Moebius terms are at most 3e-11.
- The pilot's "additive" target (splines) and its forest target bracket the frontier; a pairwise target sits
  close to the realistic end and keeps exact, cheap ground truth.

**Open:**
- Q15: is a pairwise-boosted target (EBM-like, exact SHAP at d^2/2 cost) a candidate target family next to the
  pilot's additive, forest and MLP? Choosing it is your call (DECISIONS #22 lists the families).
- Q16: `interpret`'s EBM is the standard GA2M. Use it instead of constrained sklearn boosting? It would be a new
  dependency.

## 5. Parked (not run, with the reason)

- **Q6** (random-target arm) and **Q8** (agreement by generator vs target): both need the Tier 0 pilot run
  (~1.3 h), which is above this session's compute cap.
- **Q9** (observational estimand via ARF conditional sampling): needs `forge(evidence=...)` in the arfpy fork.
  Section 1 makes it more valuable: conditional SHAP is the estimand XAI-BENCH uses.
- **Q3** (add ITErpretability, Looking Glass, Credence, RealCause, Athey et al. and Chen et al. to the review):
  re-running `lit_references.py` takes ~7 min of network calls.
