# Decisions

Newest first. Never delete a superseded entry.

---

## #24 - Transfer pilot v2: redesigned for noise, fairness and exact ground truth

**Date:** 2026-10-20
**Status:** active (supersedes the design part of #22; the question in #22 is unchanged)

**Decision:** Replace `transfer_pilot.py` by `transfer_pilot_v2.py`:
- every world's ranking is the mean over K tasks, the oracle is K fresh reality tasks, and reality's reference
  uses 30 tasks;
- primary metric, fixed before the full run: model-level agreement on reality's *resolved* explainer pairs
  (paired 95% interval excludes zero); tau_b, regret and top-1 (only when reality's best is resolved) are
  secondary;
- a generic null world that never sees D (Gaussian, rho 0.5, nonlinear additive target, as in XAI-BENCH);
- two reality families (forest; MLP, which no builder can choose) to measure projection;
- four generators (independent, Gaussian, row bootstrap on a train/explain split of D, ARF from the user's
  fork) x three targets (additive with quantile knots, forest, MLP);
- nine explainers instead of six SHAP variants (adds Permutation SHAP, Owen values, LIME, occlusion;
  drops the background-size variant);
- 1,000-row truth backgrounds;
- every feature value in every world snapped to a 2^-12 grid.

**Why:** The independent review (2026-10-20) and the smoke run showed:
- two independent 100-row truths differed by 0.09 relative error, as large as the explainer gaps;
- reality's top pair was unresolved (t about -0.3);
- the oracle compared one task with a 5-task mean, so builders beat it;
- the bootstrap world trained on more rows than D held;
- KernelSHAP ignored its seed (shap 0.54.1 swallows `seed=`; now seeded through numpy's global state).

The grid fixes the ground truth. shap's interventional TreeSHAP missed exact enumeration by up to 2.2e-2 on
California rows, and jitter only reduced that to the same order. The cause: split thresholds that fall
between nearly equal values are not exact in float32. On the grid (values and midpoints exact in float32),
TreeSHAP matched exact enumeration to 3.3e-8 on 8 models and to 2.9e-9 with a 1,000-row background, at
TreeSHAP speed. The truth code still raises above 1e-6. Additive truths use the identity
phi_j = mean_b f(b, b_j := x_j) - mean_b f(b); `tests/test_truth.py` checks every shortcut against
enumeration or a closed form.

**Alternatives rejected:**
- exact enumeration with 1,000 rows: 43 s (GBR) to 96 s (forest) per truth;
- averaging chunks of 100 rows: a single chunk still failed without the grid;
- six SHAP variants: low resolution, and orderings that hold in any world inflate agreement;
- random ranking as the only null: too easy.

**What would change this:**
- reality resolving too few pairs, which needs more reality tasks;
- the timing runs making the full design unaffordable;
- the null matching builders, which would mean realism adds nothing.

---

## #23 - Packages for the transfer pilot; the ARF implementation is left open

**Date:** 2026-10-19
**Status:** ARF part resolved 2026-10-20: the user asked for a fork. `arfpy` is installed from
github.com/HaykTarkhanyan/arfpy at commit da71011 (branch `feature/random-state`, stacked on
`fix/leaf-coverage-and-oob`: leaf coverage renormalised (upstream issue #4), `forde(oob=True)` fixed, `random_state`
added; 27 tests pass). `lime==0.2.0.1` was added after all: it runs on numpy 2.5 (0.05 s per explanation).

**Decision:** Run the transfer pilot (#22) with `shap==0.54.1`, `scikit-learn==1.9.1`, `numpy==2.5.4`,
`pandas==3.0.6`, `scipy==1.18.1` and `plotly==7.1.0` (report), in `synthetic_data/`'s uv env. Leave the
ARF generator out of the first pilot rather than pick its implementation.

**Why:** shap is the reference implementation of every explainer in the pilot (TreeSHAP interventional and
path-dependent, Saabas, KernelSHAP) and of exact interventional SHAP; plotly is the repo default for reports.
`arfpy==0.1.1` (last release 2023-09-22) is broken on current libraries: it calls `np.in1d` (removed in
numpy 2.4) and indexes a pandas Series by position (removed in pandas 3); measured 2026-10-19, both crash
`arf()`/`forge()` on the diabetes data. It was removed from the env again. `lime==0.2.0.1` (2020) was not added.

**Alternatives rejected (for now, not decided):** patching arfpy locally; downgrading numpy/pandas for it;
the maintained R `arf` package (R 4.4.1 is installed) via Rscript; writing ARF on top of scikit-learn. Which
ARF implementation the thesis uses is a fundamental choice and is the user's.

**What would change this:** the user's choice of ARF implementation; a new arfpy release that works on
numpy 2.4+ and pandas 3.

---

## #22 - First study of the thesis direction: does a data-fitted benchmark reproduce explainer rankings?

**Date:** 2026-10-19
**Status:** active

**Decision:** Start the synthetic-data thesis (#19) with a transfer-validity study rather than by building the
benchmark: in a nested known-truth design (Advani et al. 2019), "reality" keeps real feature rows and a known
true function fitted to the real target; a benchmark builder sees one sample, builds a world from it (feature
generator x fitted target) and ranks explainers there; agreement with reality's ranking (Kendall tau, top-1
match, regret, against a random pick and an oracle that only differs by task noise) is the outcome. Pilot:
`synthetic_data/scripts/transfer_pilot.py`, California housing, three generators (independent columns,
Gaussian with the sample covariance as in XAI-BENCH section 5.3, row bootstrap) x three target families
(linear, additive splines, random forest), six SHAP-family explainers, interventional SHAP as the estimand,
computed for ground truth by exact enumeration through `model.predict` with an additivity check. shap's own
interventional TreeSHAP is not used for ground truth: on California rows it lost additivity by up to 0.12
(shap 0.54.1, measured 2026-10-19; exact on Gaussian data, and shap's own tree traversal matches sklearn's
predict), so it is only logged as a diagnostic.

**Why:** It tests the premise the pitch rests on (realistic data gives rankings that transfer) and the open
problem Karim (2026) names (a data-computable check of difficulty and ranking transfer); no XAI work found
runs this test (review corpus of 3,067 records plus five web searches on 2026-10-19; the nearest are
XAI-BENCH section 5.3, Islam et al. 2025 and arXiv 2502.07153, none of which compares a benchmark's ranking
with a known reference ranking). Either result is publishable: transfer that improves with realism motivates
the benchmark; no transfer is an Advani-style negative result. It is CPU-sized, and it measures projection
directly by crossing target families.

**Alternatives rejected:** building the full benchmark first (its main claim would stay untested); the
proxy-metric meta-evaluation (review implication 9) first (it is a natural second study on the same harness,
but does not test the premise); a realism-only comparison of rankings across generators (subsumed: without a
reference ranking one cannot say which generator is right).

**Smoke-run evidence (2026-10-19, 1 replicate, `results/transfer_pilot/smoke.json`):** the pipeline runs end
to end with exact ground truth (44 s per task, 2 threads). Single-task rankings are noise: the oracle (a fresh
sample of reality itself) agreed with reality's 5-task ranking at Kendall tau 0.20 only, because the gaps
between explainers' mean errors (0.29 to 0.46) are comparable to their task-to-task sd (0.05 to 0.13). The
full pilot therefore needs several task replicates per world, which multiplies the cost.

**What would change this:** if the pilot shows rankings identical everywhere (no signal) or no better than
the oracle's noise floor anywhere, the explainer set or estimand must change before scaling up; if "reality"
built from a fitted forest proves too close to the builders' forest target (projection by construction), add
realities from other function families before drawing conclusions.

---

## #21 - Scope and sources of the robotics-foundation-model explainability review

**Date:** 2026-10-19
**Status:** active

**Decision:** Review explanation and mechanistic interpretability of robot foundation models (VLAs and large
visuomotor/diffusion policies) as the core, with world models, embodied LLM planners, driving VLAs and RL-agent
mech interp as adjacent background. Retrieve with the arXiv API (16 field-scoped boolean queries), Crossref,
a web sweep for workshop and venue-only work, and backward chaining from the downloaded PDFs' reference lists.
Keep it self-contained in `robotics_foundation_models_explainability/` with its own scripts and uv env.

**Why:** The literature is almost entirely arXiv-native and 2025-2026: one probe query
(`abs:"vision-language-action" AND interpretability terms`) returned 205 hits with about 20 directly on-topic
papers, none older than February 2025. Much of the newest work exists only as workshop papers (ICML 2026 MI
workshop, NeurIPS 2026 RoboPAD), which no database indexes yet, hence the web sweep. Semantic Scholar's keyless
pool answered one request and then 429'd for 7+ minutes straight, and OpenAlex had $0.017 of its $0.10 keyless
daily budget left after the synthetic-data review (#20) and lists 0 references for the main seed paper
2509.00328, so neither could do citation chaining today; reference lists read from the PDFs replace them for
backward chaining.

**Alternatives rejected:**
- Reusing `synthetic_data/scripts/` with new themes: its search is OpenAlex-led and its themes, seeds and
  paths are hard-coded, the same reason #20 gave for not reusing the August scripts.
- Strict mech-interp-only scope: the user asked for "explaining / mech interp", and failure prediction from
  internals, self-explanation faithfulness and attribution work are where much of the robotics activity is.
- Waiting for Semantic Scholar or OpenAlex: forward chaining is the one thing lost; it is recorded as pending.

**What would change this:** OpenAlex MCP login or a Semantic Scholar key (run forward chaining with
`lit_search.py --reuse-keyword`), or the review turning into a publication-grade systematic review (needs a
second screener and forward citations).

---

## #20 - Scope and pipeline of the synthetic-data-for-XAI scoping review

**Date:** 2026-10-19
**Status:** active

**Decision:** Review data built fully or partly from real datasets for scoring explanation methods,
together with four neighbouring literatures the pitch depends on (realistic simulation in causal
inference/econometrics/epidemiology, tabular generators and their fidelity, symbolic regression,
explanation estimands). Run it with a new self-contained pipeline in `synthetic_data/scripts/`, add
OpenAlex forward-citation chaining from 14 seed papers, and use arXiv in place of DBLP.

**Why:** The pitch's novelty question cannot be answered from XAI benchmarks alone: the closest
precedents (plasmode, RealCause, Credence, WGAN Monte Carlo) and the sharpest critiques (Advani et al.
2019, Karim 2026) are in other fields. Chaining was decisive: 165 of 888 screened-in records and
889 of 3,067 unique records came only from it. DBLP answered HTTP 202 with an anti-bot page
(checked 2026-10-19), and arXiv's API worked. A separate pipeline keeps `synthetic_data/`
independent of the TabPFN material (#19); the August scripts hard-code the mech-interp topic in
every file (term lists, title lists, paths).

**Alternatives rejected:**
- Generalising the August `scripts/` with a topic config - touches five working scripts and ties
  the new direction to the old repo layout.
- XAI-only scope - would miss the validity literature that most constrains the pitch's claims.
- Paying for an OpenAlex key or Semantic Scholar key - not needed at this scale; the keyless OpenAlex
  budget covered the run (~$0.08 of $0.10).

**What would change this:** a systematic review for publication (needs a second screener and
Semantic Scholar or Google Scholar citation data), or the thesis direction changing back (#19).

---

## #19 - The thesis is now more likely to be the semi-synthetic XAI ground-truth benchmark, not TabPFN interpretability

**Date:** 2026-10-19
**Status:** active (leaning, not final)

**Decision:** Treat the semi-synthetic benchmark in `synthetic_data/` (real feature distribution
learned by an adversarial random forest, closed-form target from symbolic regression, exact
reference explanations) as the likely thesis direction. Work in `synthetic_data/` stands on its
own and is not linked to the TabPFN material (`fe_mech_pipeline/`, `paper_explainers/`,
`meeting_prep/`).

**Why:** The user's call after reading the pitch (`synthetic_data/xai_benchmark_pitch.pdf`). The
pitch addresses a gap the existing ground-truth benchmarks leave open: they either use toy
feature distributions (XAI-BENCH, OpenXAI) or have no ground truth at all (LATEC).

**Alternatives rejected:** Continuing the TabPFN feature-effects mechanism thesis (#9-#18). That
work is kept as-is, not deleted.

**What would change this:** the literature review (`synthetic_data/`) turning up a published
benchmark that already does real-covariate generation plus a known target for XAI, or the
supervisor preferring the TabPFN direction.

---

## #18 - fe_mech_pipeline pilot scale: 256-row context, 128 reference rows, 25 null sets per type, CPU only

**Date:** 2026-10-19
**Status:** active

**Decision:** Run the pilot (`fe_mech_pipeline/run_pilot.py`) with one context of 256 rows,
128 reference rows (8 per (X1, X2) cell), ALE grids of 8 bins (first order) and 4 x 4 cells
(second order), 25 random null sets per component type, 16 no-ablation floor passes per
placement (3 query, 3 grid, 3 context, 7 joint), and the zero-ablation contrast on the
same-pair placement only. About 290 forward passes in total.

**Why:** Measured on this laptop with 4 torch threads. One forward pass at 512 context + 2304
query rows took 6.2-7.6 s, so the first design (512 context, 256 reference rows, ~330 passes)
would have run about 40 minutes. At 256 context + 864 query rows a pass took 0.76-0.89 s on an
idle machine and 3-8 s while other processes loaded the CPU. The chosen size is about 5 minutes
at ~1 s per pass. Larger nulls, several contexts and the nanoTabPFN prior ablation go to
`fe_mech_pipeline/RUNPOD.md`.

**Alternatives rejected:**
- 512 context rows and 256 reference rows - about 40 minutes on this machine.
- 64 reference rows to afford a bigger null - 4 rows per second-order cell makes the noise floor
  swamp everything, so the null would be large but uninformative.
- Dropping the different-pair placement - it is the H2 test.

**What would change this:** a GPU run (RunPod), which removes the time constraint; or the floor
being larger than every Delta_B in the pilot, which would mean more reference rows come first.

---

## #17 - Pre-register H1 and H2 with mechanical decision rules before the pilot runs

**Date:** 2026-10-19 (registered 14:00 local time, before `run_pilot.py` was run for the first time)
**Status:** active

**Decision:** The pilot tests two hypotheses with fixed rules, coded in
`fe_mech_pipeline/delta_b.py` and stated in `fe_mech_pipeline/README.md`.
- A candidate set (all heads of one attention type in layer L, or all MLP units of layer L)
  **passes** iff Delta_target > floor_target, its null percentile is >= 95, and selectivity S > 1.
- **H1** (the X1*X2 interaction is carried mainly by feature attention, not item attention), per
  placement: if no attention set passes, inconclusive; otherwise the passing attention set with
  the highest S decides: feature attention means supported, item attention means contradicted.
- **H2** (it matters whether X1 and X2 share a feature pair, columns 0,1, or not, columns 0,2):
  no attention set passes in either placement means inconclusive; identical sets of passing
  attention components means not supported; different sets where at least one flipped set
  changes Delta_target by more than the larger floor means supported; otherwise inconclusive.

Seen before registering: the ALE validation output and `check_model.py` (including the
prediction-level effect of ablating layer 0). No Delta_B, null or floor value had been computed.

**Why:** Review item 5 asks for the feature-attention hypothesis as a pre-registered prediction,
and review section 8 item 5 asks for the pair placement as a controlled condition. Fixing the
thresholds before seeing data prevents choosing them afterwards.

**Alternatives rejected:**
- Descriptive reporting only - nothing could contradict it.
- p-values from a parametric test - with one context and 25 null sets the null percentile, at a
  resolution of 4 points, is the honest unit; a p-value would suggest more precision than exists.

**What would change this:** the main study needs its own registration over several contexts;
these verdicts are for one context draw and are pilot evidence only.

---

## #16 - Noise floor = largest Delta_B among joint no-ablation resamples; selectivity is floored by it

**Date:** 2026-10-19
**Status:** active

**Decision:** Recompute B with no ablation under four resampling sources: a new reference
(query) sample, a jittered ALE grid (interior edges moved by up to 15% of a bin), a new context
draw, and all three at once ("joint"). Report each source; use floor = the largest joint value
(7 resamples) in the pass rule. Selectivity S = Delta_target / max(Delta_nontarget,
floor_nontarget), target = second-order ALE of (X1, X2), non-target = first-order ALE of X2.

**Why:** Review fixes (ii) and (iii). The joint source is the conservative one: it includes the
context variability that RQ3 is about. With 7 values a 95th percentile is just an
interpolation between the top two, so the maximum is used and called that. Flooring the
denominator stops a near-zero non-target change from producing an arbitrarily large S.

**Alternatives rejected:**
- Floor at fixed context only - ignores that g_D changes with D, which is the RQ3 question.
- A paired bootstrap of Delta_B itself - tighter, because Delta_B compares the same rows on the
  same grid, while the floor measures how far B moves between independent estimates. Recorded
  as a known conservatism of the floor; the pilot follows the review's definition.

**What would change this:** many more resamples (GPU), which allow a real 95th percentile; or the
floor exceeding every Delta_B, which would make the paired bootstrap the better yardstick.

---

## #15 - Null: random component sets of matched size and type, drawn across all layers, shared by both placements

**Date:** 2026-10-19
**Status:** active

**Decision:** For a layer set of 6 feature-attention heads, the null is 6 feature-attention heads
drawn without replacement from all 72; the same for item attention; for an MLP layer (768 hidden
units) the null is 768 units drawn from all 9,216. 25 sets per type, the same sets for both
placements. Null percentile = 100 x (number below + 0.5 x ties) / 25.

**Why:** Review fix (i). Matching the type keeps the feature-vs-item comparison fair. Drawing
across layers asks whether this layer is special compared with the same amount of the same
machinery anywhere in the network. Ablating MLPs at the unit level is what makes a size-matched
MLP null possible at all; a list of 12 whole MLPs is just the scan itself.

**Alternatives rejected:**
- The other whole layers as the null - only 12 values, identical to the scan.
- Mixed-type random sets - would compare MLP damage with attention damage.

**What would change this:** if every concentrated layer set beats every scattered random set,
the null is measuring concentration rather than relevance, and a within-layer null (random
subsets of k heads of layer L) is needed.

---

## #14 - Distance: relative integrated L2 on a fixed grid

**Date:** 2026-10-19
**Status:** active

**Decision:** d(B, B') = sqrt(mean over cells of (c(B') - c(B))^2) / sqrt(mean over cells of
c(B)^2), where c() averages the centred ALE over each bin (midpoint) or cell (four corners).
Cells carry uniform weights because they have equal population mass by construction (iid
Uniform(-2, 2) features, equal-width bins: 8 for first order, 4 x 4 for second order).

**Why:** Integrating over the data distribution weights regions by how much data they hold.
Centring removes the arbitrary ALE constant. Dividing by the unablated norm puts the
interaction and the main effect on one scale, which the selectivity ratio needs.

**Alternatives rejected:**
- Sup norm - one noisy cell would dominate.
- Change in a scalar interaction strength - misses shape changes such as a sign flip in one region.
- Absolute L2 - not comparable between behaviors of different size.

**What would change this:** a behavior whose unablated norm is near zero. That is the flat-main-
effect task's main effects; the relative distance is undefined there, so the pilot does not run
selectivity on that task, and the main study needs an absolute or target-normalised distance for it.

---

## #13 - Mean ablation of per-head slices before the output projection and of MLP hidden units; zero ablation as contrast

**Date:** 2026-10-19
**Status:** active

**Decision:** Attention heads (both types) are ablated by a forward pre-hook on the block's
`out_projection`, replacing the head's 32-dim slice of the concatenated head outputs. MLPs are
ablated at the GELU output (768 hidden units). Mean ablation is primary: the replacement is the
unit's mean in a clean pass over the context plus the unmodified reference rows, computed per
token column and separately for context and query rows. Zero ablation is a contrast on the
same-pair layer scan.

**Why:** `out_projection` has no bias, so the head slice is exactly the head's contribution to
the residual stream. The second MLP linear has no bias either, so mean-ablating all 768 units
equals replacing the MLP output by its mean, and the unit level allows a size-matched null.
Context and query rows get separate means because the target token holds the label only in
context rows. Mean beats zero as the primary because zero pushes activations off-distribution:
in `check_model.py`, zeroing the layer-0 feature-attention heads moved log-odds by 0.62 on
average, mean-ablating them by 0.01. Note: TabPFN v2 query rows use multi-query item attention
(all six query heads read key/value head 0), so for query rows an item "head" is a query head
reading a shared key/value head. Verified: inactive hooks and self-patching are bitwise no-ops.

**Alternatives rejected:**
- Resample ablation (activation from another row) - needs its own corruption design and more passes.
- Patching from a counterfactual context (gamma = 0) - a good next step, needs matched corrupted contexts.
- One mean pooled over context and query rows - would inject or remove label information.

**What would change this:** mean and zero ablation disagreeing on which sets matter, which would
mean the answer depends on the replacement value (proposal section 6) and resample ablation or
patching should become primary.

---

## #12 - ALE in numpy: Apley-Zhu first order, ALEPlot second order, population grid, stratified reference rows

**Date:** 2026-10-19
**Status:** active

**Decision:** `fe_mech_pipeline/ale.py` implements first-order ALE as in Apley and Zhu (2020)
and the second-order (pure interaction) ALE as in ALEPlot's `ALE.2D` (Apley's R reference code).
Bin edges come from the population (equal width on [-2, 2]), not the sample. Reference rows are
stratified, 8 per (X1, X2) cell, X3 and N1 left random. Empty bins or cells raise an error.

**Why:** No new IML library without approval, and the estimator is short enough to validate
fully: `validate_ale.py` recovers linear and additive functions exactly (relative error under
1e-12) and x1*x2 to 0.6% with 50,000 rows. A population grid does not move between resamples,
so B values from different samples sit on the same grid. Stratification is proportional
allocation (cells have equal mass), so the estimand is unchanged and no cell is ever empty. X3
is deliberately not stratified: it drives the interaction region, and stratifying it would hide
the main source of estimator noise and understate the floor.

**Alternatives rejected:**
- Sample-quantile grids - move with every resample, forcing interpolation everywhere.
- Partial dependence or the H-statistic as the primary - extrapolate off the data (proposal section 2).
- Filling empty cells from neighbours as ALEPlot does - a silent fill.
- alibi or PyALE - a new dependency the user has not approved.

**What would change this:** correlated features (planned task variants), where equal-width bins
are no longer equal-mass; the grid should then use quantiles of the correlated distribution.

---

## #11 - Behavior on the log-odds scale

**Date:** 2026-10-19
**Status:** active

**Decision:** g_D(x) = logit_1 - logit_0 from the raw model output at temperature 1.

**Why:** The tasks define the interaction on the logit scale, so the true second-order ALE is
exactly 0 for the gamma = 0 control and 0.75 x1 x2 for gamma = 1.5. On the probability scale the
sigmoid creates an interaction even in the additive control, so "the interaction" would not be
zero where the data have none. The softmax temperature (0.9 by default) is a constant factor and
cancels in relative distances.

**Alternatives rejected:**
- Probability scale - sigmoid-made interaction; saturation compresses differences where the
  model is confident.
- log P(class 1) - also not additive in the logit.

**What would change this:** a regression or multi-class task, where log-odds is not the natural scale.

---

## #10 - fe_mech_pipeline: wrapper preprocessing off, one estimator, fixed column order

**Date:** 2026-10-19
**Status:** active

**Decision:** `n_estimators=1`, `FEATURE_SHIFT_METHOD=None`, `CLASS_SHIFT_METHOD=None`,
`FINGERPRINT_FEATURE=False`, `POLYNOMIAL_FEATURES="no"`, `OUTLIER_REMOVAL_STD=None`, preprocessing
`"none"` with categorical `"numeric"`, no SVD and no appended originals, temperature 1. Still
active, inside the model: constant-feature removal, NaN indicators, mean imputation, z-scoring
fitted on context rows, feature-group scaling, and the per-group positional embedding. After
verification, forward passes call the model module directly.

**Why:** The v2 defaults would quantile-transform, append the original columns, add SVD
components and a fingerprint column, and shuffle columns, so column j would no longer be token j
and the pair placement (H2) would be meaningless. `check_model.py` verified that the wrapper
hands the model exactly our columns (max difference 6e-8, float32 rounding) and that its raw
logits equal the direct forward pass (max difference 9e-6).

**Alternatives rejected:**
- The default 8-member ensemble - shuffles columns, breaks the feature-to-token mapping.
- Calling the wrapper for every pass - slower, same output.

**What would change this:** studying the model as deployed with its default preprocessing, which
is a different g_D.

---

## #9 - Model: the original TabPFN v2 classifier checkpoint, loaded by file name

**Date:** 2026-10-19
**Status:** active

**Decision:** `tabpfn-v2-classifier.ckpt` from `Prior-Labs/TabPFN-v2-clf`, loaded through tabpfn
8.2.0 with `TabPFNClassifier.create_default_for_version(ModelVersion.V2, model_path=...)`.
29.0 MB, sha256 `f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f`, 7,244,554
parameters, 12 blocks, d_model 192, 6 feature-attention and 6 item-attention heads, features
embedded in pairs.

**Why:** The user's choice: results must be comparable with Bilos et al. and Balef et al., who
both study the 12-layer v2 (v2.5 and v2.6 are excluded by their licenses; review section 8
item 6). Which file: Balef's released code vendors tabpfn 2.0.9, whose v2 classifier default is
`tabpfn-v2-classifier.ckpt`, and the parameter count matches their Table A.2 exactly. Bilos et al.
do not name the file. The name must be explicit because tabpfn 8.2.0 defaults to v3, and its v2
default is a different file, `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`.

**Alternatives rejected:**
- The package default (v3) - a different architecture from the one the prior work studied.
- tabpfn 8.2.0's v2 default (the `finetuned` file) - not the file Balef loaded; an extra
  fine-tuning step would also confound any "the prior shapes the mechanism" argument.
- nanoTabPFN - planned for the prior-ablation extension, which needs a GPU (`RUNPOD.md`).

**What would change this:** evidence that Bilos et al. used the fine-tuned file, or the thesis
moving its main study to nanoTabPFN.

---

## #8 - Rename the repo `mech_interp` → `mech_interp_tabpfn`

**Date:** 2026-09-11
**Status:** active

**Decision:** Rename the repository, locally and on GitHub, to `mech_interp_tabpfn`, and
declare TabPFN / tabular foundation models the repo's forward scope. The broad
language-model mech interp review stays as completed work and as the field map the TabPFN
work builds on; it is no longer the organising subject.

**Why:** Requested by the repo owner, and it matches what the repo has actually become.
Of the work now in it, the TabPFN side is the bulk: a 39-paper verified collection, 17
research directions, a thesis-proposal review, a benchmark and a toy demo. Decision #1
explicitly named "a stated research direction for this repo" as the thing that would
change its scope. The rename is that statement, so recording it closes the loop rather
than leaving #1's trigger condition dangling and unremarked.

**Alternatives rejected:**
- Keep `mech_interp` - accurate but uninformative; the owner has several repos and the
  generic name does not distinguish this one, which is now specifically about TabPFN.
- `tabpfn_interp` or similar, dropping `mech_interp` - loses the connection to the
  language-model review that is still a third of the repo and still cited by the TabPFN
  research directions.
- Split into two repos (review / TabPFN work) - the TabPFN ideas lean on the review's
  central finding (that evaluation is mech interp's binding constraint), so splitting
  would separate an argument from its evidence.

**Spelling note:** the rename was requested as `mech_interp_tabpfm`. Confirmed as a typo
and corrected to `tabpfn` before anything was renamed; the model is TabPFN.

**Cost accepted:** the GitHub URL changes. GitHub serves a permanent redirect from the old
path, so existing clones keep working and old links resolve, but any hardcoded reference to
`HaykTarkhanyan/mech_interp` is now relying on that redirect.

**What would change this:** the repo broadening again beyond tabular models - if, say, the
mech interp methodology work outgrows the TabPFN application, the honest name would go back
to something general.

**Deliberately not changed:** decision #1's body still reads "the repo is new and named
`mech_interp`". That was true when #1 was made and the rule here is not to edit history.

---

## #7 - Commit the TabPFN PDFs into git history

**Date:** 2026-08-06
**Status:** active

**Decision:** Track all 30 TabPFN PDFs (125.5 MB, 703 pages) in git rather than gitignoring
them and relying on `scripts/download_tabpfn_papers.py` to rebuild the collection.

**Why:** Explicit call by the repo owner, made with the trade-off stated. The supporting
argument: the papers are the working corpus for reading, not a build artefact, and several
are arXiv preprints that get silently revised - two in this very collection were retitled
between versions (see `papers/tabpfn/README.md`). Committing the bytes pins the exact
versions read, which a re-download would not reproduce.

**Alternatives rejected:**
- Gitignore the PDFs, commit `manifest.json` + README + script - keeps the repo at ~20 MB and
  the collection is rebuildable in one command, but does not pin preprint *versions*, and
  arXiv preprints are mutable.
- Git LFS - solves the size problem properly, but adds a dependency and a server-side quota
  for a repo that is otherwise plain.

**Cost accepted:** repo grows ~20 MB → ~145 MB. Clone and fetch get slower. The blobs are
permanent - removing them later requires `git filter-repo` or BFG, which rewrites history and
invalidates every existing clone.

**What would change this:** the collection growing to a scale where clone time actually hurts
(the excluded ~40 domain-application papers would roughly double it), or the repo gaining
collaborators for whom a 145 MB clone is a real cost. Git LFS would be the migration path.

---

## #6 - Version `.claude/skills` with the repository

**Date:** 2026-08-06
**Status:** active

**Decision:** Track all 64 skills under `.claude/skills/` in git (675 files, ~13 MB) rather
than gitignoring them.

**Why:** Explicit call by the repo owner. The practical argument for it: the review pipeline
was built while following the `literature-review` skill, and several decisions here
(DECISIONS #3, #4) are deliberate departures from what that skill instructs. Those departures
only make sense to a future reader if the instructions they depart from are in the repo at the
same commit. Skills also drift upstream, so pinning the versions actually used is what makes
the work reproducible.

**Alternatives rejected:**
- Gitignore `.claude/` - keeps the repo small, but detaches the review from the methodology it
  was produced under and makes DECISIONS #3/#4 read as unmotivated.
- Track only `literature-review/` - the review touched several skills, and cherry-picking
  invites the tree and the record diverging later.

**Cost accepted:** ~13 MB and 675 files of the diff surface are third-party content, so
`git log --stat` and file-count metrics on this repo are dominated by skills rather than by
project code.

**What would change this:** the skills tree growing large enough to slow clone/checkout, or
the skills being moved to their own versioned package that this repo could pin by reference.

---

## #5 - Ship the review without the supplementary retrieval stage, and disclose it

**Date:** 2026-08-06
**Status:** active

**Decision:** Publish the review with `scripts/non_essential/seed_and_enrich.py` unrun,
recording the omission in `data/processed/prisma_counts.json`, in §4.4 of the review, and
in `scripts/non_essential/README.md` - rather than blocking on it or quietly dropping it.

**Why:** OpenAlex began returning HTTP 429 to this network partway through the session and
did not recover across ~50 minutes of exponential backoff (8s→256s, six attempts, repeated
by an outer retry loop). The stage would have delivered two things: explicit lookups of 46
canonical titles, and abstracts for the 49% of deduplicated records that carry none. Both
were substantially covered by other means - foundational works were resolved from arxiv.org
and publisher URLs instead, and the screening logic was corrected so that unambiguous domain
terms no longer require abstract text to corroborate them (this alone recovered 99 records,
243→342).

The alternative to shipping was waiting on a third-party rate limit with no known reset time.
The cost of shipping is a real but bounded loss of recall, which is stated as a limitation
rather than papered over. `screen_papers.py` requires an explicit
`--allow-missing-supplementary` flag to run in this mode, so the degradation cannot happen
silently or by default.

**Alternatives rejected:**
- Block until OpenAlex recovers - unbounded wait, no reset signal.
- Drop the stage from the repo - loses a working component and hides a real methodological gap.
- Loosen screening thresholds to compensate for missing abstracts - would trade a known,
  disclosed recall gap for an unknown, undisclosed precision loss.

**What would change this:** OpenAlex becoming reachable. The stage is resumable via
`data/processed/seed_cache.json`; re-running it and then `screen_papers.py` *without* the flag
would raise recall and let the limitation be removed from §4.4.

---

## #4 - Generate the PRISMA diagram from the counts file, not with an image model

**Date:** 2026-08-05
**Status:** active, revisited 2026-08-06

**Decision:** Render the PRISMA flow diagram from `data/processed/prisma_counts.json` via
`scripts/plot_prisma.py`, rather than generating it with the `scientific-schematics`
skill as the `literature-review` skill instructs.

**Revision 2026-08-06:** originally rendered as Mermaid in the markdown source. Switched to
a matplotlib figure because Mermaid does not render through the pandoc/xelatex PDF build -
it appeared in the PDF as a raw code block. A PNG renders in GitHub markdown *and* the PDF.
The original rationale is unchanged and in fact strengthened: the figure is still generated
deterministically from the counts file, and the generating script is still diffable in review.
Regenerating the figure after a re-run of the pipeline requires no manual editing.

**Why:** `literature-review/SKILL.md:32` declares AI-generated figures MANDATORY, and
routes them through `scientific-schematics`, which requires `OPENROUTER_API_KEY` (a paid
key we do not have). Beyond the cost: a PRISMA diagram is a box-and-arrow chart whose
entire content is exact record counts. An image model has no mechanism to guarantee the
number it draws in a box matches the number in the data, and a wrong count in a PRISMA
diagram invalidates the review's reproducibility claim. Mermaid renders the counts as
text taken directly from `data/raw/search_counts.json`.

**Alternatives rejected:**
- `scientific-schematics` via OpenRouter - needs a paid key; risks hallucinated counts.
- matplotlib-drawn flowchart - works, but is more code than Mermaid for the same output
  and is not diffable in review.

**What would change this:** A figure that is genuinely pictorial rather than structural
(e.g. a conceptual illustration of superposition geometry) would justify an image model,
since there are no exact numbers to corrupt.

---

## #3 - Verify arXiv-native papers against OpenAlex, not Crossref

**Date:** 2026-08-05
**Status:** active

**Decision:** Split citation verification by DOI registrar. Journal/conference DOIs are
verified against Crossref via the skill's `verify_citations.py`; arXiv-native DOIs
(`10.48550/arXiv.*`) are verified against OpenAlex and the DOI handle system.

**Why:** arXiv DOIs are registered with DataCite, not Crossref. Measured directly:

```
curl https://api.crossref.org/works/10.48550/arXiv.2209.10652  ->  404
```

The skill's `verify_citations.py` checks existence via `doi.org/api/handles` (which is
registrar-agnostic and passes) but pulls metadata only from Crossref, so an arXiv paper
would be reported "verified" with an empty metadata record. That is a false pass, and in
a field where most primary sources are arXiv preprints it would cover most of the
bibliography.

**Alternatives rejected:**
- Crossref only - silently produces empty metadata for most mech interp papers.
- DataCite API - would work, but OpenAlex already returns the same metadata plus
  citation counts in one call, and is already a search backend here.

**What would change this:** If the review's scope shifted to published-venue-only
sources, Crossref alone would cover it.

---

## #2 - Search OpenAlex + Crossref + DBLP; arXiv and Semantic Scholar are unavailable

**Date:** 2026-08-05
**Status:** active

**Decision:** Use OpenAlex (primary), Crossref, and DBLP as the three literature
databases. Probe arXiv and Semantic Scholar on every run and record the result rather
than silently omitting them.

**Why:** Both preferred sources are blocked from this network. Measured 2026-08-05:

| Endpoint | Result |
|---|---|
| `export.arxiv.org/api/query` (trivial query) | HTTP 429, then read timeout |
| same, browser User-Agent | HTTP 503 |
| `api.semanticscholar.org` keyless, 3 tries 5s apart | HTTP 429, 429, 429 |
| `api.openalex.org`, 3 tries | HTTP 200, 200, 200 |
| `api.crossref.org` | HTTP 200 |
| `dblp.org/search/publ/api` | HTTP 200 |

Losing the arXiv API does **not** lose arXiv coverage: OpenAlex indexes arXiv preprints
and exposes their `10.48550/arXiv.*` DOIs, which is how preprints enter this corpus.
DBLP was added as the third database specifically because it carries authoritative CS
venue strings (NeurIPS/ICML/ICLR/ACL) that OpenAlex often leaves blank for preprints.

**Alternatives rejected:**
- Wait out the arXiv 429 - the block persisted across ~20 minutes and several UA/scheme
  variants, so it looks like IP-level throttling, not a burst limit.
- Pay for a Semantic Scholar key - not warranted for one review; S2's main advantage
  (citation counts) is also served by OpenAlex.
- Proceed with two databases - the review methodology calls for a minimum of three.

**What would change this:** arXiv or S2 becoming reachable (the probe reports this on
every run), or obtaining an `S2_API_KEY`, which would add citation-graph traversal for
forward/backward citation chaining.

---

## #1 - Scope the review to mechanistic interpretability of transformer language models

**Date:** 2026-08-05
**Status:** active

**Decision:** Cover mechanistic interpretability of transformer-based language models:
representation structure (superposition, linear features, SAEs), circuit-level analysis,
causal intervention methodology, and the field's evaluation/critique literature. Exclude
general XAI (saliency maps, LIME/SHAP-style attribution), vision-only interpretability,
and interpretability of non-transformer architectures except where they are the direct
object of a mech interp study.

**Why:** The repo is new and named `mech_interp` with no prior code to constrain scope,
so a broad orientation review of the field is the useful default. The exclusions are
what separate mechanistic interpretability from XAI generally: mech interp aims at
reverse-engineering internal computation into human-understandable algorithms, whereas
post-hoc attribution answers a different question (which inputs mattered) and has its
own large, largely disjoint literature.

**Alternatives rejected:**
- Full XAI survey - an order of magnitude larger and mostly not what "mech interp" names.
- Single-subarea deep dive (e.g. SAEs only) - premature before the field map exists.

**What would change this:** A stated research direction for this repo. A deep dive on
one thread (SAEs, circuit discovery, evaluation) would then be worth doing separately.
