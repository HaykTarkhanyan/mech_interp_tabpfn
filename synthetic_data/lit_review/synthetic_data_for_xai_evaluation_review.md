---
title: "Synthetic Data Built from Real Data for Evaluating Explanation Methods"
subtitle: "A scoping review"
date: "2026-10-19"
---

- **Search date:** 2026-10-19
- **Databases:** OpenAlex, Crossref, arXiv, plus forward-citation chaining from 14 seed papers (OpenAlex)
- **Records:** 3,582 retrieved, 3,067 after deduplication, 888 passed the automated screen
- **References cited:** 81 (81/81 verified against registries, 0 title mismatches)
- **Review type:** Scoping review, single reviewer (an AI assistant, Claude; see Section 4.5)
- **Scope:** Data that is built, fully or partly, from real datasets so that explanation methods can be scored against a known answer, plus the neighbouring literatures such a benchmark depends on (DECISIONS.md #20)

---

## Abstract

Explanation methods for tabular models are hard to evaluate because the correct explanation of a real dataset is unknown. Simulations fix that by construction but use feature distributions nobody believes in. This review asks what lies in between: data generated from real datasets, with a target whose relationship to the features is known exactly. It synthesises 81 verified sources found by a keyword search of three databases, by forward-citation chaining from 14 seed papers, and by reading the five most central papers in full.

Six findings structure the area. First, "ground truth" for an explanation comes from three sources (a constructed data-generating process, a signal planted in real data, or human annotation) and at two levels (what the data-generating function uses, and what a fitted model uses). The two levels coincide only under restrictive conditions, and the best-known tabular benchmark buys them by making all features independent. Second, every fully synthetic tabular benchmark found uses an invented feature distribution, while the failure modes that matter most for explainers (suppressor variables, correlation, extrapolation) are driven by exactly the dependence structure those distributions lack. Third, only two XAI works derive tabular data from real datasets, and both are thin: a Gaussian fitted to real covariances with a nearest-neighbour target, on two datasets, and a cluster-perturbation scheme on one dataset. Fourth, causal inference, econometrics and pharmacoepidemiology have used "real covariates, simulated outcome" designs for over a decade and have produced a sharp critical literature: such benchmarks can pick the least-biased estimator worse than random, favour methods whose form matches the simulated outcome, and can refute a method but cannot certify it. The XAI benchmark literature reviewed here does not engage with it. Fifth, tabular generators now offer tractable conditional densities (adversarial random forests, probabilistic circuits), which is what exact explanation targets need, but generator fidelity is measured with metrics that ignore conditional structure. Sixth, symbolic regression can supply readable closed-form targets fitted to real data, but its solutions are unstable under small data changes.

The gap is therefore not the idea of realistic semi-synthetic data, which is old in neighbouring fields, but its combination with exact, estimand-specific explanation targets at both the data and the model level, together with the validity checks the neighbouring fields learned to require.

---

## 1. Introduction

### 1.1 The evaluation problem

Feature attribution and feature effect methods (SHAP, LIME, permutation importance, partial dependence, accumulated local effects and their many variants) are routinely used to explain tabular models, and increasingly to draw conclusions about the data itself. A review of 181 business-research studies found that treating post hoc explanations as evidence about data relationships is widespread, and that on simulated data with known structure SHAP and LIME "can fail to be aligned even when their predictive accuracy is high" (Wang et al., 2024). Whether an explanation is correct can only be checked against something known. A systematic review of more than 300 XAI papers found that one in three evaluated exclusively with anecdotal evidence (Nauta et al., 2023).

Without a known answer, evaluation falls back on proxy metrics, and these disagree. LATEC compared 17 explanation methods under 20 metrics across 7,560 configurations and found that metric choice changes method rankings enough to induce selection bias (Klein et al., 2024). Practitioners face the same disagreement between explainers on a single model (Krishna et al., 2024). Proxy metrics themselves need validation, which is again hard without ground truth (Hedström et al., 2023; Miró-Nicolau et al., 2024a).

### 1.2 The middle ground this review is about

Two standard escapes exist. Simulations specify the data-generating process, so the answer is known, but their feature distributions are invented. Real data has realistic dependence, but its true explanation is unknowable. XAI-BENCH states the trade-off plainly: "there is a trade-off between data realism and ground truth availability" (Liu et al., 2021).

This review is about the space between: data whose feature distribution is learned from, or resampled from, real data, combined with a target whose relationship to the features is known. The motivating case is a benchmark proposal (`synthetic_data/xai_benchmark_pitch.pdf`) that fits an adversarial random forest to real covariates, fits a closed-form target by symbolic regression, adds calibrated noise, and publishes exact reference explanations against both the target and each fitted model. Section 4.4 maps the findings onto that design. A ChatGPT conversation saved in the same folder served as a starting point for the XAI benchmarks. Its claims were checked against the papers, and where they differ (for example, OpenXAI's metric count changes between versions) the papers are followed.

### 1.3 Questions

- **RQ1.** How do existing XAI benchmarks obtain ground truth, and at which level (data or model)?
- **RQ2.** Which works derive evaluation data from real tabular datasets, and how?
- **RQ3.** What have neighbouring fields learned about the validity of simulations built from real data?
- **RQ4.** Which tabular generators support exact explanation targets, and how is their fidelity measured?
- **RQ5.** What must be specified for "exact ground truth" to be well defined, and where are the gaps?

---

## 2. Methods

### 2.1 Search strategy

Eighteen thematic queries were run on 2026-10-19, capped at 50 records per query per database. Themes covered ground-truth XAI benchmarks, synthetic tabular XAI data, semi-synthetic XAI data, ground-truth Shapley values, faithfulness metrics, saliency ground truth, plasmode simulation, semi-synthetic causal benchmarks, generative Monte Carlo design, empirical Monte Carlo, tabular generative models, synthetic data fidelity, symbolic regression benchmarks, model-versus-data explanations, conditional feature importance, feature-effect estimation error, simulation study design, and graph explainer ground truth. Full query strings are in `lit_review/data/raw/search_counts.json`; the pipeline is `synthetic_data/scripts/lit_search.py`.

| Database | Status on 2026-10-19 | Records |
|---|---|---|
| OpenAlex | Used (keyless; shared daily budget of $0.10, about $0.08 used) | 900 |
| Crossref | Used | 900 |
| arXiv API | Used (it returned HTTP 429 in an earlier review in this repository; it worked this time) | 744 |
| DBLP | Unavailable: HTTP 202 with an anti-bot challenge page instead of JSON | 0 |
| Semantic Scholar | Unavailable: HTTP 429 on the keyless tier | 0 |

### 2.2 Citation chaining

Keyword search misses work that does not use the field's vocabulary, so every work citing one of 14 seed papers was retrieved through OpenAlex's `cites:` filter. Any benchmark that pairs a learned real-data generator with a known target would very likely cite XAI-BENCH, OpenXAI or the adversarial random forest paper, which makes this the most direct novelty check available.

| Seed | OpenAlex records | Citing works |
|---|---|---|
| XAI-BENCH (Liu et al., 2021) | 1 (arXiv record only) | 19 |
| OpenXAI (Agarwal et al., 2022) | 1 | 112 |
| Adversarial random forests (Watson et al., 2023) | **not indexed** | 0 |
| WGAN Monte Carlo design (Athey et al., 2024) | 2 | 78 |
| RealCause (Neal et al., 2020) | 1 | 18 |
| Mostly harmless simulations (Advani et al., 2019) | 1 (arXiv record only) | 0 |
| Geoscience attribution benchmark (Mamalakis et al., 2022b) | 1 | 101 |
| Local explanations on ground truth (Guidotti, 2020) | 1 | 97 |
| BAM (Yang and Kim, 2019) | 1 | 84 |
| True to the model or the data (Chen et al., 2020) | 1 (via DOI fallback) | 74 |
| PD and PFI vs the data-generating process (Molnar et al., 2023b) | 1 | 150 |
| SRBench (La Cava et al., 2021) | 1 | 137 |
| Plasmode simulation (Franklin et al., 2013) | 1 | 142 |
| SR interpretability benchmark (Aldeia and de França, 2022) | 1 | 26 |

Two coverage failures matter. The adversarial random forest paper (AISTATS 2023) has no OpenAlex record at all, so its citers were not chained. XAI-BENCH has only its arXiv record, with 19 citing works, which is certainly an undercount for a NeurIPS Datasets and Benchmarks paper. For the Advani et al. paper, the published journal version is a separate OpenAlex record with 16 citing works that was not chained because the free daily budget was nearly used up. Semantic Scholar, which links conference and preprint versions better, was rate-limited throughout.

After a first draft, the five works most central to the review's question were read in full, figures and appendices included: the two real-data XAI precedents (Liu et al., 2021; Islam et al., 2025), the generator the motivating proposal builds on (Watson et al., 2023), and the two sharpest critiques of data-anchored simulation (Advani et al., 2019; Karim, 2026). Their reference lists served as backward citation chaining, which added one work the search had missed (Schuler et al., 2017), and the full reading corrected several statements that had been based on abstracts.

### 2.3 Screening

Duplicates were merged field by field on DOI, OpenAlex ID and normalised title, keeping the most complete value of each field. The topic spans several communities, so the automated screen is a set of six rules, one per strand of the review, and a record passes if any rule fires: explanation vocabulary with evaluation and synthetic-data vocabulary; explanation vocabulary with "ground truth"; semi-synthetic or plasmode vocabulary with comparison vocabulary; tabular generation with quality vocabulary; symbolic regression with benchmark vocabulary; and data-generating-process or conditional-sampling vocabulary with explanation vocabulary. An earlier review in this repository found that screening on abstracts silently rejects records whose database omits them, so abstracts were backfilled from OpenAlex by DOI first (325 filled), and the 663 records still without one were screened by a looser title-only rule and flagged. Application papers (XAI for a clinical or financial task) were excluded on title unless they were benchmarks.

Passing records were ranked by a relevance score (4 points per rule fired plus 1 per matched vocabulary term, ties broken by citation count), and the 700 highest-ranked were screened manually by title, then by abstract, and the closest candidates in full text. Inclusion required one of: a method for obtaining ground truth for explanations; a design that builds evaluation data from real data, in any field; evidence about the validity of such designs; a tabular generator or fidelity measure relevant to building one; a symbolic-regression benchmark or stability result; or a definition of the estimand an explanation method targets.

### 2.4 PRISMA flow

Figure 1 shows the flow from retrieval to inclusion.

![PRISMA flow diagram for this review. All numbers are read from the pipeline's count files by `scripts/lit_prisma.py`.](figures/prisma_flow.png)

### 2.5 Citation verification

All 81 references were resolved from a source and title-checked at build time (`scripts/lit_references.py`), then verified a second time by the repository's registry-routed verifier (`scripts/verify_references.py`): arXiv for preprints, Crossref for journal and proceedings DOIs. **Result: 81/81 verified, 0 failures, 0 title mismatches** (`lit_review/verification_report.json`). Two corpus records had author lists broken by "Family, Given" names separated by commas; both were re-resolved from their registries rather than repaired by hand. Where a registry held only the preprint of a paper published since, the published version is cited: its DOI where it has one, otherwise its venue and year, each checked by web search.

### 2.6 A retrieval finding

The keyword databases again behaved as nearly disjoint samples: of 3,067 unique records, 2,053 were found by exactly one keyword database, 122 by two and 3 by all three. Citation chaining contributed 889 records that no keyword query found, and 165 of the 888 records that passed screening came only from chaining. Only 54 of the 81 cited works came through screening. Of the other 27, nine were retrieved but missed by it: six failed the automated screen (among them two works central to Section 3.5, Advani et al. and RealCause) and three passed it but ranked below the top-700 cut. Eighteen were not retrieved by the search at all. All 27 were added by targeted lookup or backward chaining, among the not-retrieved ones the conference record of OpenXAI, Credence, LATEC, and two papers that also served as hand-picked seeds: the adversarial random forest paper and the "true to the model or true to the data" paper. These works do not use the vocabulary of "ground-truth explanations", and several sit in venues that bibliographic databases index poorly. A keyword-only search of this topic would have missed much of its core.

---

## 3. Results

### 3.1 What "ground truth" means: three sources, two levels

The benchmarks reviewed obtain ground truth in three ways. **Constructed** ground truth specifies the data-generating process, so the relevant features and their effects are known by design (Liu et al., 2021; Agarwal et al., 2022; Barr et al., 2020; Guidotti, 2020; Mamalakis et al., 2022b). **Injected** ground truth plants a signal with known effect into otherwise real inputs and checks whether explainers find it (Yang and Kim, 2019; Zhou et al., 2022; Adebayo et al., 2022; Bastings et al., 2022). **Annotated** ground truth relies on human judgement and is outside this review's scope. A recent review of synthetic ground-truth methods for images names the constructed family SAIG, "Synthetic Artificial Intelligence Ground truth", and finds little consensus on how to use it (Miró-Nicolau et al., 2026b).

Orthogonal to the source is the level. **Data-level** ground truth describes what the data-generating function uses. **Model-level** ground truth describes what a trained model uses. They differ because an accurate model can rely on proxies. OpenXAI spells this out: if labels depend on features A and B, and these are correlated with C and D, a model may predict from C and D, and an explainer that correctly reports this would be scored as wrong against the data's ground truth (Agarwal et al., 2022).

The literature answers this mismatch in three ways, and the choice determines how realistic the data can be.

1. **Restrict the data until the levels coincide.** OpenXAI's SynthGauss generator samples well-separated Gaussian clusters with identity covariance, so "all the features are independent", and proves that under independence, separation and a unique labelling function the most accurate model adheres to the ground truth (Agarwal et al., 2022). Realistic dependence is given up by design.
2. **Build the model by hand.** AttributionLab sets network weights manually because a model fitted to designed data has "no guarantee" of using all and only the designed features (Zhang et al., 2023). Procedurally generated models with known mechanisms (Lee et al., 2025), synthetic convolutional models with derived ground truth (Brandt et al., 2023) and transparent models with perfect-fidelity explanations (Miró-Nicolau et al., 2024a) follow the same logic. The data then matters little, because the model is not learned from it.
3. **Compute the model-level target exactly for a trained model.** If the data distribution is known, explanation targets that need conditional expectations (conditional Shapley values, faithfulness, remove-and-retrain) can be computed for any trained model. This is XAI-BENCH's route: synthetic distributions "allow the efficient computation of conditional expected values that are needed to evaluate ground-truth Shapley values" (Liu et al., 2021). Intervention-based ground truth, which perturbs inputs within the training distribution and reads the model's response, is a related answer (Miró-Nicolau et al., 2026a; Hesse et al., 2023).

Only the third route is compatible with realistic, dependent features, because it does not need data-level and model-level ground truth to coincide. It needs the data distribution to be known and its conditionals to be computable.

### 3.2 Fully synthetic tabular ground truth: correct but invented

The tabular benchmarks found draw features from a small set of invented distributions. XAI-BENCH implements multivariate Gaussian, Gaussian-mixture and multinomial features, with linear, piecewise-constant, nonlinear-additive and piecewise-linear labelling functions, and lets the user dial feature correlation (Liu et al., 2021). OpenXAI uses independent Gaussian clusters (Agarwal et al., 2022). Barr et al. (2020) specify marginals from `scipy.stats` and dependence through a copula, with labels from a `sympy` symbolic expression thresholded into classes. Guidotti (2020) generates synthetic transparent classifiers whose decision reasons are known. Additively separable functions, for which the attribution ground truth is known a priori, are used for regression in geoscience (Mamalakis et al., 2022a, 2022b). XAI-Units pairs datasets and models with known mechanisms such as interactions, cancellations and discontinuities (Lee et al., 2025). The newest framework builds a local tabular ground truth on a discretised input space, from a functional benchmark, with interventions of one discrete level (Miró-Nicolau et al., 2026a). The Weber group's earlier work argued for data that represents ground-truth explanations in the first place (Amiri et al., 2020).

These benchmarks have produced real findings. Most explainers fail to separate important features from suppressor variables in linear data with well-defined dependencies (Wilming et al., 2022), and false positive attributions persist in non-linear image benchmarks (Clark et al., 2023). Local additive explainers can fail to explain even linear additive models (Akhavan Rahnama et al., 2023). In XAI-BENCH, explainers become less faithful as feature correlation increases, and LIME's faithfulness drops by as much as 90% when features are almost perfectly correlated (Liu et al., 2021). On simulated data with known structure, misalignment between explanations and the true data-generating process is driven by feature correlation and by the Rashomon effect (Wang et al., 2024). Graph and vision benchmarks follow the same constructed pattern: ShapeGGen generates graphs with known explanations and is designed to "mimic the data generated by various real-world applications" (Agarwal et al., 2023), FunnyBirds composes synthetic birds from parts whose removal defines importance (Hesse et al., 2023), and SMERF controls the complexity of a model's reasoning in a synthetic image benchmark and reveals significant limitations of saliency methods (Kim et al., 2022).

The pattern across these results is striking. The failures that matter most are caused by dependence between features: suppressors, correlation and proxies. Yet dependence in these benchmarks is either removed (OpenXAI) or set by a single correlation parameter on a Gaussian or copula (XAI-BENCH, Barr et al.). Mixed types, heavy tails, non-linear dependence and the specific structure of real tables are absent. The benchmarks can show that dependence breaks explainers, but not how badly it breaks them on the kind of dependence real data has.

### 3.3 Ground truth injected into real inputs

Outside tabular data, the standard way to keep inputs realistic is to plant a known signal. BAM pairs a crafted dataset with models trained under known relative feature importance (Yang and Kim, 2019). Zhou et al. (2022) propose a dataset modification procedure that induces ground truth and find deficiencies in saliency maps, rationales and attention. Adebayo et al. (2022) build semi-synthetic datasets with pre-specified spurious artifacts and find that post hoc explanations are ineffective at detecting them when the artifact is unknown at test time, especially for non-visible artifacts such as a background blur. For text, partially synthetic data with planted shortcuts shows that "some of the most popular method configurations provide poor results even for simplest shortcuts" (Bastings et al., 2022). CLEVR-XAI derives pixel-level relevance from the task's generative program and describes its testbed as selective, controlled and realistic (Arras et al., 2022). The image-focused SAIG review adds an observation that transfers directly: all the generalisable methods it surveyed use real images as input, while those using simple or complex synthetic figures are tied to their pre-designed datasets (Miró-Nicolau et al., 2026b).

Injected ground truth is partial. It tells whether the planted part is found, not what the full attribution vector should be, because the relevance of the untouched real inputs stays unknown. For tabular data, where one wants exact attribution values for every feature, injection alone is not enough, which is why the tabular literature has favoured constructed ground truth.

### 3.4 Evaluation data derived from real tabular data: the two precedents

Only two XAI works found build tabular evaluation data from real datasets, plus one generator that resembles the pitch's design without being fitted to real data.

**XAI-BENCH, Section 5.3, "Simulating real datasets".** For the wine-quality dataset (11 continuous features), features are normalised, their empirical covariance is used to generate multivariate Gaussian features, and the simulated target "is labeled by a k-nearest neighbor model based on real datapoints" (Liu et al., 2021). Fidelity is reported as the mean Jensen-Shannon divergence of marginals (0.20 for wine, 0.17 for forest fires) and by training models on simulated and real data and comparing their explanations on a held-out real test set. The headline ground-truth metric, GT-Shapley, is the Pearson correlation between an explainer's attributions and the true Shapley values of the trained model, so it ignores scale, and the synthetic experiments train on 1,000 points.

**Privacy-preserving ground-truth data with additive CBR.** Islam et al. (2025) cluster a real regression dataset (flight delays), pick an equal number of seed instances from each cluster (described as the instances closest to the centroids in their Section 3.2 but as random in Section 3.3.1), perturb the continuous features with zero-mean Gaussian noise while keeping categorical features unchanged, and draw target values at random within each cluster's original range. The "ground truth" is the cluster behaviour. Because target values are drawn at random within each cluster's range, the features carry no information about the target beyond cluster membership, so there is no data-generating function to explain. The reference attribution (AddCBR) comes from a model instead: a three-nearest-neighbour regression weighted by the XGBoost model's global feature importances, rescaled so that the contributions sum to the prediction. Fidelity is measured by Jensen-Shannon divergence (average 0.105, which the authors compare favourably with XAI-BENCH's 0.20, although the two were measured on different datasets). It is one dataset and regression only.

**Copulas and symbolic targets, not fitted.** Barr et al. (2020) pair a copula with a symbolic target expression, the closest structural relative of the pitch, but both are written by the user rather than fitted to data. Their paper states the pitch's motivation precisely: methods that mimic the statistics of other datasets "are incapable of providing ground truth for explanations since they lack the complete data generation process that imposes a functional dependence between X and y".

| | XAI-BENCH 5.3 | Islam et al. 2025 | Barr et al. 2020 | Pitch |
|---|---|---|---|---|
| Features | Gaussian with real covariance | real seeds plus Gaussian noise; categoricals kept | copula, user-chosen marginals | ARF fitted to real covariates |
| Target | kNN fitted to real data | random within cluster range | user-written symbolic expression | symbolic regression fitted to real data, calibrated noise |
| Explanation ground truth | Gaussian conditionals known exactly; expectations by sampling | additive CBR attributions | global coefficients; local only in special cases | exact or error-bounded targets against f* and each fitted model |
| Fidelity reported | marginal JSD; cross-model explanation MSE | marginal JSD | none | distributional, task and explanation fidelity |
| Scale | 2 datasets, regression | 1 dataset, regression | illustrative cases | many datasets planned |

The reverse direction, using explanations to evaluate synthetic data, is newer and relevant to fidelity. A detector trained to tell real from synthetic rows, explained with permutation importance, partial dependence, Shapley values and counterfactuals, shows why synthetic data is distinguishable (Kapar et al., 2025), and the cosine distance between global SHAP vectors of models trained on real and on synthetic data captures shifts that KL divergence and train-on-synthetic-test-on-real accuracy miss (Yu et al., 2025).

### 3.5 Lessons from realistic simulation in causal inference, econometrics and epidemiology

Designs that keep real covariates and simulate the outcome are long established elsewhere, and the critical literature around them is the most important input this review found for an XAI benchmark of this kind.

**The designs.** Plasmode simulation resamples features from a real dataset and generates the outcome from a known, user-chosen model; it was used to evaluate pharmacoepidemiologic methods in large healthcare databases (Franklin et al., 2013), and a recent review notes that it is "widely claimed that plasmodes come very close to reality with some aspects of the 'truth' known" while no explicit guidelines existed (Schreck et al., 2024). Semi-synthetic causal benchmarks such as IHDP and the ACIC 2016 competition data became standard for treatment-effect estimators (Hill, 2011; Dorie et al., 2019). Empirical Monte Carlo studies base their data-generating processes on actual data; one compared eleven causal machine learning estimators over 24 such processes (Knaus et al., 2020). Synth-validation goes one step further and selects a causal method for a given dataset by its estimation error on generative distributions fitted to that dataset; in its own simulations, this lowered the expected error compared with always using any single method (Schuler et al., 2017). It is the closest analogue in causal inference to what the pitch proposes for explanations. Generative models replaced resampling next: Wasserstein GANs fitted to real economic datasets to limit the researcher's freedom in choosing a design (Athey et al., 2024), RealCause, which uses flexible generative models to yield ground-truth causal effects while staying realistic and evaluates over 1,500 estimators (Neal et al., 2020), Credence, which generates data "anchored at the empirical distribution" with user-specified effects and confounding (Parikh et al., 2022), and generative designs for multilevel data in the social sciences (Suk et al., 2026). The general framework for designing simulation studies (aims, data-generating mechanisms, estimands, methods, performance measures) applies to all of them (Morris et al., 2019).

**The critiques.** Each of these is directly transferable to explanation methods.

- *Rankings can be wrong.* Advani et al. (2019) study two proposals for empirically motivated Monte Carlo studies, show theoretically that neither is likely to be informative except under restrictive conditions, and find on a real setting with known estimator performance that both are "worse than random at selecting estimators which minimise absolute bias". They did better at selecting for mean squared error, but a simple bootstrap was at least as good and often better, and the authors advise using a range of estimators and comparing them for robustness. The mechanism matters for explanations: the simulations reproduced the estimators' variances but not their biases, and biases come from the mismatch between an estimator's assumptions and the true process, which a simulation fitted under its own assumptions does not carry over. Explainer errors that come from violated assumptions (independence, additivity) are of this second kind. Athey et al. (2024) themselves find that no estimator outperforms the others in all three of their settings.
- *The benchmark can favour some methods.* The two most-used semi-synthetic causal benchmarks have characteristics that "favor some algorithms over others", which is "rarely acknowledged" (Curth and van der Schaar, 2021).
- *Refute, not certify.* Karim (2026) argues that "a plasmode can refute a universal claim about an estimator but cannot certify one": refuting needs only that the simulated data belong to the class a guarantee covers, while certifying needs the benchmark's difficulty to carry over to the target population, which cannot be verified. The paper names *projection*, where the injected outcome model "hands a matching estimator the right answer by construction" and "can reverse estimator rankings even when overlap is intact", and names synth-validation as precisely the practice projection diagnoses, since a fitted generator is a working model that some estimator matches. It also shows that fidelity bounds the gap between the simulated and the real estimand but says nothing about the ranking, so "our generator is very realistic" is not by itself a reason to trust the ordering a benchmark reports.
- *Design details create artefacts.* One of two common plasmode frameworks can make some estimators "incorrectly appear overly biased" (Shaw et al., 2026). Whether plasmode or parametric simulation is preferable depends on the number of features, on how far parametric assumptions deviate from the truth, and on the resampling strategy (Stolte et al., 2024).
- *The truth must actually be computed.* True estimand values are often not available analytically, and Monte Carlo integration with controlled error is the practical route (Naimi et al., 2025). Combining synthetic experiments with real-data negative controls is one way to check a simulation's conclusions (Tian et al., 2018).

None of the XAI benchmarks reviewed cites this literature, and none reports how stable its method rankings are across alternative outcome functions.

### 3.6 Generators of realistic tabular data and how their fidelity is measured

For exact explanation targets, a generator has to do more than sample realistic rows: its conditional distributions must be computable. Adversarial random forests are provably consistent and provide "smooth (un)conditional densities", performing comparably to probabilistic circuits and deep models on tabular benchmarks while running about two orders of magnitude faster (Watson et al., 2023). The density is a coverage-weighted mixture over tree leaves, and within each leaf the features are modelled as independent (truncated Gaussian marginals for continuous features), so conditioning on some features reduces to reweighting leaves. That is what makes exact conditional targets feasible. It also means the generator carries its own independence assumption at the scale of a leaf, and the consistency guarantee is asymptotic only. Probabilistic circuits also "deliver tractable probabilistic generation and inference", and a recent study finds them competitive with diffusion models at a fraction of the cost (Scassola et al., 2026). Conditional GANs (Xu et al., 2019) and diffusion models (Kotelnikov et al., 2023) generate high-fidelity samples but do not give tractable conditionals. Model-X knockoffs are another route to conditional sampling (Candès et al., 2018). A survey covers the wider landscape of deep models for tabular data (Borisov et al., 2022).

Fidelity measurement is unsettled. Sample-level precision, recall and authenticity metrics address fidelity, diversity and copying (Alaa et al., 2022). Using synthetic data as if it were real leads to models and analyses that do not generalise to real data (van Breugel et al., 2023). The apparent saturation of progress in tabular generation is "largely due to the use of inadequate metrics" (Scassola et al., 2026), and structural fidelity (alignment of causal structure between real and synthetic data) is proposed as a missing evaluation dimension (Jiang et al., 2025). Explanation-aware fidelity measures are emerging (Kapar et al., 2025; Yu et al., 2025). None of these metrics targets what an explanation benchmark depends on most: the fidelity of the conditional distributions that conditional Shapley values, conditional feature importance and accumulated local effects integrate over.

Generators are already used *inside* explanation methods. ARF-based conditional feature importance (cARFi) uses ARFs to create on-manifold feature values (Blesch et al., 2025), knockoff imputation makes SHAP and SAGE harder to fool (Blesch et al., 2023b), and sequential knockoffs enable conditional feature importance for mixed data (Blesch et al., 2023a; Watson and Wright, 2021). This matters for benchmark design: an explainer that uses the same generator as the benchmark's ground truth would be evaluated against its own internal model.

### 3.7 Symbolic regression as a source of closed-form targets

Symbolic regression returns analytic expressions and is benchmarked on both real-world data without known form and ground-truth physics equations. SRBench assessed 14 symbolic regression methods and 7 machine learning methods on 252 problems and found that the best methods for real-world regression combine genetic algorithms with parameter estimation and/or semantic search (La Cava et al., 2021). Its successor adds domain-expert interpretation, because expression size is only one of many factors that experts use to judge interpretability (de França et al., 2024). The Feynman equations underpin AI Feynman (Udrescu and Tegmark, 2020) and were rebuilt with realistic sampling ranges as 120 SRSD datasets (Matsubara et al., 2024). PySR is a widely used open tool (Cranmer, 2023). Symbolic regression models also make good subjects for explainer benchmarks: with 100 physics equations as ground truth, Partial Effects and SHAP were the most robust explainers, while Integrated Gradients was unstable with tree-based models (Aldeia and de França, 2022).

The limitation that matters for a benchmark is stability. "Small changes in the training data might cause highly dissimilar solution[s]", which degrades interpretability, the method's key selling point (Kammerer et al., 2024). A symbolic-regression target fitted to real data is therefore one draw from a set of plausible closed forms, not the closed form of the data.

### 3.8 Estimands: what the exact answer is an answer to

"Exact ground truth" is defined only relative to an estimand, and explanation methods target different ones. The choice between interventional and observational conditional expectations in Shapley values is "true to the model" versus "true to the data", and is application dependent (Chen et al., 2020). Partial dependence and permutation feature importance can be formalised as estimators of ground-truth estimands rooted in the data-generating process; their estimates deviate through statistical bias, model variance and Monte Carlo error (Molnar et al., 2023b). Interpretation methods can be designed as descriptors of the joint distribution of the data rather than only of the model (Freiesleben et al., 2024). Permutation-based importance extrapolates outside the data (Hooker et al., 2021); on-manifold Shapley values (Frye et al., 2021), conditional subgroups (Molnar et al., 2023a), conditional predictive impact (Watson and Wright, 2021) and the marginal-versus-conditional distinction (Blesch et al., 2023a) are responses to that.

Two lines of work already measure explanation error against exact targets on known distributions. Kernel SHAP was extended to dependent features and shown to give "more accurate approximations to the true Shapley values" on linear and non-linear models with varying dependence (Aas et al., 2021). A large comparison of conditional Shapley estimators recommends parametric methods when the data distribution can be specified almost correctly, and generative or regression-based methods otherwise (Olsen et al., 2024). For global feature effects, a mean-squared-error decomposition separates model bias, estimation bias, model variance and estimation variance, and shows that ALE is particularly sensitive to sample size (Heiß et al., 2026). All of these studies rely on parametric, mostly Gaussian, data-generating processes, the realism problem of Section 3.2 again.

### 3.9 Proxy metrics and the meta-evaluation problem

Where ground truth is absent, quality estimators disagree: competing estimators of the same property "frequently present conflicting rankings" (Hedström et al., 2023), fidelity metrics lack verification and disagree (Miró-Nicolau et al., 2024a, 2024b), and faithfulness evaluation varies across metrics, modalities and models (Li et al., 2023; Klein et al., 2024). Ground-truth benchmarks offer a second use here: on data where the correct explanation is known, one can measure which proxy metrics track actual explanation error. CLEVR-XAI already reported results from its ground-truth evaluation "sometimes contradicting with conclusions from previous comparative studies" (Arras et al., 2022).

---

## 4. Discussion

### 4.1 Where the literature agrees

Three points are uncontested. Explanations must be evaluated against something known, and proxy metrics alone cannot do it (Nauta et al., 2023; Klein et al., 2024). Dependence between features is the main source of explainer failure (Wilming et al., 2022; Liu et al., 2021; Wang et al., 2024). And the right answer depends on the estimand, so "the true explanation" has to be stated per method (Chen et al., 2020; Molnar et al., 2023b).

### 4.2 Where it disagrees or is unresolved

The field has not settled which level ground truth should live at. OpenXAI restricts data so that data and model levels coincide; AttributionLab and XAI-Units bypass learning; XAI-BENCH computes model-level targets from a known distribution. Each choice trades realism against a different kind of certainty. Neighbouring fields disagree on how much realism helps: plasmode is "widely claimed" to be closer to reality (Schreck et al., 2024), but whether it beats parametric simulation depends on the setting (Stolte et al., 2024), and empirically motivated designs can mislead (Advani et al., 2019). Whether realistic data changes explainer rankings at all, compared with invented data, has not been measured for explanation methods.

### 4.3 Gaps

- **G1. The combination is missing.** No benchmark found combines a generator learned from real covariates with tractable conditionals, a readable target fitted to real data, and exact or error-bounded targets at both the data and the model level. The closest are XAI-BENCH Section 5.3 (Gaussian features, a kNN target, two datasets) and the additive CBR work (cluster perturbation, one dataset).
- **G2. The validity literature has not crossed over.** The critiques of realistic simulation (estimator selection worse than random, benchmarks favouring methods, refute but not certify, projection) are absent from the XAI benchmarks reviewed, none of which reports rank stability across alternative target functions.
- **G3. Fidelity for explanation purposes is unmeasured.** Generator fidelity is measured on marginals, samples, structure or downstream accuracy, not on the conditional distributions that explanation targets integrate over, and not as "how much does the generator's imperfection change explainer rankings".
- **G4. Exactness is asserted, not reported.** Benchmarks call their targets exact, but none of the tabular ones reviewed reports the numerical error of targets that require integration. XAI-BENCH, for example, describes its metrics as computed exactly because the Gaussian conditionals are known, yet evaluates the expectations by sampling from them (Liu et al., 2021). Epidemiology has a worked procedure for reporting that error (Naimi et al., 2025).
- **G5. Classification and mixed types.** Both real-data-derived XAI precedents are regression only.
- **G6. No tabular review.** The one review of synthetic ground-truth methods covers images only (Miró-Nicolau et al., 2026b).

### 4.4 Implications for the proposed benchmark

The pitch sits squarely in gap G1, and the literature supports its core idea. It also shows where the pitch's claims need to change.

1. **Position it as a generalisation, not a first.** The idea of fitting simulated data to real data for XAI appears in XAI-BENCH Section 5.3 (Liu et al., 2021), and the copula-plus-symbolic-target structure in Barr et al. (2020). The defensible novelty is the combination: a non-parametric generator with tractable conditionals (Watson et al., 2023), a fitted closed-form target, model-level exact targets, a per-dataset fidelity audit, and scale. The causal-inference precedents (plasmode, RealCause, Credence, WGAN designs, and above all synth-validation, which already selects methods by their accuracy on generators fitted to the dataset at hand) should be cited as the methodological lineage.
2. **Replace the transfer claim.** The slide stating that "anything you observe on the simulated side transfers to the real one" is contradicted by the evidence in Section 3.5 (Advani et al., 2019; Karim, 2026). The defensible claim is Karim's: the benchmark can refute an explainer's claims (if it fails on realistic data with known truth, it can fail) but cannot certify that it works on the real data. Transfer should be measured per dataset as a result, which the pitch's "task fidelity" and "explanation fidelity" checks already start to do.
3. **Guard against projection twice over.** A symbolic target whose form matches an explainer's assumptions (an additive target for additive explainers) hands that explainer the right answer by construction (Karim, 2026; Curth and van der Schaar, 2021). Separately, an explainer that uses ARFs internally, such as cARFi (Blesch et al., 2025), would be scored against its own generator. Both are addressed the same way: several target families per dataset (several members of a symbolic-regression Pareto front, plus a non-symbolic target such as a fitted tree ensemble) and at least one second generator (for example a probabilistic circuit (Scassola et al., 2026)), with method rankings reported for each and their stability as a headline result.
4. **Treat symbolic-regression instability as a feature.** Because small data changes yield very different expressions (Kammerer et al., 2024), the target is a draw from a set of plausible closed forms. Publishing several draws per dataset turns a weakness into a sensitivity analysis.
5. **Say "error-bounded", not "exact", where integration is needed.** ARF conditionals are tractable, but integrating a non-linear target over them generally needs numerical or Monte Carlo integration; report the error as Naimi et al. (2025) do.
6. **List the estimands.** For each explanation method, state the target it estimates (interventional or conditional Shapley, partial dependence, ALE, permutation or conditional feature importance), and report error against that target separately from the gap between that target and the data-generating function (Chen et al., 2020; Molnar et al., 2023b; Heiß et al., 2026).
7. **Measure fidelity where it matters, and do not lean on it.** Add conditional and structural fidelity (Jiang et al., 2025) and explanation-aware fidelity (Kapar et al., 2025; Yu et al., 2025) to the discriminator AUC, MMD and per-feature tests, since standard metrics can saturate (Scassola et al., 2026). Fidelity bounds how far the benchmark's targets are from the real ones, not whether its explainer ranking is the real ranking (Karim, 2026), so it cannot stand in for a direct test of ranking transfer.
8. **Cover classification.** A link function (for example Bernoulli of a sigmoid of the target) and categorical features are needed; both precedents are regression only.
9. **A natural headline experiment.** Compute popular proxy metrics on the benchmark and measure which ones track actual explanation error. This addresses the meta-evaluation problem (Hedström et al., 2023; Klein et al., 2024) with ground truth, which neither of those papers had.

The benchmark venues used by the closest prior work (XAI-BENCH, OpenXAI, LATEC and SRBench all appeared in the NeurIPS Datasets and Benchmarks track) indicate where a paper of this kind would be judged.

### 4.5 Limitations of this review

- **Single reviewer, and not a human one.** Searching, screening, data extraction and writing were done in one session by an AI assistant (Claude). Every quotation was checked verbatim against the source text or abstract, and every reference against its registry, but no human has yet double-checked the inclusion decisions or the borderline exclusions.
- **Coverage.** DBLP and Semantic Scholar were unavailable. OpenAlex does not index the ARF paper and holds only the preprint records of XAI-BENCH and Advani et al., so citation chaining from those seeds is incomplete.
- **Screening depth.** 663 records had no abstract after backfilling and were screened on title. Only the 700 highest-ranked of the 888 passing records were screened manually. The automated screen rejected six works that later proved relevant, two of them central, so its recall is imperfect; works it rejected were not screened manually.
- **Recency.** Several cited works are 2025-2026 preprints and not peer reviewed, including the two closest critiques of plasmode designs (Shaw et al., 2026; Karim, 2026) and the newest synthetic ground-truth framework (Miró-Nicolau et al., 2026a).
- **Appraisal.** No formal quality-appraisal tool applies to methods papers of this kind. Claims are reported with their source and, where a number is cited, from the paper itself.
- **Novelty is not proven.** The search makes it unlikely that a benchmark combining all elements of G1 exists in the indexed literature, but cannot rule out work under different vocabulary, in workshop papers, or unpublished.

---

## 5. Conclusion

Realistic semi-synthetic data is not a new idea. Epidemiology, econometrics and causal machine learning have used real covariates with simulated outcomes for over a decade, and have learned, through theory and failed predictions, that such benchmarks can mislead about which method is best and can refute but not certify. The XAI benchmark literature has largely taken the other road: invented feature distributions that make ground truth easy, at the price of the dependence structure that causes explainers to fail. The two exceptions that draw on real tabular data are small and use simple generators and targets.

A benchmark that learns the feature distribution from real data with a generator whose conditionals are computable, fits readable targets to real data, computes estimand-specific targets with stated error at both the data and the model level, and reports how stable explainer rankings are across generators and target families would fill a real gap. It would also be the first to bring the hard-won validity lessons of the neighbouring fields into explanation evaluation. The strongest version of the proposal is the one that says so explicitly, and claims refutation rather than certification.

## References

Aas, K., Jullum, M., Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *Artificial Intelligence*. <https://doi.org/10.1016/j.artint.2021.103502>

Adebayo, J., Muelly, M., Abelson, H., et al. (2022). Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation. *ICLR 2022*. <https://arxiv.org/abs/2212.04629>

Advani, A., Kitagawa, T., Słoczyński, T. (2019). Mostly harmless simulations? Using Monte Carlo studies for estimator selection. *Journal of Applied Econometrics*. <https://doi.org/10.1002/jae.2724>

Agarwal, C., Ley, D., Krishna, S., et al. (2022). OpenXAI: Towards a Transparent Evaluation of Model Explanations. *NeurIPS 2022 Datasets and Benchmarks*. <https://arxiv.org/abs/2206.11104>

Agarwal, C., Queen, O., Lakkaraju, H., et al. (2023). Evaluating explainability for graph neural networks. *Scientific Data*. <https://doi.org/10.1038/s41597-023-01974-x>

Akhavan Rahnama, A. H., Bütepage, J., Geurts, P., et al. (2023). Can local explanation techniques explain linear additive models? *Data Mining and Knowledge Discovery*. <https://doi.org/10.1007/s10618-023-00971-3>

Alaa, A. M., van Breugel, B., Saveliev, E., et al. (2022). How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models. *ICML 2022*. <https://arxiv.org/abs/2102.08921>

Aldeia, G. S. I., de França, F. O. (2022). Interpretability in symbolic regression: a benchmark of explanatory methods using the Feynman data set. *Genetic Programming and Evolvable Machines*. <https://doi.org/10.1007/s10710-022-09435-x>

Amiri, S. S., Weber, R. O., Goel, P., et al. (2020). Data Representing Ground-Truth Explanations to Evaluate XAI Methods. *arXiv preprint arXiv:2011.09892*. <https://arxiv.org/abs/2011.09892>

Arras, L., Osman, A., Samek, W. (2022). CLEVR-XAI: A benchmark dataset for the ground truth evaluation of neural network explanations. *Information Fusion*. <https://doi.org/10.1016/j.inffus.2021.11.008>

Athey, S., Imbens, G. W., Metzger, J., et al. (2024). Using Wasserstein Generative Adversarial Networks for the design of Monte Carlo simulations. *Journal of Econometrics*. <https://doi.org/10.1016/j.jeconom.2020.09.013>

Barr, B., Xu, K., Silva, C., et al. (2020). Towards Ground Truth Explainability on Tabular Data. *ICML 2020 Workshop on Human Interpretability in ML*. <https://arxiv.org/abs/2007.10532>

Bastings, J., Ebert, S., Zablotskaia, P., et al. (2022). "Will You Find These Shortcuts?" A Protocol for Evaluating the Faithfulness of Input Salience Methods for Text Classification. *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*. <https://doi.org/10.18653/v1/2022.emnlp-main.64>

Blesch, K., Watson, D. S., Wright, M. N. (2023a). Conditional feature importance for mixed data. *AStA Advances in Statistical Analysis*. <https://doi.org/10.1007/s10182-023-00477-9>

Blesch, K., Wright, M. N., Watson, D. (2023b). Unfooling SHAP and SAGE: Knockoff Imputation for Shapley Values. *Communications in Computer and Information Science*. <https://doi.org/10.1007/978-3-031-44064-9_8>

Blesch, K., Koenen, N., Kapar, J., et al. (2025). Conditional Feature Importance with Generative Modeling Using Adversarial Random Forests. *Proceedings of the AAAI Conference on Artificial Intelligence*. <https://doi.org/10.1609/aaai.v39i15.33712>

Borisov, V., Leemann, T., Seßler, K., et al. (2022). Deep Neural Networks and Tabular Data: A Survey. *IEEE Transactions on Neural Networks and Learning Systems*. <https://doi.org/10.1109/tnnls.2022.3229161>

Brandt, R., Raatjens, D., Gaydadjiev, G. (2023). Precise Benchmarking of Explainable AI Attribution Methods. *arXiv preprint arXiv:2308.03161*. <https://arxiv.org/abs/2308.03161>

Candès, E., Fan, Y., Janson, L., et al. (2018). Panning for Gold: 'Model-X' Knockoffs for High Dimensional Controlled Variable Selection. *Journal of the Royal Statistical Society Series B: Statistical Methodology*. <https://doi.org/10.1111/rssb.12265>

Chen, H., Janizek, J. D., Lundberg, S., et al. (2020). True to the Model or True to the Data? *arXiv preprint arXiv:2006.16234*. <https://arxiv.org/abs/2006.16234>

Clark, B., Wilming, R., Haufe, S. (2023). XAI-TRIS: Non-linear image benchmarks to quantify false positive post-hoc attribution of feature importance. *Machine Learning*. <https://doi.org/10.1007/s10994-024-06574-3>

Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. *arXiv preprint arXiv:2305.01582*. <https://arxiv.org/abs/2305.01582>

Curth, A., van der Schaar, M. (2021). Doing Great at Estimating CATE? On the Neglected Assumptions in Benchmark Comparisons of Treatment Effect Estimators. *ICML 2021 Workshop on the Neglected Assumptions in Causal Inference*. <https://arxiv.org/abs/2107.13346>

Dorie, V., Hill, J., Shalit, U., et al. (2019). Automated versus Do-It-Yourself Methods for Causal Inference: Lessons Learned from a Data Analysis Competition. *Statistical Science*. <https://doi.org/10.1214/18-sts667>

Franklin, J. M., Schneeweiß, S., Polinski, J. M., et al. (2013). Plasmode simulation for the evaluation of pharmacoepidemiologic methods in complex healthcare databases. *Computational Statistics & Data Analysis*. <https://doi.org/10.1016/j.csda.2013.10.018>

de França, F. O., Virgolin, M., Kommenda, M., et al. (2024). SRBench++: Principled Benchmarking of Symbolic Regression With Domain-Expert Interpretation. *IEEE Transactions on Evolutionary Computation*. <https://doi.org/10.1109/tevc.2024.3423681>

Freiesleben, T., König, G., Molnar, C., et al. (2024). Scientific Inference with Interpretable Machine Learning: Analyzing Models to Learn About Real-World Phenomena. *Minds and Machines*. <https://doi.org/10.1007/s11023-024-09691-z>

Frye, C., de Mijolla, D., Begley, T., et al. (2021). Shapley explainability on the data manifold. *ICLR 2021*. <https://arxiv.org/abs/2006.01272>

Guidotti, R. (2020). Evaluating local explanation methods on ground truth. *Artificial Intelligence*. <https://doi.org/10.1016/j.artint.2020.103428>

Hedström, A., Bommer, P., Wickstrøm, K. K., et al. (2023). The Meta-Evaluation Problem in Explainable AI: Identifying Reliable Estimators with MetaQuantus. *TMLR 2023*. <https://arxiv.org/abs/2302.07265>

Heiß, T., Bögel, C., Bischl, B., et al. (2026). Analyzing Error Sources in Global Feature Effect Estimation. *arXiv preprint arXiv:2603.15057*. <https://arxiv.org/abs/2603.15057>

Hesse, R., Schaub-Meyer, S., Roth, S. (2023). FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods. *ICCV 2023*. <https://arxiv.org/abs/2308.06248>

Hill, J. L. (2011). Bayesian Nonparametric Modeling for Causal Inference. *Journal of Computational and Graphical Statistics*. <https://doi.org/10.1198/jcgs.2010.08162>

Hooker, G., Mentch, L., Zhou, S. (2021). Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance. *Statistics and Computing*. <https://doi.org/10.1007/s11222-021-10057-z>

Islam, M. R., Weber, R. O., Ahmed, M. U., et al. (2025). Privacy-preserving ground-truth data for evaluating additive feature attribution in regression models with additive CBR and CQV. *Knowledge-Based Systems*. <https://doi.org/10.1016/j.knosys.2025.114599>

Jiang, X., Simidjievski, N., Jamnik, M. (2025). How Well Does Your Tabular Generator Learn the Structure of Tabular Data? *arXiv preprint arXiv:2503.09453*. <https://arxiv.org/abs/2503.09453>

Kammerer, L., Kronberger, G., Winkler, S. (2024). Bias and Variance Analysis of Contemporary Symbolic Regression Methods. *Applied Sciences*. <https://doi.org/10.3390/app142311061>

Kapar, J., Koenen, N., Jullum, M. (2025). What's Wrong with Your Synthetic Tabular Data? Using Explainable AI to Evaluate Generative Models. *Communications in Computer and Information Science*. <https://doi.org/10.1007/978-3-032-08327-2_2>

Karim, M. E. (2026). A Plasmode Benchmark Can Refute but Cannot Certify: Projection and Regularity in Causal-Inference Simulation. *arXiv preprint arXiv:2609.26186*. <https://arxiv.org/abs/2609.26186>

Kim, J. S., Plumb, G., Talwalkar, A. (2022). Sanity Simulations for Saliency Methods. *ICML 2022*. <https://arxiv.org/abs/2105.06506>

Klein, L., Lüth, C. T., Schlegel, U., et al. (2024). Navigating the Maze of Explainable AI: A Systematic Approach to Evaluating Methods and Metrics. *NeurIPS 2024 Datasets and Benchmarks*. <https://arxiv.org/abs/2409.16756>

Knaus, M. C., Lechner, M., Strittmatter, A. (2020). Machine learning estimation of heterogeneous causal effects: Empirical Monte Carlo evidence. *The Econometrics Journal*. <https://doi.org/10.1093/ectj/utaa014>

Kotelnikov, A., Baranchuk, D., Rubachev, I., et al. (2023). TabDDPM: Modelling Tabular Data with Diffusion Models. *ICML 2023*. <https://arxiv.org/abs/2209.15421>

Krishna, S., Han, T., Gu, A., et al. (2024). The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective. *TMLR 2024*. <https://arxiv.org/abs/2202.01602>

La Cava, W., Orzechowski, P., Burlacu, B., et al. (2021). Contemporary Symbolic Regression Methods and their Relative Performance. *NeurIPS 2021 Datasets and Benchmarks*. <https://arxiv.org/abs/2107.14351>

Lee, J. R., Emami, S., Hollins, M. D., et al. (2025). XAI-Units: Benchmarking Explainability Methods with Unit Tests. *FAccT 2025*. <https://arxiv.org/abs/2506.01059>

Li, X., Du, M., Chen, J., et al. (2023). $\mathcal{M}^4$: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods across Metrics, Modalities and Models. *Advances in Neural Information Processing Systems 36*. <https://doi.org/10.52202/075280-0081>

Liu, Y., Khandagale, S., White, C., et al. (2021). Synthetic Benchmarks for Scientific Research in Explainable Machine Learning. *NeurIPS 2021 Datasets and Benchmarks*. <https://arxiv.org/abs/2106.12543>

Mamalakis, A., Barnes, E. A., Ebert‐Uphoff, I. (2022a). Investigating the Fidelity of Explainable Artificial Intelligence Methods for Applications of Convolutional Neural Networks in Geoscience. *Artificial Intelligence for the Earth Systems*. <https://doi.org/10.1175/aies-d-22-0012.1>

Mamalakis, A., Ebert‐Uphoff, I., Barnes, E. A. (2022b). Neural network attribution methods for problems in geoscience: A novel synthetic benchmark dataset. *Environmental Data Science*. <https://doi.org/10.1017/eds.2022.7>

Matsubara, Y., Chiba, N., Igarashi, R., et al. (2024). Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery. *DMLR 2024*. <https://arxiv.org/abs/2206.10540>

Miró-Nicolau, M., Jaume-i-Capó, A., Moyà-Alcover, G. (2024a). A comprehensive study on fidelity metrics for XAI. *Information Processing & Management*. <https://doi.org/10.1016/j.ipm.2024.103900>

Miró-Nicolau, M., Jaume-i-Capó, A., Moyà-Alcover, G. (2024b). Assessing fidelity in XAI post-hoc techniques: A comparative study with ground truth explanations datasets. *Artificial Intelligence*. <https://doi.org/10.1016/j.artint.2024.104179>

Miró-Nicolau, M., Spinnato, F., Guidotti, R. (2026a). A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods. *arXiv preprint arXiv:2609.30397*. <https://arxiv.org/abs/2609.30397>

Miró-Nicolau, M., Moyà-Alcover, G., Arias-Duart, A. (2026b). Exploring SAIG Methods for an Objective Evaluation of XAI. *arXiv preprint arXiv:2602.08715*. <https://arxiv.org/abs/2602.08715>

Molnar, C., König, G., Bischl, B., et al. (2023a). Model-agnostic feature importance and effects with dependent features: a conditional subgroup approach. *Data Mining and Knowledge Discovery*. <https://doi.org/10.1007/s10618-022-00901-9>

Molnar, C., Freiesleben, T., König, G., et al. (2023b). Relating the Partial Dependence Plot and Permutation Feature Importance to the Data Generating Process. *Communications in Computer and Information Science*. <https://doi.org/10.1007/978-3-031-44064-9_24>

Morris, T. P., White, I. R., Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine*. <https://doi.org/10.1002/sim.8086>

Naimi, A. I., Benkeser, D. C., Rudolph, J. E. (2025). Computing True Parameter Values in Simulation Studies Using Monte Carlo Integration. *Epidemiology*. <https://doi.org/10.1097/ede.0000000000001873>

Nauta, M., Trienes, J., Pathak, S., et al. (2023). From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI. *ACM Computing Surveys*. <https://doi.org/10.1145/3583558>

Neal, B., Huang, C. W., Raghupathi, S. (2020). RealCause: Realistic Causal Inference Benchmarking. *arXiv preprint arXiv:2011.15007*. <https://arxiv.org/abs/2011.15007>

Olsen, L. H. B., Glad, I. K., Jullum, M., et al. (2024). A comparative study of methods for estimating model-agnostic Shapley value explanations. *Data Mining and Knowledge Discovery*. <https://doi.org/10.1007/s10618-024-01016-z>

Parikh, H., Varjao, C., Xu, L., et al. (2022). Validating Causal Inference Methods. *ICML 2022*. <https://arxiv.org/abs/2202.04208>

Scassola, D., Ponsford, D., Javaloy, A., et al. (2026). A Sobering Look at Tabular Data Generation via Probabilistic Circuits. *arXiv preprint arXiv:2603.23016*. <https://arxiv.org/abs/2603.23016>

Schreck, N., Slynko, A., Saadati, M., et al. (2024). Statistical plasmode simulations - Potentials, challenges and recommendations. *Statistics in Medicine*. <https://arxiv.org/abs/2305.06028>

Schuler, A., Jung, K., Tibshirani, R., et al. (2017). Synth-Validation: Selecting the Best Causal Inference Method for a Given Dataset. *arXiv preprint arXiv:1711.00083*. <https://arxiv.org/abs/1711.00083>

Shaw, P. A., Gruber, S., Williamson, B. D., et al. (2026). A Cautionary Note for Plasmode Simulation Studies in the Setting of Causal Inference. *Statistics in Medicine*. <https://doi.org/10.1002/sim.70676>

Stolte, M., Schreck, N., Slynko, A., et al. (2024). Simulation study to evaluate when Plasmode simulation is superior to parametric simulation in estimating the mean squared error of the least squares estimator in linear regression. *PLOS ONE*. <https://doi.org/10.1371/journal.pone.0299989>

Suk, Y., Pan, C., Xiao, W. (2026). Generative AI-Based Monte Carlo Simulation for Method Evaluation Using Synthetic Multilevel Data. *arXiv preprint arXiv:2605.05752*. <https://arxiv.org/abs/2605.05752>

Tian, Y., Schuemie, M. J., Suchard, M. A. (2018). Evaluating large-scale propensity score performance through real-world and synthetic data experiments. *International Journal of Epidemiology*. <https://doi.org/10.1093/ije/dyy120>

Udrescu, S. M., Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*. <https://doi.org/10.1126/sciadv.aay2631>

van Breugel, B., Qian, Z., van der Schaar, M. (2023). Synthetic data, real errors: how (not) to publish and use synthetic data. *ICML 2023*. <https://arxiv.org/abs/2305.09235>

Wang, T., Ragodos, R., Feng, L., et al. (2024). From Model Explanation to Data Misinterpretation: A Cautionary Analysis of Post Hoc Explainers in Business Research. *arXiv preprint arXiv:2408.16987*. <https://arxiv.org/abs/2408.16987>

Watson, D. S., Wright, M. N. (2021). Testing conditional independence in supervised learning algorithms. *Machine Learning*. <https://doi.org/10.1007/s10994-021-06030-6>

Watson, D. S., Blesch, K., Kapar, J., et al. (2023). Adversarial random forests for density estimation and generative modeling. *AISTATS 2023*. <https://arxiv.org/abs/2205.09435>

Wilming, R., Budding, C., Müller, K. R., et al. (2022). Scrutinizing XAI using linear ground-truth data with suppressor variables. *Machine Learning*. <https://doi.org/10.1007/s10994-022-06167-y>

Xu, L., Skoularidou, M., Cuesta-Infante, A., et al. (2019). Modeling Tabular data using Conditional GAN. *NeurIPS 2019*. <https://arxiv.org/abs/1907.00503>

Yang, M., Kim, B. (2019). Benchmarking Attribution Methods with Relative Feature Importance. *arXiv preprint arXiv:1907.09701*. <https://arxiv.org/abs/1907.09701>

Yu, K., Ishikura, S., Usukura, Y., et al. (2025). SHAP Distance: An Explainability-Aware Metric for Evaluating the Semantic Fidelity of Synthetic Tabular Data. *2025 IEEE International Conference on Big Data (BigData)*. <https://doi.org/10.1109/bigdata66926.2025.11402338>

Zhang, Y., Li, Y., Brown, H., et al. (2023). AttributionLab: Faithfulness of Feature Attribution Under Controllable Environments. *NeurIPS 2023 Workshop XAIA*. <https://arxiv.org/abs/2310.06514>

Zhou, Y., Booth, S., Ribeiro, M. T., et al. (2022). Do Feature Attribution Methods Correctly Attribute Features? *Proceedings of the AAAI Conference on Artificial Intelligence*. <https://doi.org/10.1609/aaai.v36i9.21196>
