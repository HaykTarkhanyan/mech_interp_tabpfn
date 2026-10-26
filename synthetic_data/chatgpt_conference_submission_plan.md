# Conference Submission Plan

ChatGPT conversation of 2026-10-24 (started 2:38 PM), from the workspace share link https://chatgpt.com/share/e/6adcac9a-5b34-8012-9ade-7fc51ca69514, saved from the page the user downloaded on 2026-10-24 16:30.
Math is restored to LaTeX; citation chips are inline links with tracking parameters removed. A chip's extra sources ("+7") are not in the saved page. Hidden reasoning is not part of the share and is left out.

---

## Question 1

> I want to try to write a paper on this topic. Like imagine people like to have some synthetic datasets because there is a closed form solution, so they can verify their explainability method and so on. But synthetic methods are usually not representative of the real world. So I would like to write a paper on taking real datasets and for them generating closed form solutions for the approximations for the target, and kind of publish this dataset as a benchmark and also a toolkit of already fitted interpretability models on top of it and so on and so forth. I'm curious for which conference can I submit that and when is the deadline for that, and also for which track.

### ChatGPT

Your idea is a strong fit for **Datasets & Benchmarks tracks at major machine learning conferences**, particularly NeurIPS and KDD.

I would characterize the proposed contribution as:

**A semi-synthetic benchmark for evaluating explainable AI methods using real-world feature distributions and analytically defined prediction functions.**

The key distinction is that you would preserve the characteristics of real datasets—feature correlations, distributions, imbalances, and other complexities—while introducing a known mathematical function for the target. This would allow researchers to evaluate explanation methods against a precisely defined reference.

I've checked the official conference announcements and submission deadlines.

## 1. Conferences, tracks, and deadlines

**As of October 24, 2026**, these are the most relevant opportunities for your paper.

| Conference | Recommended track | Submission deadline | Suitability |
|----|----|----|----|
| **NeurIPS 2027** | Evaluations & Datasets (if retained for 2027) | Not announced. The 2026 deadline was May 6. | ⭐⭐⭐⭐⭐ |
| **KDD 2027** | Datasets & Benchmarks | Second-cycle deadline not announced. First cycle closed July 26, 2026. | ⭐⭐⭐⭐⭐ |
| **ICML 2027** | Main Conference — Evaluation / Trustworthy ML | Not announced. The 2026 deadline was January 28. | ⭐⭐⭐⭐ |
| **IJCAI 2027** | Main Track | **January 4, 2027** (abstract); **January 11, 2027** (paper) | ⭐⭐⭐⭐ |
| **ECML PKDD 2027** | Research Track | Not announced | ⭐⭐⭐⭐ |
| **PAKDD 2027** | Research Track | **November 20, 2026** | ⭐⭐⭐ |
| **ECAI 2027** | Main Technical Conference | **April 14, 2027** | ⭐⭐⭐⭐ |
| **EXPLAINS 2027** | Technology — XAI Benchmarking | **April 27, 2027** (regular-paper cycle) | ⭐⭐⭐⭐ |

*Suitability is my assessment of the fit between your proposed contribution and each venue, not an acceptance probability.*

The dates and tracks come from the official conference websites. Where the 2027 call has not been released, I've deliberately avoided presenting estimated deadlines as confirmed. ([Call For Evaluations & Datasets 2026](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets))

### My top recommendation: NeurIPS 2027

NeurIPS is probably the best venue if your paper demonstrates that the benchmark enables scientifically meaningful evaluations that existing benchmarks cannot.

An important detail: **NeurIPS renamed its Datasets & Benchmarks track to Evaluations & Datasets in 2026.**

The updated track explicitly welcomes:

- New datasets and dataset collections.
- Data generators and evaluation tools.
- Benchmarking methodologies.
- Studies exposing shortcomings in existing evaluation practices.

It also explicitly states that submissions do not need to introduce a new model or outperform existing methods.

This matches your idea particularly well. ([Call For Evaluations & Datasets 2026](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets))

The 2027 submission dates have not yet been announced. The 2026 deadlines were May 4 for abstracts and May 6 for full papers.

Official track description: [NeurIPS Evaluations & Datasets](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)

### Second recommendation: KDD 2027

KDD has a dedicated **Datasets & Benchmarks Track**.

Its official call specifically welcomes collections constructed from existing datasets, benchmarking tools, and data generators.

However, it emphasizes that a collection of previously available datasets must provide substantial new value beyond the original data.

Your contribution could satisfy this through analytically defined targets, reference explanations, and standardized evaluation protocols.

KDD 2027 has two submission cycles. The first has closed, and the second cycle's dates are not yet published. ([Datasets and Benchmarks Track: Call for Papers – KDD 2027](https://kdd2027.kdd.org/datasets-and-benchmarks-track-call-for-papers/))

Official call: [KDD 2027 Datasets & Benchmarks](https://kdd2027.kdd.org/datasets-and-benchmarks-track-call-for-papers/)

### Other opportunities worth considering

**IJCAI 2027** is particularly interesting because the deadline is January 11, 2027, with an earlier mandatory abstract deadline of January 4. Its Main Track welcomes significant original contributions across AI. It would be a stronger choice if your work includes a novel methodological component rather than primarily a dataset release. ([IJCAI 2027 Call for papers (Main Track) – IJCAI 2027](https://2027.ijcai.org/ijcai-2027-call-for-papers-main-track/))

**ECML PKDD 2027** takes place in Eindhoven, Netherlands, from August 30 to September 3, 2027. The Research Track is an appropriate target, although the 2027 submission deadline has not been published. ([ecmlpkdd.org](https://ecmlpkdd.org/2027/))

**EXPLAINS 2027** is a more specialized venue. Its Technology area explicitly includes *Evaluation of Explainability* and *XAI Benchmarking*. The regular-paper deadline is April 27, 2027, with another submission round listed for June 3. ([EXPLAINS 2027 - CallForPapers](https://explains.scitevents.org/CallForPapers.aspx?y=2027))

One additional specialist venue is the **World Conference on Explainable Artificial Intelligence (XAI 2027)**. It explicitly lists XAI benchmarking among its topics, but its 2027 website currently contains inconsistent, apparently outdated 2026 submission dates. I would not rely on those dates yet. ([Call for papers](https://xaiworldconference.com/2027/call-for-papers/))

For completeness, the ICLR 2027 deadline has already passed: September 25, 2026. ([ICLR 2027 Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers))

------------------------------------------------------------------------

## 2. How I would position your paper

I think the framing will matter substantially for acceptance.

Consider a tentative title:

**RealXAI-Bench: Bridging Synthetic and Real-World Data for Ground-Truth Evaluation of Explainable AI**

The central research problem would be:

> Existing explainability benchmarks face a trade-off between the availability of ground-truth explanations and the realism of evaluation data. We propose a framework that preserves real-world feature distributions while introducing analytically tractable prediction functions, enabling controlled evaluation of explanation methods under realistic data conditions.

Your methodology could look like this.

Start with a real dataset:

$$D=\{(x_i,y_i)\}_{i=1}^{n}$$

Fit an interpretable approximation:

$$\hat f(x)=\sum_j f_j(x_j)+\sum_{j<k}f_{jk}(x_j,x_k)$$

Here, the functions might come from generalized additive models, symbolic regression, or other analytically tractable model families.

Then define new benchmark targets:

$$y_i^*=\hat f(x_i)$$

You now have a semi-synthetic dataset with real-world inputs and a known mathematical target function.

Depending on the function family and your definition of explanation, you can derive reference feature effects, gradients, interaction effects, or attribution values.

**One important methodological distinction:** the fitted function is not the true data-generating process behind the original dataset. It provides a known reference function for the benchmark. Consequently, you should be explicit about which explanation quantities are mathematically known and which depend on assumptions about the feature distribution.

Likewise, explaining a black-box model trained to approximate $\hat f$ is not necessarily equivalent to explaining $\hat f$ itself.

Addressing these distinctions rigorously would strengthen the paper.

------------------------------------------------------------------------

## 3. Important: similar research already exists

Before committing to the paper, I would carefully examine three publications.

| Paper | Venue | Relevance to your proposal |
|----|----|----|
| **Synthetic Benchmarks for Scientific Research in Explainable Machine Learning (XAI-BENCH)** | NeurIPS 2021, Datasets & Benchmarks | Introduces synthetic benchmark datasets and a library for evaluating feature-attribution methods. |
| **OpenXAI: Towards a Transparent Evaluation of Model Explanations** | NeurIPS 2022, Datasets & Benchmarks | Provides real datasets, pretrained models, explainability algorithms, evaluation metrics, and benchmarking infrastructure. |
| **A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods** | arXiv, September 2026 | Introduces a framework for evaluating XAI methods using controlled synthetic ground truth across multiple data modalities. |

These papers are directly relevant and should be among the starting points for your literature review. ([Synthetic Benchmarks for Scientific Research in Explainable Machine Learning](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/c16a5320fa475530d9583c34fd356ef5-Abstract-round2.html))

The first two are especially informative because they show that the research community has already accepted explainability benchmarking frameworks in the NeurIPS Datasets & Benchmarks track.

However, they also establish that simply releasing datasets, fitted models, and explainability tools will not necessarily constitute sufficient novelty.

**Your strongest differentiation is the combination of realistic input distributions and analytically defined evaluation targets.**

I would make that the central scientific contribution.

------------------------------------------------------------------------

## 4. What would make this publishable at NeurIPS or KDD?

My assessment is that the benchmark could be competitive at these venues if you establish three things.

**First, the generated benchmarks must preserve meaningful real-world complexity.**

In particular, feature dependence, skewness, class imbalance, and interactions should be investigated. The point is not merely that the source observations came from real datasets, but that those characteristics create evaluation conditions that conventional synthetic benchmarks fail to reproduce.

**Second, the benchmark must offer mathematically defensible reference explanations.**

This will require carefully specifying the explanation target. For example, the marginal effects encoded by a generalized additive model and the Shapley values obtained under a correlated feature distribution are not automatically the same quantity.

**Third, the benchmark should reveal findings that existing evaluation setups miss.**

Imagine demonstrating that LIME, SHAP, or Integrated Gradients performs well on conventional synthetic datasets but substantially worse under realistic feature dependencies—or that their relative rankings change.

A result like that would make the benchmark a scientific contribution rather than just a software resource.

I would also include a reusable toolkit, multiple real datasets, several families of analytical functions, reproducible evaluation scripts, and pretrained baseline models.

For NeurIPS specifically, the 2026 Evaluations & Datasets track requires executable benchmark artifacts to be accessible at submission time and requires Croissant metadata for datasets. That is worth accounting for early in development, although the 2027 requirements may change. ([Call For Evaluations & Datasets 2026](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets))

------------------------------------------------------------------------

## 5. My suggested submission strategy

Given that you are currently developing the research idea rather than describing an already completed study, **I would aim for NeurIPS 2027 as the primary target**.

That offers time to establish the novelty, implement the toolkit, and conduct a systematic comparison of existing XAI methods.

The alternative would be KDD 2027's second Datasets & Benchmarks submission cycle, once its dates are announced.

If you can develop the work more quickly, IJCAI 2027 provides a confirmed January 11 deadline.

I would not rush toward PAKDD's November 20 deadline unless substantial experimental work is already completed.

### One question that will determine the best conference and track

There are actually two quite different potential papers here:

**A. Benchmark paper:** You assemble a collection of real-world datasets, fit analytical target functions, provide reference explanations, and release a comprehensive evaluation toolkit.

**B. Methodology paper:** You introduce a novel algorithm for transforming arbitrary real datasets into mathematically tractable XAI benchmarks, and demonstrate that it generalizes across datasets and model families.

Both are potentially publishable, but I would prioritize **NeurIPS Evaluations & Datasets or KDD Datasets & Benchmarks for A**, whereas **ICML, IJCAI, or the NeurIPS Main Track** could become more compelling for B.

Your current description sounds closer to **A, with B as a possible additional contribution**. That distinction is worth resolving before you design the experiments.

<details><summary>Sources cited in this answer</summary>

- [Call For Evaluations & Datasets 2026](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
- [Datasets and Benchmarks Track: Call for Papers – KDD 2027](https://kdd2027.kdd.org/datasets-and-benchmarks-track-call-for-papers/)
- [IJCAI 2027 Call for papers (Main Track) – IJCAI 2027](https://2027.ijcai.org/ijcai-2027-call-for-papers-main-track/)
- [ecmlpkdd.org](https://ecmlpkdd.org/2027/)
- [EXPLAINS 2027 - CallForPapers](https://explains.scitevents.org/CallForPapers.aspx?y=2027)
- [Call for papers](https://xaiworldconference.com/2027/call-for-papers/)
- [ICLR 2027 Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [Synthetic Benchmarks for Scientific Research in Explainable Machine Learning](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/c16a5320fa475530d9583c34fd356ef5-Abstract-round2.html)

</details>

---

## Question 2

> Thsoe 3 papers u mentioned, how mnay citations they have and pls also read them in full to identifying gaps i can address

### ChatGPT

I reviewed the three papers, including their methodologies, experiments, results, and limitations. I also searched for more closely related work.

**My main finding is that your idea is viable, but there is an important novelty issue:** using real-world features with synthetic targets has already been explored. To make this a strong NeurIPS or KDD submission, I would focus on **fitting analytically tractable target functions to actual real-world labels, constructing mathematically precise reference explanations, and demonstrating what conventional synthetic benchmarks fail to reveal.**

There is also a fourth paper—and a fifth—that you should know about before starting.

# 1. Citation counts

Citation counts vary significantly between academic indexes. Here are the figures I could verify as of **October 24, 2026**.

| Paper | Year | Citations | Source |
|----|----|---:|----|
| **Synthetic Benchmarks for Scientific Research in Explainable Machine Learning (XAI-Bench)** | 2021 | **91** | Scholar Feed |
| **OpenXAI: Towards a Transparent Evaluation of Model Explanations** | 2022 | **212** | Scholar Feed |
| **A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods** | 2026 | **0 indexed so far** | mlsanity |

The first two figures come from the same service. The third paper appeared on arXiv only on September 24, 2026, so its citation count is not particularly informative yet. ([Synthetic Benchmarks for Scientific Research in Explainable Machine Learning \| Scholar Feed](https://www.scholarfeed.org/paper/2106.12543))

For comparison, mlsanity tracks **61 citing papers** for XAI-Bench and **129** for OpenXAI in its more restricted citation collection. These are not necessarily the complete citation totals. ([Synthetic Benchmarks for Scientific Research in Explainable Machine Learning · mlsanity](https://mlsanity.com/paper/2106.12543/))

## Full papers

You can access the complete papers here:

1.  [XAI-Bench — NeurIPS 2021 PDF](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round2.pdf) — 14 pages.
2.  [OpenXAI — NeurIPS 2022 PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf) — 16 pages.
3.  [Synthetic Ground-Truth Framework — 2026 PDF](https://arxiv.org/pdf/2609.30397) — 19 PDF pages.

------------------------------------------------------------------------

# 2. Paper 1: XAI-Bench (NeurIPS 2021)

**Authors:** Yang Liu, Sujay Khandagale, Colin White, Willie Neiswanger.

**My assessment: This is the closest of the original three papers to your motivation.**

## What they actually did

The authors introduce a framework for evaluating feature-attribution methods using synthetic datasets with known statistical properties.

Their framework supports different feature distributions, including multivariate Gaussians, Gaussian mixtures, and multinomial distributions.

They evaluate methods such as SHAP, LIME, MAPLE, SHAPR, and others using metrics including:

- Ground-truth Shapley agreement.
- Faithfulness and monotonicity.
- Remove-and-retrain (ROAR).
- Infidelity.

Because they control the distributions, they can calculate reference conditional expectations that are generally difficult to estimate from real data. ([datasets-benchmarks-proceedings.neurips.cc](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round2.pdf))

### Particularly relevant: Section 5.3

They already attempt to make synthetic datasets resemble real ones.

They take two real datasets:

1.  Wine Quality.
2.  Forest Fires.

For the Wine Quality dataset, they:

1.  Normalize the original features.
2.  Calculate their empirical covariance matrix.
3.  Generate new multivariate Gaussian observations using that covariance matrix.
4.  Assign synthetic labels using a k-nearest-neighbor model trained on the original data.

This is an important overlap with your proposed idea.

However, they **generate new feature vectors**, rather than retaining the original feature vectors. They also use kNN to produce targets, not an analytically tractable fitted function. (<https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round2.pdf>)

## The gap you could address

Consider a real dataset with the following properties:

- Strong nonlinear dependencies.
- Mixed categorical and numerical variables.
- Skewed feature distributions.
- Multiple correlated subgroups.
- Rare combinations of features.

If you reproduce that dataset through a multivariate Gaussian using its covariance matrix, you preserve the covariance structure but not necessarily those other characteristics.

Your alternative could be:

$$X_{\text{benchmark}}=X_{\text{real}}$$

rather than:

$$X_{\text{benchmark}}\sim \mathcal N(\hat\mu,\hat\Sigma)$$

Then you fit a tractable function to the real labels:

$$g^*=\arg\min_{g\in\mathcal G}
\sum_{i=1}^{n}L(g(x_i),y_i)$$

And define the benchmark targets:

$$y_i^*=g^*(x_i)$$

That gives you an analytical target function while preserving the exact observed feature vectors.

**This distinction is meaningful.**

It eliminates the feature-distribution approximation error introduced by synthetic resampling—at least for the retained observations.

### What I would investigate

An experiment could directly compare:

| Characteristic | XAI-Bench approach | Your approach |
|----|----|----|
| Marginal feature distributions | Approximated | Original observations |
| Feature correlations | Approximated | Original observations |
| Higher-order dependencies | Not generally preserved | Preserved in observed samples |
| Categorical combinations | Requires appropriate modeling | Original combinations |
| Target function | Synthetic/kNN-based | Fitted analytical function |
| Reference explanations | Available under specified synthetic distribution | Potentially exact for selected explanation definitions |

The last distinction requires care, which I will discuss later.

### What would strengthen the paper?

Show that **explanation-method rankings differ** when moving from Gaussian-simulated features to actual real-world features.

For example, suppose SHAP outperforms LIME on Gaussian-generated data, but the ranking reverses on original feature vectors under the same benchmark target.

That would demonstrate that the choice of evaluation data affects the scientific conclusions.

### What the authors acknowledge

Their conclusion explicitly identifies the trade-off between data realism and access to ground truth.

They recognize that synthetic datasets cannot fully replicate real-world characteristics. ([datasets-benchmarks-proceedings.neurips.cc](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round2.pdf))

**My assessment:** A good foundation for your related-work section, but preserving the original feature distribution is not sufficient novelty by itself.

------------------------------------------------------------------------

# 3. Paper 2: OpenXAI (NeurIPS 2022)

**Authors:** Chirag Agarwal and colleagues.

**My assessment: This paper overlaps heavily with your proposed toolkit, but considerably less with your proposed method of generating reference explanations.**

## What they did

OpenXAI is a relatively comprehensive evaluation ecosystem.

It includes:

| Component | Implementation |
|----|----|
| Real datasets | 7 |
| Synthetic datasets | 1 generator |
| Pretrained predictive models | 16 |
| Explanation methods | 6 established methods, plus a random baseline |
| Evaluation metrics | 22 in the published paper |
| Leaderboards | Yes |
| Extensible framework | Yes |

Its real datasets include Adult Income, German Credit, HELOC, COMPAS, Pima Diabetes, and others. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf))

The important implication for your work is:

**Providing datasets, pretrained models, multiple explainers, evaluation metrics, and leaderboards is already an established contribution.**

You should not rely on those components as your principal novelty.

Instead, they should support the methodological contribution.

## Their fundamental limitation

OpenXAI essentially uses two evaluation settings.

**Setting A: Synthetic datasets**

These provide controlled explanations against which methods can be evaluated.

**Setting B: Real datasets**

These provide realistic evaluation settings, but generally lack known ground-truth explanations of the original data-generating process.

That leaves the two advantages separated.

Your proposal aims to combine them.

### Something particularly interesting

The authors discuss a problem involving correlated features.

Imagine the actual target depends on:

$$y=h(x_1,x_2)$$

But the dataset also contains:

$$x_3\approx x_1$$

A trained neural network might learn to rely on $x_3$ instead of $x_1$.

If an explanation method correctly identifies that the neural network relies on $x_3$, should it be penalized because the original data-generating function depends on $x_1$?

**No—not if the goal is to explain the trained neural network.**

OpenXAI specifically recognizes this problem.

To avoid it in its synthetic generator, SynthGauss creates separated Gaussian clusters with independent features within each cluster. This supports stronger alignment between the known label-generation process and learned model behavior. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf))

## The opportunity for your research

The independence assumption simplifies evaluation.

But real-world tabular datasets frequently contain precisely the dependencies that make explainability difficult.

So you could attempt to retain these dependencies while distinguishing two reference targets:

**Target 1: Explain the known function that generated the labels.**

$$E_{\text{oracle}}(x)=E(g^*,x)$$

**Target 2: Explain the actual predictive model.**

$$E_{\text{model}}(x)=E(f,x)$$

These are different objects.

If a model $f$ learns a different strategy from the fitted analytical function $g^*$, both explanations could be valid within their respective definitions.

I would explicitly distinguish them in your benchmark.

### A potential research question

> Under realistic feature dependencies, how strongly does agreement with the label-generating function predict faithfulness to the trained model?

This is scientifically interesting and directly addresses a problem recognized by OpenXAI.

**My assessment:** There is meaningful room for a benchmark that provides carefully defined mathematical reference explanations alongside the original real-world feature vectors.

But you will need to resolve the distinction between explaining the oracle and explaining the learned model.

------------------------------------------------------------------------

# 4. Paper 3: A Synthetic Ground-Truth Framework (2026)

**Authors:** Miquel Miró-Nicolau, Francesco Spinnato, Riccardo Guidotti.

**My assessment: This is especially relevant because its authors explicitly identify the research direction you want to pursue.**

## What they did

They construct synthetic settings where they can determine ground-truth feature contributions through controlled interventions.

They cover three modalities:

1.  Binary images.
2.  Time series.
3.  Tabular data.

Unlike methods that define ground truth only from the original label-generation rule, they emphasize **model-aligned explanations**.

For example, a component's contribution can be defined by removing or modifying it and measuring the resulting difference in model output.

This provides a reference for evaluating whether explanation methods correctly identify the model's actual sensitivities. ([A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods](https://arxiv.org/pdf/2609.30397))

## Their tabular experiment

This part is particularly relevant.

The authors create an entirely synthetic dataset with:

- Three input features.
- Approximately four million samples.
- A discretized input space.
- A sine-based target function.
- An MLP trained to predict those labels.

The label function uses weighted sine terms with coefficients:

$$w=[0.5,0.25,0.125]$$

They then construct reference feature contributions using controlled input interventions.

The framework does not use the observed feature distribution from a naturally occurring tabular dataset. ([A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods](https://arxiv.org/pdf/2609.30397))

## Their experimental findings

For tabular explanations, their reported similarity scores include:

| Method               |       SIM |
|----------------------|----------:|
| Gradient             | **0.980** |
| LORE                 |     0.784 |
| Integrated Gradients |     0.642 |
| Kernel SHAP          |     0.601 |
| LIME                 |     0.560 |

These scores measure agreement with the paper's specifically defined intervention-based reference—not universal explanation correctness. ([arxiv.org](https://arxiv.org/pdf/2609.30397))

## The gap you could address

Their evaluation is highly controlled but has limited real-world tabular complexity.

In particular, the three-dimensional feature space is very different from a dataset with dozens of mixed features and complex dependencies.

You could examine whether their findings generalize when the input distribution comes from real applications.

That would make a useful extension, particularly if you preserve their intervention metric as one of your benchmark targets.

### The authors explicitly acknowledge this limitation

In their conclusion, they suggest future research involving more realistic data-generation processes and hybrid evaluation strategies combining synthetic ground truth with real-world validation.

That is closely aligned with your research motivation. ([A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods](https://arxiv.org/pdf/2609.30397))

**My assessment:** This provides strong motivation for your paper, but it is not the closest methodological predecessor.

There are other papers that overlap more directly.

------------------------------------------------------------------------

# 5. Two additional papers you absolutely need to consider

My literature search uncovered two papers that change how I would position your contribution.

## A. ITErpretability — NeurIPS 2022

**Benchmarking Heterogeneous Treatment Effect Models through the Lens of Interpretability**

[Read the paper — NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/file/4fd7b4ed13f78b9ba7afcd9d01615896-Paper-Datasets_and_Benchmarks.pdf)

This is probably the most important additional discovery.

Its methodology is strikingly similar to a central part of your proposal.

The authors explicitly:

1.  Take feature vectors from real-world datasets.
2.  Preserve those vectors.
3.  Generate synthetic outcomes using a known mathematical process.
4.  Train black-box models on the resulting semi-synthetic data.
5.  Evaluate whether the models identify the known influential features.

The methodology is illustrated in Figure 1 and formalized in Section 4 of the paper. (<https://papers.nips.cc/paper_files/paper/2022/file/4fd7b4ed13f78b9ba7afcd9d01615896-Paper-Datasets_and_Benchmarks.pdf>)

In other words:

**The combination of real $X$, synthetic $y$, and known important features is not new.**

This is important to establish before investing substantial effort.

### How your work could differ

ITErpretability focuses on heterogeneous treatment effects.

Its principal objective is to identify prognostic and predictive features, particularly features that modify treatment effects.

Your benchmark could be more general:

- Not limited to treatment-effect estimation.
- Fitted to the original real-world target.
- Supporting regression and classification.
- Providing quantitative local attributions, not only known relevant-feature sets.
- Supporting several mathematically defined explanation targets.

The distinction is meaningful, but it needs to be demonstrated rather than merely asserted.

## B. Through the Looking Glass — published online 2023

**Through the Looking Glass: Evaluating Post Hoc Explanations Using Transparent Models**

[Read the full article](https://link.springer.com/article/10.1007/s41060-023-00445-1)

This paper uses **14 real-world tabular datasets** and fits transparent models, including linear regression, logistic regression, decision trees, and Naive Bayes.

It then evaluates the explanations produced by methods including SHAP and LIME against information extracted from those transparent models.

That overlaps with your idea of fitting interpretable functions to existing real-world data. ([Through the looking glass: evaluating post hoc explanations using transparent models \| International Journal of Data Science and Analytics \| Springer Nature Link](https://link.springer.com/article/10.1007/s41060-023-00445-1))

### Where I see a gap

Their evaluation relies on feature-selection and ranking procedures derived from the underlying models.

For linear models, for example, they use coefficient magnitudes to identify influential features.

But a large regression coefficient is not necessarily equivalent to a large local Shapley contribution.

The paper itself discusses that distinction. ([Through the looking glass: evaluating post hoc explanations using transparent models \| International Journal of Data Science and Analytics \| Springer Nature Link](https://link.springer.com/article/10.1007/s41060-023-00445-1))

Your contribution could be to replace such heuristic reference definitions with **precisely specified, mathematically computable local explanations**.

That is a more rigorous basis for a benchmark.

### One more related paper

I also found the 2025 article **Privacy-preserving Ground-truth Data for Evaluating Additive Feature Attribution in Regression Models with Additive CBR and CQV**.

It generates synthetic data inspired by an original aviation dataset, using clustering and feature perturbations. However, it does not simply retain the entire original feature matrix and fit a closed-form target function to the original labels. Its generated target values are based on cluster-specific ranges. ([Privacy-preserving ground-truth data for evaluating additive feature attribution in regression models with additive CBR and CQV](https://www.es.mdu.se/pdf_publications/7277.pdf))

This should also appear in your related-work section.

------------------------------------------------------------------------

# 6. Where I believe the strongest research gaps remain

After examining these papers, I would rank the opportunities as follows.

| Research opportunity | Novelty potential | Importance for your paper |
|----|----|----|
| Real-world inputs + synthetic targets | Low by itself | Necessary foundation |
| Analytical functions fitted to original real labels | Moderate | Important differentiation |
| Mathematically precise local attribution ground truth | **High** | Core methodological contribution |
| Separating oracle explanations from learned-model explanations | **High** | Core validity issue |
| Measuring how explainer rankings change with dataset realism | **High** | Strong experimental contribution |
| Benchmarking the trade-off between target fidelity and analytical tractability | **High** | Potentially publishable methodology |
| Dataset library, pretrained models, and toolkit | Moderate | Valuable supporting contribution |

These assessments reflect potential research contributions, not claims that no existing paper addresses any part of them.

## Gap 1: Ground-truth explanation values on real feature vectors

This is where I would start.

Suppose you have a real dataset:

$$D=\{(x_i,y_i)\}_{i=1}^{n}$$

Fit a generalized additive model:

$$g(x)=\beta_0+\sum_{j=1}^{d}g_j(x_j)$$

The functions $g_j$ could be represented by polynomial terms, splines, or another analytically tractable family.

Then define:

$$y_i^*=g(x_i)$$

You now have a dataset with real-world observations and an exactly known target function.

But here is the important part.

You can also derive **exact interventional Shapley values for this additive function**, using the empirical feature distribution as the background.

For feature $j$:

$$\boxed{
\phi_j(x)
=
g_j(x_j)
-
\frac{1}{n}
\sum_{i=1}^{n}g_j(x_{ij})
}$$

This is an exact expression for interventional Shapley values of the additive function under the specified empirical background distribution.

It remains valid even if the original feature vectors contain correlations, because the interventional definition intentionally does not preserve dependencies between present and missing features.

That's a useful property for building a benchmark.

It means that you have:

- Actual real-world feature observations.
- A known mathematical target.
- Exact local attributions under an explicit definition.
- No need to estimate those attributions using SHAP itself.

**That last point is important.** You would not want the benchmark's ground truth to depend on the same approximate explanation algorithm that you are trying to evaluate.

### An extension that could make this much more interesting

Include pairwise interactions:

$$g(x)=\beta_0+
\sum_j g_j(x_j)
+
\sum_{j<k}g_{jk}(x_j,x_k)$$

For these low-order models, exact interventional Shapley values can still be calculated efficiently by decomposing the contributions of the individual components.

You could therefore offer a progression of benchmark difficulty:

| Level | Target function                      | Explanation difficulty |
|-------|--------------------------------------|------------------------|
| 1     | Linear additive                      | Easy                   |
| 2     | Nonlinear additive                   | Moderate               |
| 3     | Nonlinear with pairwise interactions | Harder                 |
| 4     | Sparse higher-order interactions     | Hard                   |
| 5     | More complex symbolic functions      | Depends on complexity  |

The important requirement is that each level has a **mathematically justified explanation oracle**.

One qualification: exact interventional Shapley values are not the same as exact observational or conditional Shapley values. The latter depend on conditional feature distributions, which are generally not known exactly for arbitrary real-world datasets.

I would make that distinction explicit rather than promise exact ground truth for every attribution definition.

------------------------------------------------------------------------

## Gap 2: How well can an analytical target approximate a real task?

This is potentially a very interesting contribution that I did not see developed as a general-purpose benchmark in the three original papers.

Consider two possible functions fitted to a real dataset.

A linear model:

$$g_1(x)=\beta_0+\sum_j\beta_jx_j$$

And a more expressive analytical model:

$$g_2(x)=\beta_0+
\sum_jg_j(x_j)+
\sum_{j<k}g_{jk}(x_j,x_k)$$

The first is easier to explain exactly.

The second may better reproduce the original labels.

This creates an important trade-off:

**How much target realism can we retain while preserving exact mathematical explanations?**

You could measure the predictive fidelity of each analytical approximation against the original labels using held-out data.

For regression, that might involve:

$$R^2(g(X_{\text{test}}),y_{\text{test}})$$

For classification, appropriate metrics could include cross-entropy, calibration, and discrimination.

Then investigate whether higher predictive fidelity produces a more realistic XAI evaluation setting.

You could introduce a **realism–tractability frontier**.

Conceptually:

$$\text{Predictive fidelity to real labels}
\quad\text{vs.}\quad
\text{Complexity of exact attribution}$$

This would be substantially more interesting than simply fitting one symbolic-regression model per dataset.

It would also help researchers choose the appropriate benchmark difficulty for their methods.

------------------------------------------------------------------------

## Gap 3: Do benchmark rankings generalize to realistic datasets?

This is my favorite experimental direction.

Suppose you compare explanation methods across three environments.

**Environment A: Conventional synthetic data**

$$X\sim\mathcal N(0,\Sigma)$$

**Environment B: Real-world feature vectors with an analytical target**

$$X=X_{\text{real}},\qquad y^*=g(X)$$

**Environment C: The original real-world task**

$$(X_{\text{real}},y_{\text{real}})$$

Environment A has a controlled feature distribution.

Environment B has authentic observed feature structures and a known analytical target.

Environment C is the original problem, but generally lacks an explanation oracle.

You could evaluate SHAP, LIME, Integrated Gradients, and other methods across the three environments, using precisely defined metrics appropriate to each setting.

Then ask:

> Does performance on traditional synthetic benchmarks predict performance on realistic semi-synthetic benchmarks?

And perhaps more importantly:

> Which characteristics of a dataset cause explanation-method rankings to change?

Candidate factors include:

- Correlation strength.
- Nonlinear dependencies.
- Number of categorical features.
- Class imbalance.
- Feature redundancy.
- Interaction complexity.

You could study rank correlations between benchmark results in A and B, stratified by these characteristics.

That would provide a direct empirical answer to the motivation behind your project.

A crucial methodological point: you could not calculate an absolute explanation-accuracy metric for Environment C without an additional source of ground truth. However, you could still examine perturbation-based metrics, stability, prediction fidelity, and other complementary diagnostics.

**A finding that conventional synthetic benchmarks systematically misrank explanation methods under realistic dependencies would be a strong result.**

------------------------------------------------------------------------

## Gap 4: Distinguishing the label-generating function from the learned model

This may be the hardest theoretical challenge.

Suppose you fit an analytical model:

$$g(x)=x_1^2+3x_2$$

You generate benchmark labels using $g$.

Then you train a neural network:

$$f_\theta(x)\approx g(x)$$

You know the explanations of $g$.

But are they necessarily the correct explanations of $f_\theta$?

**No.**

This is especially problematic when the dataset contains redundant or correlated features.

A neural network may exploit proxy variables, even when its predictions are accurate.

Therefore, I would structure the benchmark into two evaluation regimes.

### Regime A: Oracle-model evaluation

The explanation method directly explains $g$.

Its explanation can be compared against the exact reference explanation of $g$.

This isolates errors introduced by the explanation method.

### Regime B: Learned-model evaluation

The explanation method explains a separately trained $f_\theta$.

The benchmark reports both:

$$\underbrace{L(f_\theta,g)}_{\text{Prediction mismatch}}$$

and an appropriately defined measure of explanation discrepancy.

A discrepancy between an explanation of $f_\theta$ and the reference explanation of $g$ cannot automatically be attributed to the explainer.

It may instead reflect the model's learned behavior.

If you explicitly control and analyze this distinction, your benchmark could avoid a significant evaluation pitfall.

This issue is already recognized by OpenXAI and the 2026 paper, which makes it important to address rigorously rather than claim as an entirely new observation.

------------------------------------------------------------------------

# 7. The paper I would actually propose

Based on this review, I would change the framing slightly.

### Proposed title

**RealXAI-Bench: Exact Feature Attribution Benchmarks from Real-World Data Distributions and Analytically Tractable Targets**

### Central research question

> Can we construct realistic XAI benchmarks by retaining observed real-world feature distributions, approximating real targets with analytically tractable functions, and deriving mathematically precise reference explanations?

### Proposed contributions

I would aim for four substantial contributions.

**1. A general-purpose benchmark construction methodology**

A reproducible method that accepts a real dataset, fits an analytical target function, and constructs semi-synthetic labels.

The system should evaluate the approximation's fidelity against the original task rather than merely generating an arbitrary mathematical function.

**2. A library of analytically defined reference explanations**

Support exact local explanations for selected function classes and explanation definitions.

For instance:

- Exact analytical gradients.
- Exact interventional Shapley values.
- Exact integrated gradients for supported functions and baselines.
- Known functional interactions.

Clearly document the assumptions and mathematical definitions for each.

**3. A systematic experimental study of realism versus explainability accuracy**

Compare explanation methods across traditional synthetic datasets and real-feature semi-synthetic datasets.

Measure whether their relative performance changes, and identify which data characteristics explain the changes.

**4. A reusable benchmark and toolkit**

Release real-feature benchmark datasets, analytical target functions, attribution oracles, pretrained baseline models, evaluation metrics, and reproducible scripts.

I would probably start with 10–20 public tabular datasets spanning different domains and feature characteristics.

One additional consideration is dataset licensing: some datasets permit research use but not unrestricted redistribution, so the toolkit may need to distribute transformation scripts instead of the original records.

------------------------------------------------------------------------

# 8. What I would validate before investing months

I would conduct a small feasibility study before building the full toolkit.

Take three datasets with different characteristics:

| Dataset | Why include it? |
|----|----|
| Wine Quality | Mostly numerical features; useful comparison with XAI-Bench |
| Adult Income | Mixed numerical and categorical features |
| California Housing | Continuous variables with nonlinear dependencies and geographical structure |

For each dataset, fit three analytical function families: linear, nonlinear additive, and nonlinear additive with pairwise interactions.

Then measure:

**A. Target approximation quality:** How accurately does each analytical function reproduce the original labels on held-out samples?

**B. Attribution correctness:** Can you compute exact reference explanations under clearly stated definitions?

**C. Benchmark difficulty:** Do established explanation methods produce meaningfully different scores across the analytical function families?

**D. Realism sensitivity:** Do those scores or method rankings change when replacing real feature vectors with Gaussian-simulated feature vectors?

The most important result would be **D**.

If you find that the explanation-method rankings are essentially unchanged, the primary motivation for the benchmark becomes weaker.

Conversely, if the rankings change substantially and consistently, you have an empirically demonstrated reason why the community needs your benchmark.

------------------------------------------------------------------------

# 9. My overall assessment

Having examined these papers, I would distinguish three claims.

| Claim | Assessment |
|----|----|
| Nobody has combined real-world feature vectors with synthetic targets. | **Incorrect.** ITErpretability already does this. |
| Nobody has evaluated explanation methods using fitted transparent models on real datasets. | **Incorrect.** Through the Looking Glass does this. |
| There is room for a general benchmark combining real-data-fitted analytical targets, rigorously defined local explanation oracles, and systematic evaluation of the realism–ground-truth trade-off. | **Promising, subject to a fuller novelty review.** |

The third claim is where I would concentrate your effort.

### My recommendation

I would not begin by collecting 30 datasets or implementing 15 explanation algorithms.

I would first try to establish the following result:

**Existing synthetic benchmarks can produce misleading comparisons of XAI methods because they do not reproduce the dependency structures and other characteristics of real-world feature distributions.**

Then demonstrate that your construction methodology allows researchers to evaluate those methods against mathematically defined reference explanations while retaining the observed real-world input structure.

That gives you a scientific contribution alongside the dataset and software.

**If you can establish that result convincingly, I think the project becomes a credible candidate for the NeurIPS Evaluations & Datasets track.** If the benchmark construction itself also involves a genuinely new algorithm or mathematical result, it could support a main-track submission.

The most important next step is to test whether those ranking differences actually occur. That experiment will tell you much more about the paper's potential than building the toolkit first.

<details><summary>Sources cited in this answer</summary>

- [Synthetic Benchmarks for Scientific Research in Explainable Machine Learning | Scholar Feed](https://www.scholarfeed.org/paper/2106.12543)
- [Synthetic Benchmarks for Scientific Research in Explainable Machine Learning · mlsanity](https://mlsanity.com/paper/2106.12543/)
- [datasets-benchmarks-proceedings.neurips.cc](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round2.pdf)
- [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf)
- [A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods](https://arxiv.org/pdf/2609.30397)
- [https://papers.nips.cc/paper_files/paper/2022/file/4fd7b4ed13f78b9ba7afcd9d01615896-Paper-Datasets_and_Benchmarks.pdf](https://papers.nips.cc/paper_files/paper/2022/file/4fd7b4ed13f78b9ba7afcd9d01615896-Paper-Datasets_and_Benchmarks.pdf)
- [Through the looking glass: evaluating post hoc explanations using transparent models | International Journal of Data Science and Analytics | Springer Nature Link](https://link.springer.com/article/10.1007/s41060-023-00445-1)
- [Privacy-preserving ground-truth data for evaluating additive feature attribution in regression models with additive CBR and CQV](https://www.es.mdu.se/pdf_publications/7277.pdf)

</details>

---

## Question 3

> Btw, what if we change the modality? E.g. some explanation staff for images or text? Does that even make sense here

### ChatGPT

**Yes. In fact, I think extending your idea to images or text could be more interesting than restricting it to tabular data.** Particularly for text, I see a potentially compelling research direction.

However, there is an important complication: **for images and text, a mathematically known prediction function does not automatically give you a meaningful explanation at the pixel or token level.**

I looked into existing research on this, and there are already several relevant benchmarks. Let me explain where I see opportunities.

# 1. The fundamental idea works across modalities

Your original proposal is essentially:

1.  Take a real-world dataset.
2.  Fit a mathematically tractable approximation of the target function.
3.  Use that approximation to generate new labels.
4.  Derive ground-truth explanations.
5.  Benchmark XAI methods against those explanations.

Mathematically:

$$D=\{(x_i,y_i)\}_{i=1}^{n}$$

Fit:

$$g^*=\arg\min_{g\in\mathcal G}\sum_i L(g(x_i),y_i)$$

Generate:

$$y_i^*=g^*(x_i)$$

And derive reference explanations:

$$E^*(x)=\operatorname{Explain}(g^*,x)$$

**Nothing here inherently requires tabular data.**

The challenge is constructing a function class $\mathcal G$ that simultaneously provides:

- Meaningful predictions on real data.
- Sufficient complexity to challenge modern explanation methods.
- Mathematically defined reference explanations.

These requirements become harder to satisfy for unstructured inputs.

------------------------------------------------------------------------

# 2. Text: I find this particularly promising

Imagine taking IMDb movie reviews.

You have real sentences, with real linguistic structure, sentiment, negation, and contextual dependencies.

For example:

> The acting was excellent, but the story was disappointing and the ending was terrible.

Now suppose you fit an interpretable sentiment function.

A simple version might be:

$$g(x)=\beta_0+
\sum_j \beta_j \phi_j(x)$$

Where $\phi_j(x)$ represents an interpretable linguistic feature.

For example:

| Linguistic feature                 | Contribution |
|------------------------------------|-------------:|
| "excellent"                        |         +1.5 |
| "disappointing"                    |         -1.2 |
| "terrible"                         |         -2.0 |
| Positive term preceded by negation |         -1.0 |
| Negative sentiment after "but"     |         -0.8 |

These values are illustrative; in the actual project, you would estimate them from the original labels.

The important part is that the features could include:

- Words and phrases.
- Negation patterns.
- Syntactic constructions.
- Sentiment-bearing expressions.
- Interactions between linguistic features.

You could construct an interpretable model with nonlinear interactions:

$$g(x)=\beta_0+
\sum_j g_j(\phi_j(x))+
\sum_{j<k}g_{jk}(\phi_j(x),\phi_k(x))$$

Then you would have real natural-language inputs and an analytically tractable target.

### What could you explain?

Suppose your fitted function assigns a negative contribution to:

> "the ending was terrible"

You would know exactly which linguistic features contributed to the synthetic prediction.

You could evaluate whether LIME, SHAP, or another explanation method identifies those features correctly.

You could also introduce progressively harder cases.

For example:

**Simple sentence:**

> The movie was terrible.

**Negation:**

> The movie was not terrible.

**Contrast:**

> The acting was excellent, but the movie was terrible.

**Complex dependency:**

> Despite its excellent performances, the film fails to deliver a compelling story.

The same methodology could be applied to all these cases.

But the reference explanations would need to be defined at a chosen level—such as words, spans, or linguistic features. An exact contribution assigned to a phrase does not automatically imply a unique contribution for each subword token.

## Has something similar been done?

Yes, but there is an interesting distinction.

### ERASER — ACL 2020

The paper *ERASER: A Benchmark to Evaluate Rationalized NLP Models* introduced a collection of NLP datasets with human-annotated rationales.

For example, annotators identify the words that support a classification.

The benchmark evaluates both agreement with human rationales and different measures of faithfulness. ([ERASER: A Benchmark to Evaluate Rationalized NLP Models - ACL Anthology](https://aclanthology.org/2020.acl-main.408/))

**The limitation relevant to your idea:** A human-annotated explanation is not necessarily the actual explanation of a model's prediction.

A model could make the correct prediction for entirely different reasons.

This distinction between human plausibility and model faithfulness is well established in NLP research. ([Towards Faithful Model Explanation in NLP: A Survey \| Computational Linguistics \| MIT Press](https://direct.mit.edu/coli/article/50/2/657/119158/Towards-Faithful-Model-Explanation-in-NLP-A-Survey))

Your methodology offers an alternative: evaluate explanations against a mathematically defined function rather than relying exclusively on human-annotated rationales.

That is potentially valuable.

However, I would not claim novelty yet. We would need a more targeted literature review of semi-synthetic NLP attribution benchmarks.

------------------------------------------------------------------------

# 3. Images: Also feasible, but considerably harder

Let's consider a real image dataset, such as MS COCO.

COCO contains natural images with annotated objects, including segmentation masks.

For example, an image might contain:

- A person.
- A bicycle.
- A car.
- Trees.
- Buildings.

Unlike synthetic image datasets, these are photographs of real scenes. ([coco  \|  TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/coco))

Imagine defining interpretable features:

$$z(I)=
\begin{bmatrix}
\text{number of people}\\
\text{number of bicycles}\\
\text{area occupied by vehicles}\\
\text{person-bicycle interaction}
\end{bmatrix}$$

Then fit:

$$g(I)=\beta_0+\sum_j\beta_jz_j(I)$$

You could use the original images while generating labels from the fitted analytical function.

For example, the output might represent a transportation-related score.

You would know the contribution of each semantic feature to the prediction.

And because COCO provides segmentation annotations, you could associate semantic features with image regions.

### What would the benchmark look like?

For each image, you could provide:

| Component             | Ground truth                        |
|-----------------------|-------------------------------------|
| Original photograph   | Real-world image                    |
| Synthetic target      | Calculated from analytical function |
| Relevant concepts     | Known                               |
| Concept contributions | Known                               |
| Object masks          | From segmentation annotations       |
| Expected explanation  | Defined at the concept level        |
| Trained CNN/ViT       | Available for evaluation            |

Researchers could then benchmark methods such as Grad-CAM, Integrated Gradients, and occlusion-based explanations.

### But there is a fundamental problem

Consider this function:

$$g(I)=2z_{\text{person}}(I)+3z_{\text{bicycle}}(I)$$

You know that the bicycle contributes three units.

However, **you do not automatically know how those three units should be distributed across individual pixels**.

For example:

- Should every bicycle pixel receive equal importance?
- Should the wheels matter more than the frame?
- Should pixels surrounding the bicycle receive importance?
- What if a pretrained neural network uses contextual cues to identify the bicycle?

A segmentation mask can tell you where an object is, but it does not define the exact attribution of each pixel.

This is the key difficulty.

Your benchmark could have exact **concept-level** explanations while lacking exact **pixel-level** explanations.

That is perfectly acceptable, but the paper must make this distinction explicit.

One alternative is to construct the analytical function directly from pixel statistics, such as region-wise mean intensities. That can yield exact pixel-level derivatives, but the resulting prediction task may be much less representative of real-world computer vision.

------------------------------------------------------------------------

# 4. Existing research on image benchmarks

This field is more developed than I initially expected.

Four papers deserve particular attention.

| Paper | Venue/year | Main approach |
|----|----|----|
| **CLEVR-XAI** | Information Fusion, 2022 | Rendered scenes with known relevant objects and segmentation masks |
| **FunnyBirds** | ICCV 2023 | Synthetic bird images with controllable semantic parts |
| **Assessing Fidelity in XAI Post-hoc Techniques** | Artificial Intelligence, 2024 | Synthetic image tasks with defined explanation ground truth |
| **Saliency-Bench** | KDD 2025 | Natural-image datasets with annotated explanation regions |

### FunnyBirds — ICCV 2023

This is especially relevant.

The authors construct synthetic bird images where different parts have controlled semantic significance.

They can remove parts and measure changes in model predictions.

This allows systematic evaluation of explanation methods using meaningful interventions. ([FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods](https://openaccess.thecvf.com/content/ICCV2023/papers/Hesse_FunnyBirds_A_Synthetic_Vision_Dataset_for_a_Part-Based_Analysis_of_ICCV_2023_paper.pdf))

The limitation is that the birds are synthetic.

Your proposed approach could instead use natural photographs.

However, performing meaningful interventions on natural images without introducing artifacts is considerably harder.

### CLEVR-XAI

CLEVR-XAI provides known explanation masks for visual question answering.

For instance, if a question asks about the color of a particular object, the benchmark knows which objects are relevant.

But its images are computer-generated scenes rather than natural photographs. ([CLEVR-XAI: A benchmark dataset for the ground truth evaluation of neural network explanations - ScienceDirect](https://doi.org/10.1016/j.inffus.2021.11.008))

### Saliency-Bench — KDD 2025

This is particularly important for assessing novelty.

Saliency-Bench already uses real-world images and ground-truth explanation annotations.

It includes eight datasets spanning tasks such as object classification, scene classification, medical images, and action classification.

It also supplies evaluation infrastructure and compares multiple visual explanation methods.

**So a benchmark consisting of real images, explanation masks, models, and evaluation tools is not novel on its own.**

Your potential distinction would be mathematically specified attribution targets rather than relying primarily on annotations identifying relevant regions. ([Saliency-Bench: A Comprehensive Benchmark for Evaluating Visual Explanations](https://pmc.ncbi.nlm.nih.gov/articles/PMC13499442/))

There is also earlier work, *Benchmarking Attribution Methods with Relative Feature Importance* (2019), which constructs composite images from real-world image datasets. This further narrows what can be claimed as new. ([bam/README.md at master · google-research-datasets/bam · GitHub](https://github.com/google-research-datasets/bam/blob/master/README.md))

------------------------------------------------------------------------

# 5. The most important theoretical issue

Across all three modalities, I think you need to distinguish two ideas:

**A. Knowing the correct explanation of the synthetic target-generating function.**

**B. Knowing the correct explanation of a neural network trained on the synthetic targets.**

These are not necessarily equivalent.

Suppose:

$$y^*=g(x)$$

And you train:

$$f_\theta(x)\approx g(x)$$

Even when:

$$f_\theta(x)\approx g(x)$$

their explanations may differ substantially.

This is particularly important for images and language because large neural networks may exploit correlations that the analytical function does not explicitly use.

Therefore, I would design two benchmark regimes:

**Oracle regime:** Explain the known analytical model directly. This allows comparison against exact reference attributions.

**Learned-model regime:** Explain a neural network trained on the oracle's labels. This tests explanation behavior on realistic architectures, but oracle disagreement alone cannot establish explainer error.

I would also be precise about what realism means.

Preserving real inputs gives you a real-world marginal distribution:

$$P_{\text{benchmark}}(X)=P_{\text{real}}(X)$$

at least at the level of the retained observations.

But generating new labels changes the conditional distribution:

$$P_{\text{benchmark}}(Y\mid X)
\neq
P_{\text{real}}(Y\mid X)$$

in general.

This is why your idea of **fitting analytical functions to the original labels** is important. It provides a way to retain some of the original prediction task's structure rather than introducing an arbitrary synthetic task.

------------------------------------------------------------------------

# 6. Which modality would I choose?

Here is how I would compare the options.

| Criterion | Tabular | Text | Images |
|----|----|----|----|
| Preserve authentic input data | Easy | Easy | Easy |
| Fit interpretable target functions | Easy | Moderate | Hard |
| Derive exact explanations | Easy for selected definitions | Feasible for defined linguistic features | Feasible at concept level; harder at pixel level |
| Realistic prediction tasks | Strong | Potentially strong | Challenging |
| Existing benchmark competition | High | Significant | Very high |
| Computational requirements | Low | Low–moderate | Moderate–high |
| Potential research impact | High | **Very high** | Very high |

These are my qualitative judgments, not measured comparisons.

### My preference would be text

I see an attractive combination of:

1.  Real-world complexity.
2.  Relatively inexpensive experiments.
3.  Mathematically tractable models based on linguistic features.
4.  Direct relevance to modern NLP explainability research.
5.  A clear distinction from benchmarks based primarily on human rationales.

It would also be relatively straightforward to scale from simple lexical contributions to phrase interactions, negation, and more complex linguistic structures.

The hardest part would be showing that the analytical approximation genuinely captures interesting characteristics of the real classification problem.

If the model is effectively counting positive and negative words, reviewers could reasonably question whether the benchmark represents the challenges of modern language understanding.

That is something you would need to address experimentally.

------------------------------------------------------------------------

# 7. A potentially stronger paper: a modality-independent framework

There is another possibility.

Instead of publishing separate benchmarks, you could introduce a general method:

**RealGroundXAI: Constructing Ground-Truth Explainability Benchmarks from Real-World Data Across Modalities**

The framework would take a dataset and construct a known prediction function over interpretable representations.

Conceptually:

$$\boxed{
x
\xrightarrow{\text{Feature representation}}
z(x)
\xrightarrow{\text{Analytical function}}
y^*
}$$

Where:

- For tabular data, $z(x)$ consists of the original features.
- For text, $z(x)$ consists of meaningful linguistic features.
- For images, $z(x)$ consists of semantic or regional visual features.

You would then compute ground-truth explanations for the analytical function in the corresponding feature space.

The contribution would be a **general framework for constructing reference explanations from real-world datasets**.

However, a modality-independent framework would have to handle feature extraction carefully. A black-box image encoder or language encoder would not give you exact input-level ground truth merely because the final function is analytical.

Furthermore, the September 2026 paper we discussed already proposes a synthetic-ground-truth framework across binary images, tabular data, and time series. Your contribution would need to distinguish itself clearly by its reliance on original real-world observations, fitted target functions, and reference-attribution definitions. ([A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods](https://arxiv.org/abs/2609.30397))

I would therefore begin with one modality and expand only after demonstrating that the methodology works.

------------------------------------------------------------------------

## 8. A research direction I find especially interesting

There is one experiment I would be particularly excited to see.

**Can the same explanation method perform well on synthetic text but poorly on natural text, even when the underlying target function is identical?**

Imagine constructing two groups:

**Group A:** Artificial sentences generated from controlled linguistic templates.

**Group B:** Actual human-written reviews.

Both use the same underlying analytical target function.

This isolates an important variable: the complexity of the input distribution.

You could then evaluate whether LIME, SHAP, or gradient-based attribution methods produce systematically different results as linguistic complexity increases.

And because the target function is analytically defined, you have a reference for measuring the discrepancies.

The same experiment could be conducted with synthetic versus natural images.

I find this especially attractive because it directly tests your original hypothesis:

> Synthetic data may produce misleading conclusions about explainability methods because its input distribution does not reproduce real-world complexity.

Rather than simply assuming that this is true, **you could test it experimentally**.

If it turns out to be true, and the differences are substantial, it provides a strong justification for the benchmark.

If it is not true, that is scientifically informative too.

### My recommendation

I would initially pursue **text-based classification**, using real reviews and fitted analytical functions with increasingly complex linguistic interactions.

I would retain tabular datasets as a relatively inexpensive validation modality.

Images would be a natural extension, but I would not begin there because establishing meaningful pixel-level ground truth is substantially more difficult.

The strongest version of the paper would not simply introduce another explanation benchmark. It would demonstrate **when and why existing XAI evaluation conclusions fail to generalize from synthetic inputs to real-world data**.

That is a considerably more compelling research question for NeurIPS than the dataset alone.

<details><summary>Sources cited in this answer</summary>

- [ERASER: A Benchmark to Evaluate Rationalized NLP Models - ACL Anthology](https://aclanthology.org/2020.acl-main.408/)
- [Towards Faithful Model Explanation in NLP: A Survey | Computational Linguistics | MIT Press](https://direct.mit.edu/coli/article/50/2/657/119158/Towards-Faithful-Model-Explanation-in-NLP-A-Survey)
- [coco  |  TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/coco)
- [FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods](https://openaccess.thecvf.com/content/ICCV2023/papers/Hesse_FunnyBirds_A_Synthetic_Vision_Dataset_for_a_Part-Based_Analysis_of_ICCV_2023_paper.pdf)
- [CLEVR-XAI: A benchmark dataset for the ground truth evaluation of neural network explanations - ScienceDirect](https://doi.org/10.1016/j.inffus.2021.11.008)
- [Saliency-Bench: A Comprehensive Benchmark for Evaluating Visual Explanations](https://pmc.ncbi.nlm.nih.gov/articles/PMC13499442/)
- [bam/README.md at master · google-research-datasets/bam · GitHub](https://github.com/google-research-datasets/bam/blob/master/README.md)
- [A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods](https://arxiv.org/abs/2609.30397)

</details>
