---
title: "Feature Effects as Behavioral Specifications for Mechanistic Analysis of Tabular Foundation Models"
date-modified: "2026-08-05"
primary-contribution: "methodological"
secondary-contribution: "empirical"
keywords: ["interpretable machine learning", "mechanistic interpretability", "tabular foundation models", "feature interactions"]
---

## 1. Summary

Tabular foundation models such as TabPFN do not fit new parameters for every dataset. Instead, they receive labeled observations as context and produce predictions for new observations in a forward pass. This makes them learned learning algorithms: the same network can implement different prediction functions depending on the supplied dataset. Existing model-agnostic interpretation methods can describe these functions through feature effects, feature importance, and feature interactions, while mechanistic interpretability asks how internal representations and components implement a particular behavior.

This thesis investigates whether model-agnostic explanations can serve as behavioral specifications and evaluation instruments for mechanistic analysis. A feature effect or interaction is first characterized at the model-output level. Internal methods then locate associated representations or components and test whether the proposed mechanism accounts specifically for the measured effect. The core uses controlled synthetic tasks and one open, inspectable model. The study could clarify when external explanations provide useful targets for reverse engineering, how feature interactions emerge through a tabular Transformer, and how stable the resulting mechanism claims are across context datasets.

## 2. Motivation and problem

Tabular Prior-data Fitted Networks (TabPFNs) are pretrained across many synthetic datasets and use in-context learning to predict on a new table without task-specific parameter optimization. TabPFN v2 supports classification and regression and alternates attention across observations and features, making the dataset itself part of the model input ([Hollmann et al., 2025](https://doi.org/10.1038/s41586-024-08328-6)). For fixed network parameters \(\theta\), a context dataset

\[
D=\{(x_i,y_i)\}_{i=1}^n
\]

therefore induces a dataset-specific prediction function

\[
g_D(x)=f_\theta(D,x).
\]

Model-agnostic interpretable machine learning (IML) studies properties of \(g_D\) through its inputs and outputs. Existing work has adapted individual conditional expectation, partial dependence, accumulated local effects (ALE), Shapley values, and leave-one-covariate-out importance to TabPFN ([Rundel et al., 2024](https://doi.org/10.1007/978-3-031-63797-1_23)). These methods can establish, for example, that the effect of \(x_j\) changes with \(x_k\), or that an interaction is present only in a region of the feature space. They do not by themselves identify how the network represents or computes that behavior.

Mechanistic interpretability instead seeks an implementation-level account: which hidden representations, attention heads, multilayer perceptrons, or pathways carry out a specified computation, and how those parts combine to produce the observed behavior. Recent TFM studies have begun to locate representations and readout mechanisms across layers. One study identifies distinct similarity-based readouts in TabPFN, TabICL, and Mitra ([Biloš et al., 2026](https://arxiv.org/abs/2605.21288)); another uses activation patching and head ablation on two synthetic TabPFN regression tasks ([Gupta et al., 2026](https://arxiv.org/abs/2606.12917)). These results show that mechanistic work on TFMs is viable. They also leave room for more behavior-specific evaluation than changes in prediction error or output recovery alone.

The proposed bridge starts with a model-agnostic behavioral specification. For instance, an ALE main-effect curve for feature \(j\) and context \(D\) can be written conceptually as

\[
A_{j,D}(z)=\int_{z_0}^{z}
\mathbb{E}\!\left[
\frac{\partial g_D(X)}{\partial x_j}\,\middle|\,X_j=t
\right]dt-c_D,
\]

with finite differences used in practice. ALE avoids the unconditional extrapolation of partial dependence when features are dependent ([Apley and Zhu, 2020](https://doi.org/10.1111/rssb.12377)). Pairwise Shapley–Taylor indices offer a local decomposition of a prediction into main and interaction contributions ([Sundararajan et al., 2020](https://proceedings.mlr.press/v119/sundararajan20a.html)). GADGET recursively partitions the feature space to find regions in which local feature effects are more homogeneous and can therefore specify a regional interaction behavior ([Herbinger et al., 2024](https://jmlr.org/papers/v25/23-0699.html)). These methods define different targets and need not agree.

Let \(B(g_D)\) denote one selected behavioral object, such as an effect curve, an interaction index, or a regional effect contrast. For a proposed internal component set \(C\), let \(g_D^{(-C)}\) denote the network under a defined component modification. A basic evaluation quantity is

\[
\Delta_B(C,D)=d\!\left(B(g_D),B(g_D^{(-C)})\right),
\]

where \(d\) is appropriate to the specification, such as integrated curve distance or change in interaction strength. A useful mechanistic account should explain a substantial share of the target behavior while preserving relevant non-target behavior. Prediction performance remains important, but it is too coarse to distinguish selective removal of an interaction from general model damage.

This framework makes no causal claim about real-world features or the data-generating process. It may, however, use controlled interventions inside the network, such as activation patching or ablation, to test implementation-level claims. Without such internal validation, the work would remain representational analysis rather than strong mechanistic interpretability.

## 3. Objective and scope

### Primary objective

Develop and assess a framework in which model-agnostic feature effects or feature interactions define measurable behavioral targets for identifying and evaluating internal mechanisms in a tabular foundation model.

### Core scope (minimum viable thesis)

- Study one open and inspectable TFM on a small collection of controlled binary-classification or regression tasks containing additive effects and one clearly defined pairwise or regional interaction.
- Select one primary behavioral specification, such as ALE with GADGET or pairwise Shapley–Taylor interaction values, and combine it with at least one internal localization method and one internal validation method.
- Evaluate behavioral specificity and stability across independently sampled context datasets, while treating ordinary predictive performance as a secondary diagnostic.

The thesis does not need to provide a complete circuit-level decomposition of a production-scale TabPFN. A selective and reproducible account of one feature effect or interaction is a sufficient core.

### Possible extensions

- Compare whether the identified high-level mechanism transfers from a small model to TabPFN v2, TabICL, or another open TFM architecture.
- Extend the analysis from synthetic tasks to a small number of real datasets, where the goal would be robustness of the model explanation rather than recovery of a known functional structure.

## 4. Potential research questions

1. **Potential RQ1 — Behavioral specification:** Which local, global, or regional feature-effect or interaction estimands provide sufficiently precise and stable targets for mechanistic analysis of a context-dependent tabular predictor?
2. **Potential RQ2 — Internal implementation:** At which layers and in which components does a selected main effect or interaction become represented, and how well do proposed internal mechanisms account for that behavior without broadly degrading unrelated behavior?
3. **Potential RQ3 — Context generalization:** To what extent is the proposed mechanism reused across independently sampled context datasets, interaction strengths, feature distributions, or regions of the feature space?

Initial literature study and pilot experiments may lead the student and supervisor to refine, merge, or replace these questions. Plausible but nonbinding hypotheses include that interaction information emerges later than additive main effects, that different context datasets recruit partially different components, and that regional specifications yield more selective mechanism evaluations than a single global interaction score.

## 5. Possible methodological directions

A suitable empirical setting is a family of low-dimensional synthetic tasks for which the predictive structure is controlled. One candidate binary-classification family is

\[
\Pr(Y=1\mid X)=\sigma\!\left(
h_1(X_1)+h_2(X_2)+
\gamma X_1X_2\mathbf{1}\{X_3>t\}
\right),
\]

where \(\gamma=0\) gives an additive control and \(\gamma\ne0\) introduces a regional interaction. Variants could change interaction strength, context size, correlation, redundant features, or label noise. This supplies a known functional structure at the data level, but not a ground-truth neural mechanism. Several independently sampled contexts per setting are needed because \(g_D\), and therefore its explanations, vary with \(D\).

The smallest practical model candidate is nanoTabPFN, a compact implementation of the TabPFN v2 architecture that can be pretrained quickly and modified directly ([Pfefferle et al., 2025](https://arxiv.org/abs/2511.03634); [code](https://github.com/automl/nanoTabPFN)). A released TabPFN checkpoint is another candidate if its architecture, license, and computational cost fit the final design. Using a small model first permits broader component scans and exact feature-interaction calculations before any selective confirmation on a larger model.

At the behavioral level, the student could choose one coherent combination rather than an exhaustive collection. ALE can describe average feature effects under dependent features; GADGET can identify regions with heterogeneous effects; and Shapley–Taylor indices can attribute local pairwise interactions. Partial dependence and Friedman’s \(H\)-statistic may serve as contrasts where their estimands are compatible; their interaction-related limitations are discussed by [Herbinger et al. (2024)](https://jmlr.org/papers/v25/23-0699.html). The reference distribution, prediction scale, and aggregation rule must be stated explicitly because changing them changes the behavior being explained. Methods that directly build Shapley-style explanations into a PFN, such as ShapPFN, provide an adjacent comparison but answer a different question: they change the architecture to expose attributions rather than reverse-engineering an existing model ([Sena and Azevedo, 2026](https://arxiv.org/abs/2603.29946)).

Candidate internal analyses include layerwise probes, representation similarity, attention-head or multilayer-perceptron attribution, and additive decomposition of hidden states. DePass is one possible forward-decomposition method that propagates user-defined additive components through a Transformer while fixing nonlinear routing quantities ([Hong et al., 2025](https://arxiv.org/abs/2510.18462)). Probing or decomposition can identify candidate locations, but neither establishes that a representation is used. Activation patching, targeted ablation, or another matched internal modification can provide stronger validation. Automated circuit discovery offers one established workflow for selecting a behavior metric, localizing components through patching, and searching for relevant connections ([Conmy et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html)), although a full circuit search is optional here.

Primary conclusions should use behavior-specific outcomes: changes in effect-curve shape, regional contrasts, or interaction magnitude; preservation of designated non-target effects; and reproducibility across contexts. Predictive loss, calibration, attention entropy, and raw output recovery are useful secondary diagnostics. A candidate mechanism that changes the target interaction only by destroying the predictor should not count as a successful explanation.

Important choices left to the student include the behavioral estimand, the internal unit of analysis, and the generalization target. The student should also justify whether to prioritize a transparent small model or stronger evidence on a larger released checkpoint. These choices should be fixed after initial reading and pilot work and before the main comparative analysis.

## 6. Feasibility and validity considerations

- **Behavior versus mechanism:** Recovering the known synthetic interaction shows that the model learned a behavior, not how it implemented it. Internal localization and a behavior-specific validation step are essential. If component-level localization proves unstable, a defensible fallback is a coarser layer- or module-level account.
- **Perturbation dependence:** Ablation and patching results can depend on the replacement value, corruption construction, and evaluation metric. Conclusions should be checked against at least one credible alternative and accompanied by non-target behavioral diagnostics.
- **Context variability:** A result from one context dataset may describe a sample-specific computation. Replication across contexts is necessary; consistent high-level behavior with varying component identities would itself be an informative result.
- **Computation and access:** Recomputing feature effects after many internal modifications can be expensive, and some full-scale checkpoints have restrictive licenses or substantial GPU requirements. Low feature dimension, batched queries, a restricted component search, and nanoTabPFN provide a feasible fallback.

Feature dependence is an additional interpretive constraint: ALE, marginal Shapley values, and conditional Shapley values answer different questions. The thesis should avoid causal interpretations of feature effects unless additional causal assumptions are introduced. Positive, negative, and mixed results are all useful: failure to find a stable localized mechanism may indicate distributed, redundant, or strongly context-specific computation rather than a failed investigation.

## 7. Intended outcomes

The thesis should produce a clear conceptual and computational link between model-agnostic behavioral explanations and internal analysis of a TFM. Concrete outputs may include formalized behavioral evaluation quantities, a reproducible collection of controlled tasks, and code connecting at least one feature-effect or interaction method with internal model instrumentation.

The empirical findings are deliberately open. The study may identify a selective internal mechanism for a feature interaction, show that only a coarse distributed mechanism is supported, or demonstrate that apparently stable external effects rely on different components across contexts. Any of these outcomes would clarify the evidentiary value and limitations of using model-agnostic IML as an instrument for mechanistic interpretability.

## 8. Starting literature

1. Hollmann, Noah, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. 2025. “Accurate Predictions on Small Data with a Tabular Foundation Model.” *Nature* 637: 319–326. [https://doi.org/10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6). Foundation for TabPFN v2, its in-context prediction setting, and its table-specific architecture.
2. Rundel, David, Julius Kobialka, Constantin von Crailsheim, Matthias Feurer, Thomas Nagler, and David Rügamer. 2024. “Interpretable Machine Learning for TabPFN.” In *Explainable Artificial Intelligence*, 465–476. [https://doi.org/10.1007/978-3-031-63797-1_23](https://doi.org/10.1007/978-3-031-63797-1_23). Closest foundation for adapting model-agnostic feature effects and importance to TabPFN.
3. Apley, Daniel W., and Jingyu Zhu. 2020. “Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models.” *Journal of the Royal Statistical Society: Series B* 82(4): 1059–1086. [https://doi.org/10.1111/rssb.12377](https://doi.org/10.1111/rssb.12377). Original ALE paper.
4. Herbinger, Julia, Marvin N. Wright, Thomas Nagler, Bernd Bischl, and Giuseppe Casalicchio. 2024. “Decomposing Global Feature Effects Based on Feature Interactions.” *Journal of Machine Learning Research* 25(381): 1–65. [https://jmlr.org/papers/v25/23-0699.html](https://jmlr.org/papers/v25/23-0699.html). Introduces GADGET for regional feature effects and interaction-related heterogeneity.
5. Sundararajan, Mukund, Kedar Dhamdhere, and Ashish Agarwal. 2020. “The Shapley Taylor Interaction Index.” In *Proceedings of the 37th International Conference on Machine Learning*, 9259–9268. [https://proceedings.mlr.press/v119/sundararajan20a.html](https://proceedings.mlr.press/v119/sundararajan20a.html). Defines axiomatic local attributions for feature interactions.
6. Conmy, Arthur, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. 2023. “Towards Automated Circuit Discovery for Mechanistic Interpretability.” *Advances in Neural Information Processing Systems* 36: 16318–16352. [https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html). Provides a behavior-metric, patching, and circuit-discovery workflow.
7. Hong, Xiangyu, Che Jiang, Kai Tian, Biqing Qi, Youbang Sun, Ning Ding, and Bowen Zhou. 2025. “DePass: Unified Feature Attributing by Simple Decomposed Forward Pass.” *NeurIPS 2025*. [https://arxiv.org/abs/2510.18462](https://arxiv.org/abs/2510.18462). Candidate method for decomposing Transformer information flow into chosen components.
8. Sena, Luan Borges Teodoro Reis, and Francisco Galuppo Azevedo. 2026. “Real-Time Explanations for Tabular Foundation Models.” 2nd FM4Science Workshop at ICLR 2026. [https://arxiv.org/abs/2603.29946](https://arxiv.org/abs/2603.29946). Introduces ShapPFN, an adjacent explanation-by-design approach.
9. Biloš, Marin, James T. Wilson, Anderson Schneider, and Yuriy Nevmyvaka. 2026. “A Mechanistic Study of Tabular Foundation Models.” arXiv:2605.21288. [https://arxiv.org/abs/2605.21288](https://arxiv.org/abs/2605.21288). Closest cross-model mechanistic study of learned readouts and invariances in TabPFN, TabICL, and Mitra.
10. Gupta, Atharva, Dhruv Kumar, Murari Mandal, and Saurabh Deshpande. 2026. “Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function.” arXiv:2606.12917. [https://arxiv.org/abs/2606.12917](https://arxiv.org/abs/2606.12917). Applies activation patching and head ablation to synthetic TabPFN regression tasks.
11. Pfefferle, Alexander, Johannes Hog, Lennart Purucker, and Frank Hutter. 2025. “nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN.” arXiv:2511.03634. [https://arxiv.org/abs/2511.03634](https://arxiv.org/abs/2511.03634). Provides the compact experimental model and training code proposed as the feasible starting point.
