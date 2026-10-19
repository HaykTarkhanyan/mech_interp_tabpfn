# NeurIPS XAI Benchmark Paper

ChatGPT conversation (gpt-5-6-thinking), saved from the share link https://chatgpt.com/share/6aabad6b-81dc-83eb-8222-c74810a16fa4 on 2026-10-19 18:38.
Citation markers are converted to the inline links ChatGPT shows; tracking parameters removed. Hidden reasoning and tool calls are not part of the share and are left out.

---

## Question 1 (2026-09-17 10:33)

> I remember a paper in neurips benchmark and datasets track that publishes something about evaluation of explainations what was it?

### ChatGPT (status note)

I suspect you mean a NeurIPS Datasets & Benchmarks paper on **systematic evaluation of post-hoc explanations**, but there are a few close candidates. I’ll narrow it down by venue/year and the exact evaluation setup.

### ChatGPT

The **most likely paper is _OpenXAI: Towards a Transparent Evaluation of Model Explanations_**, Agarwal et al., **NeurIPS 2022, Datasets & Benchmarks Track**. It introduced a framework specifically for **benchmarking post-hoc explanations**, including 22 metrics covering faithfulness, stability/robustness, and fairness, plus datasets, pretrained models, attribution methods, and public leaderboards. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/65398a0eba88c9b4a1c38ae405b125ef-Abstract-Datasets_and_Benchmarks.html))

[NeurIPS paper: OpenXAI](https://proceedings.neurips.cc/paper_files/paper/2022/hash/65398a0eba88c9b4a1c38ae405b125ef-Abstract-Datasets_and_Benchmarks.html)

Two other very plausible ones, depending on what you remember:

- **LATEC — _Navigating the Maze of Explainable AI: A Systematic Approach to Evaluating Methods and Metrics_**, NeurIPS **2024 D&B**. This is more explicitly a **meta-evaluation of XAI evaluation**: 17 XAI methods × 20 metrics × architectures/modalities, 7,560 combinations. A key result is that evaluation metrics often disagree and can induce unstable/conflicting rankings. They also release **326k saliency maps and 378k metric scores** as an evaluation dataset. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7beb1ddb04401ba21869f23a77f3e4e1-Abstract-Datasets_and_Benchmarks_Track.html))
- **M⁴ — _A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods across Metrics, Modalities and Models_**, NeurIPS **2023 D&B**. It focuses specifically on **faithfulness metrics**, provides a taxonomy, and evaluates attribution methods across image/text modalities and multiple architectures. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html))

If what you remember was a paper arguing that **different explanation-evaluation metrics disagree / rankings are unreliable**, then it was almost certainly **LATEC**. If you remember a general evaluation framework/leaderboard, it was **OpenXAI**.

<details><summary>Sources cited in this answer</summary>

- https://proceedings.neurips.cc/paper_files/paper/2022/hash/65398a0eba88c9b4a1c38ae405b125ef-Abstract-Datasets_and_Benchmarks.html
- https://proceedings.neurips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/7beb1ddb04401ba21869f23a77f3e4e1-Abstract-Datasets_and_Benchmarks_Track.html

</details>

---

## Question 2 (2026-09-17 10:36)

> is there some paper that provides ground-truths?

### ChatGPT (status note)

Yes. There are benchmark papers that construct or curate **ground-truth explanatory structure** so explanation methods can be scored against something known, rather than only via proxy metrics. I’ll separate true causal/constructive ground truth from annotation-based “gold explanations,” because those are quite different.

### ChatGPT

Yes. If you mean **ground-truth feature attributions/explanations**, the strongest NeurIPS Datasets & Benchmarks matches are:

- **OpenXAI: Towards a Transparent Evaluation of Model Explanations** — NeurIPS 2022 D&B. Its synthetic generator explicitly constructs **ground-truth local explanations**. Each local region has a feature mask \(m_k\) indicating which features are relevant and weights \(w_k\) encoding their influence. The construction is designed so that an accurate learned model is guaranteed, under their assumptions, to respect that explanatory structure. They then compare estimated attributions to ground truth using feature agreement, rank agreement, sign agreement, rank correlation, etc. ([Zitnik Lab](https://zitniklab.hms.harvard.edu/publications/papers/OpenXAI-NeurIPS22.pdf))

- **Synthetic Benchmarks for Scientific Research in Explainable Machine Learning (XAI-BENCH)** — NeurIPS 2021 D&B. This provides configurable synthetic distributions where quantities needed for **ground-truth Shapley-value evaluation** can be computed efficiently. So its notion of ground truth is especially relevant if you care about evaluating SHAP-like attribution estimators. ([NeurIPS Datasets Benchmarks Proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/c16a5320fa475530d9583c34fd356ef5-Abstract-round2.html))

Outside NeurIPS, a particularly clean example is **CLEVR-XAI** (Arras, Osman, Samek, *Information Fusion*, 2022). It gives **pixel-level ground-truth masks** for visual explanations: the objects/pixels that are actually relevant to answering each CLEVR question are known from the scene/question generation process. They introduce relevance-mass and relevance-rank accuracy against these masks. ([DOI](https://doi.org/10.1016/j.inffus.2021.11.008))

There is an important distinction here. **OpenXAI is probably closest to what you're after if you want ground truth for local feature attribution.** XAI-BENCH gives you ground-truth quantities such as Shapley values; CLEVR-XAI gives ground-truth relevant regions. None of these solves the general problem of having a unique “true explanation” for arbitrary real-world models.

Also, OpenXAI explicitly points out a subtle problem with older synthetic benchmarks: knowing the **data-generating ground truth** does not automatically tell you what a fitted model actually learned. Their synthetic construction tries to close that gap. ([alphaXiv](https://www.alphaxiv.org/abs/2206.11104v5))

If you're thinking about using such benchmarks for a paper, I can also give you the **main ground-truth-XAI benchmark papers from ~2018–2026**, categorized by tabular / vision / text and by what exactly counts as “ground truth.”

<details><summary>Sources cited in this answer</summary>

- https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/c16a5320fa475530d9583c34fd356ef5-Abstract-round2.html
- https://doi.org/10.1016/j.inffus.2021.11.008
- https://www.alphaxiv.org/abs/2206.11104v5
- https://zitniklab.hms.harvard.edu/publications/papers/OpenXAI-NeurIPS22.pdf

</details>

---

## Question 3 (2026-09-17 10:38)

> summarize all those findings briefly here

### ChatGPT

Across these papers, the picture is fairly consistent: **there is no single reliable, universally accepted way to evaluate feature-attribution explanations, and metric choice can materially change conclusions.**

- **OpenXAI (NeurIPS 2022 D&B):** standardizes evaluation across **22 metrics** covering faithfulness, stability, and fairness. Its synthetic setting includes known explanatory structure, allowing direct agreement checks against ground truth. Main message: explanation quality is multidimensional; methods can look good under one criterion and weak under another, so single-metric claims are fragile. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/65398a0eba88c9b4a1c38ae405b125ef-Abstract-Datasets_and_Benchmarks.html))

- **M⁴ (NeurIPS 2023 D&B):** studies faithfulness across **metrics × modalities × model architectures**. It emphasizes that many commonly used faithfulness metrics encode different notions of “correctness” and require different forms of ground truth or intervention. Rankings of attribution methods therefore depend substantially on the evaluation metric and experimental configuration. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html))

- **LATEC (NeurIPS 2024 D&B):** the strongest large-scale evidence for **metric disagreement**. Across 17 XAI methods, 20 metrics, and 7,560 configurations, metrics often produce conflicting rankings. Their conclusion is essentially that selecting one or a few metrics can induce substantial selection bias; they advocate evaluating across multiple validated metrics. Expected Gradients performed strongly in their aggregate experiments, despite being largely absent from earlier comparative studies. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7beb1ddb04401ba21869f23a77f3e4e1-Abstract-Datasets_and_Benchmarks_Track.html))

- **XAI-BENCH (NeurIPS 2021 D&B):** attacks the missing-ground-truth problem via configurable synthetic distributions. Because the data-generating distribution is known, conditional expectations and hence **ground-truth Shapley quantities** can be computed efficiently. This enables controlled comparisons of SHAP, LIME, MAPLE, SHAPR, L2X, etc., including under changing feature dependence, model classes, and distributions. ([NeurIPS Datasets Benchmarks Proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/c16a5320fa475530d9583c34fd356ef5-Abstract-round2.html))

- **CLEVR-XAI (Information Fusion 2022):** provides **pixel-level ground-truth relevance masks** derived from CLEVR's scene graph and question program. Comparing 10 methods, they find LRP, Integrated Gradients, and Guided Backprop among the strongest; Grad-CAM and Deconvnet perform poorly, with vanilla gradient variants in between. Importantly, some of these findings **contradict conclusions from proxy evaluations** such as randomization/sanity tests. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253521002335))

The interesting synthesis for evaluation research is therefore:

> **Ground-truth benchmarks reveal that proxy XAI metrics need not agree with actual explanation recovery; meanwhile, when ground truth is unavailable, different proxy metrics themselves disagree enough to change method rankings.**

So there are really **two evaluation problems**: whether an XAI method recovers a known explanatory target, and whether an evaluation metric is itself a valid surrogate for that recovery. CLEVR-XAI/XAI-BENCH/OpenXAI give you leverage on the former; **LATEC and M⁴ expose the latter**.

<details><summary>Sources cited in this answer</summary>

- https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/c16a5320fa475530d9583c34fd356ef5-Abstract-round2.html
- https://proceedings.neurips.cc/paper_files/paper/2022/hash/65398a0eba88c9b4a1c38ae405b125ef-Abstract-Datasets_and_Benchmarks.html
- https://proceedings.neurips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/7beb1ddb04401ba21869f23a77f3e4e1-Abstract-Datasets_and_Benchmarks_Track.html
- https://www.sciencedirect.com/science/article/pii/S1566253521002335

</details>

---

## Question 4 (2026-09-17 10:40)

> also precisely describe in what form they provide the ground-truth, only for the explanations or also for the feature-target relationships?

### ChatGPT (status note)

I’ll distinguish three things for each benchmark: the **object they call ground truth**, the **representation it takes**, and whether the benchmark also exposes the underlying **feature→target mechanism** rather than only a reference explanation.

### ChatGPT

Yes. The key distinction is whether the benchmark gives you (i) a reference explanation, (ii) the actual \(X\rightarrow Y\) data-generating relationship, or (iii) both. They differ substantially.

| Benchmark | Form of “ground truth” | Ground-truth \(X\rightarrow Y\) relationship? | What it actually lets you validate |
|---|---|---|---|
| **OpenXAI / SynthGauss** | Tabular feature-importance vector, constructed from cluster-specific masks/weights | **Yes, explicitly** | Recovery of known feature relevance/importance |
| **XAI-BENCH** | Exact/near-exact Shapley quantities derived from known synthetic distribution | **Yes, explicitly** | Whether estimated attributions approximate a particular formal attribution target |
| **CLEVR-XAI** | Binary pixel masks identifying question-relevant objects | **Yes, but symbolically at object/task level** | Whether saliency is spatially concentrated on semantically relevant objects |
| **M⁴ synthetic GT** | Binary image patch masks; also pseudo-GT attribution maps in another evaluation regime | **Partly / by construction** | Whether attribution finds an intentionally injected predictive signal |
| **LATEC** | **No explanatory ground truth** | **No** | Agreement/disagreement and behavior of proxy evaluation metrics |

More precisely:

**OpenXAI is the closest to “we know the feature-target relationship and therefore know what the explanation should be.”** Its SynthGauss generator samples well-separated Gaussian clusters. For each cluster \(k\), it samples

\[
m_k\in\{0,1\}^d
\]

indicating which variables influence \(Y\), and

\[
w_k\in\mathbb R^d
\]

describing their relative effects. Labels are generated from the features using the masked weight vector \(m_k\odot w_k\), passed through a labeling function such as a sigmoid. Thus they know the complete local data-generating relation. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf))

The paper refers to \(m_k\) as the cluster's ground-truth explanation, although the underlying generator additionally contains \(w_k\), which supplies magnitude/direction information. The implementation describes the ground-truth explanations used for agreement metrics in terms of coefficients/feature importance, allowing rank and sign comparisons. ([GitHub](https://github.com/AI4LIFE-GROUP/OpenXAI/blob/main/README.md))

So schematically:

\[
X \xrightarrow[\text{known }m_k,w_k]{\text{known generator}} Y
\]

and from that mechanism they construct

\[
e^\star(x).
\]

This is stronger than simply attaching an annotation saying “features 2 and 7 are relevant.” Importantly, OpenXAI also tries to solve the **data-ground-truth versus model-ground-truth problem**: independent features and well-separated neighborhoods are imposed so that an accurate fitted model cannot simply exploit correlated proxy variables instead. The authors explicitly motivate SynthGauss this way. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf))

---

**XAI-BENCH is different and arguably cleaner if your target is a mathematically defined attribution functional such as conditional SHAP.** The benchmark specifies the complete synthetic joint distribution. Features come from known families such as multivariate Gaussians, Gaussian mixtures, or multinomials, and the labels are generated as

\[
Y=h(X).
\]

They explicitly implement linear, nonlinear additive, piecewise-constant and piecewise-linear \(h\). For example, the linear case is \(y_{\mathrm{raw}}=w^\top x\); their nonlinear additive case uses known coordinate-wise transformations. ([NeurIPS Datasets Benchmarks Proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round2.pdf))

So **yes, it provides the feature-target relationship itself**, not merely explanations.

Because \(P(X)\) is also known, they can calculate distributions such as

\[
p(X_S\mid X_{\bar S}=x_{\bar S})
\]

and therefore conditional expectations required for Shapley calculations. ([NeurIPS Datasets Benchmarks Proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round2.pdf))

The “ground-truth explanation” is consequently not simply the coefficients of \(h\). Instead it can be the **ground-truth Shapley vector**

\[
\phi^\star(x;f,P_X),
\]

obtained from the known distribution and the predictive model. Their GT-Shapley metric compares an explainer's feature weights against these Shapley values. ([NeurIPS Datasets Benchmarks Proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round2.pdf))

This distinction matters:

\[
\underbrace{h:X\rightarrow Y}_{\text{known DGP}}
\quad\neq\quad
\underbrace{\phi^\star(x)}_{\text{derived attribution target}}.
\]

XAI-BENCH provides enough information to obtain both.

There is a caveat, which is exactly what OpenXAI later criticizes: **knowing the data-generating feature-target relation does not imply that a fitted model \(f\) uses those same features in the same way**. With correlated predictors, \(f\) can learn proxy relationships. OpenXAI explicitly discusses this limitation of earlier synthetic benchmarks. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf))

---

**CLEVR-XAI gives considerably weaker “ground truth” in the attribution sense.** CLEVR itself has an explicit symbolic generative process:

\[
\text{scene graph} + \text{functional question program}
\longrightarrow Y.
\]

The scene graph specifies object attributes and relations; a functional program such as filtering, relating, counting, or querying is executed against it to produce the correct answer. Thus at the **semantic/object level**, the true task relationship is known exactly. ([DOI](https://doi.org/10.1016/j.inffus.2021.11.008))

But the explanation GT distributed by CLEVR-XAI is mainly **binary spatial masks**, not real-valued attribution vectors.

For simple questions they provide:

\[
M^\star\in\{0,1\}^{H\times W}
\]

for the single target object, plus a mask containing all objects.

For complex questions they provide variants such as **GT Unique**, **GT Unique First-non-empty**, **GT Union**, and **GT All Objects**, obtained by tracing the intermediate object sets in the functional program. ([DOI](https://doi.org/10.1016/j.inffus.2021.11.008))

So you know that, say, pixels belonging to objects \(A,C,D\) are relevant according to the task-generating program. You **do not** get a true continuous pixel attribution such as

\[
(0.73,-0.14,0,\ldots)
\]

nor a pixel-level causal-effect function relating each RGB input dimension to the answer.

There is another conceptual issue: these masks describe **which information should be sufficient/relevant according to the CLEVR task**, not necessarily which pixels a trained Relation Network actually uses. CLEVR-XAI evaluates explanations of a trained model against this task-semantic GT. ([DOI](https://doi.org/10.1016/j.inffus.2021.11.008))

---

**M⁴ has two kinds of reference targets, neither equivalent to OpenXAI's known feature-target mechanism.**

The first is **pseudo-ground truth**. For an attribution method \(A\), they compute attribution maps across several well-trained models, normalize/average them, and regard the resulting consensus map as the pseudo-GT. An individual explanation is compared to that consensus using cosine similarity. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/05957c194f4c77ac9d91e1374d2def6b-Paper-Datasets_and_Benchmarks.pdf))

So there is **no true feature-target relationship here**. It is essentially

\[
e_{\text{pseudo}}(x)
=
\frac1M\sum_m A(x,f_m).
\]

The second is **synthetic ground truth**. They insert conspicuous image patches and reverse labels for patched examples. The patch is deliberately made the predictive signal, and its spatial mask is therefore treated as the GT explanation. They compare attribution maps to this patch mask using AP/AUC-ROC. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/05957c194f4c77ac9d91e1374d2def6b-Paper-Datasets_and_Benchmarks.pdf))

So in this setting there *is* an injected relationship:

\[
\text{patch present} \Rightarrow \text{changed label},
\]

but it is a deliberately constructed binary relationship, not a full known \(P(Y\mid X)\) or a per-feature effect function.

---

**LATEC does not provide ground-truth explanations or ground-truth feature-target relations.** Its released meta-evaluation dataset consists principally of pretrained model weights, **326,790 generated saliency maps**, and **378,000 evaluation scores**. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/7beb1ddb04401ba21869f23a77f3e4e1-Paper-Datasets_and_Benchmarks_Track.pdf))

Its contribution is therefore orthogonal: given

\[
A_1(x),\ldots,A_m(x)
\]

and evaluation metrics

\[
q_1,\ldots,q_r,
\]

how stable are conclusions/rankings across metrics, architectures, modalities, etc.? It does not tell you which attribution is truly correct. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7beb1ddb04401ba21869f23a77f3e4e1-Abstract-Datasets_and_Benchmarks_Track.html))

The conceptual taxonomy I'd use is therefore:

\[
\boxed{\text{Known DGP} \rightarrow \text{derived attribution GT}}
\]

for **XAI-BENCH**, and largely **OpenXAI**;

\[
\boxed{\text{Known semantic reasoning process} \rightarrow \text{binary relevance GT}}
\]

for **CLEVR-XAI**;

\[
\boxed{\text{Injected shortcut} \rightarrow \text{binary relevance GT}}
\]

for **M⁴'s synthetic evaluation**; and

\[
\boxed{\text{No GT; evaluate metrics/explanations against proxies}}
\]

for **LATEC**.

For what I think you're getting at, **OpenXAI and XAI-BENCH are the two most relevant** if you want to study whether an explanation recovers *known feature–target structure*. CLEVR-XAI is instead closer to **known support/relevant-region recovery**. The distinction between those two notions of “ground-truth explanation” is quite important.

---

If you want, I can:

- Explain differences in ground-truth forms across XAI benchmarks
- Discuss significance of joint ground-truth explanations and feature-target models
- Summarize key challenges in using ground-truth to validate explanation methods

<details><summary>Sources cited in this answer</summary>

- https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round2.pdf
- https://doi.org/10.1016/j.inffus.2021.11.008
- https://github.com/AI4LIFE-GROUP/OpenXAI/blob/main/README.md
- https://proceedings.neurips.cc/paper_files/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf
- https://proceedings.neurips.cc/paper_files/paper/2023/file/05957c194f4c77ac9d91e1374d2def6b-Paper-Datasets_and_Benchmarks.pdf
- https://proceedings.neurips.cc/paper_files/paper/2024/file/7beb1ddb04401ba21869f23a77f3e4e1-Paper-Datasets_and_Benchmarks_Track.pdf
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/7beb1ddb04401ba21869f23a77f3e4e1-Abstract-Datasets_and_Benchmarks_Track.html

</details>
