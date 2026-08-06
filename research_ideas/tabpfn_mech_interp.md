# Mechanistic interpretability on TabPFN: research directions

Written 2026-08-06. Grounded in the 30 papers in `papers/tabpfn/` and the field map in
`lit_review/mechanistic_interpretability_review.md`.

Ideas are grouped by **ambitiousness** (A/B/C) and tagged with **compute tier** (T0-T4)
calibrated to this machine: 16 GB RAM, Intel Iris Xe integrated graphics, no CUDA, plus
Colab CLI access in WSL (T4/L4/A100/H100).

---

## 1. Why TabPFN is an unusually good target

The literature review concluded that mechanistic interpretability's binding constraint is
**evaluation**: the field cannot reliably tell a correct explanation from a compelling one.
Heap et al. (2025) found that automated interpretability metrics do not distinguish trained
transformers from randomly initialised ones. Hase et al. (2023) found that localising a fact
does not tell you where to edit it. Both are failures of *verification*, not of generation.

TabPFN is, as far as I can tell, the only practically important model where that constraint
can actually be lifted. Four properties, none of which hold for an LLM:

1. **The training distribution is a known generative process.** TabPFN is pre-trained on
   synthetic datasets sampled from an explicit prior over structural causal models and
   Bayesian neural networks. You know exactly what generated the training data, because you
   wrote it.
2. **The target algorithm is specified.** The training objective is to approximate the
   posterior predictive distribution. "Is the model doing Bayesian inference?" is a precise
   question with a checkable answer, unlike "what algorithm is GPT-2 running?"
3. **Exact ground truth is computable at small scale.** On small synthetic problems the true
   posterior predictive can be computed exactly or approximated by long MCMC. You can compare
   the model's output *and* its internal computation against the right answer.
4. **The prior is an experimental variable.** nanoTabPFN (Pfefferle et al., 2025) pre-trains
   in **one minute on a single GPU** - the paper reports 160,000x faster than TabPFN v2
   pre-training. That converts "retrain with a modified prior" from an industrial-scale
   project into a loop you can run hundreds of times in an afternoon.

Property 4 is the one that matters most. It gives **interventional control over the training
distribution**, which is flatly impossible for any frontier LLM. This is the *Toy Models of
Superposition* setup - controlled synthetic data, known ground truth - except the model is
also genuinely state of the art at a real task.

### Architecture, and why it makes the questions different

From the Nature paper (Hollmann et al., 2025): each **cell** of the table gets its own
representation. Each layer applies **attention over features** (a cell attends to other
features in its row, i.e. its sample), then **attention over samples** (a cell attends to the
same feature across all other rows), then an MLP sublayer, each with residual and normalisation.
Random feature embeddings are added before the first layer to break symmetry between features
with identical statistics. Training-sample representations are not influenced by test samples.

So the residual stream is indexed by (row, column), not by token position, and there are two
structurally distinct attention mechanisms with different jobs. Almost the entire existing
mech interp toolkit assumes a one-dimensional token stream. That is a problem *and* an
opportunity: the row/column factorisation means components have a natural semantic type
before you start, which is exactly the interpretive foothold LLM work lacks.

---

## 2. What has already been done

Read this before starting anything below; several obvious ideas are taken.

| Paper | What it did | What it did NOT do |
|---|---|---|
| Gupta et al., *Through The Looking Glass* (2601.08181) | Probing for regression coefficients, intermediate values, final answer across layers | Correlational only; no causal intervention |
| Gupta et al., *Where Computation Lives Inside TabPFN* (2606.12917) | **First causal** analysis. Activation patching, ablation, attention entropy on **feature-wise** heads. Found one head 2-5x more causally necessary at peak layer; steering failed to transfer across samples | Only feature-wise heads; only **two** synthetic regression datasets; no sample-wise attention; no SAEs |
| Swelam et al., *Does TabPFN Understand Causal Structures?* (2511.07236) | Learned adapter + causal tokens decode adjacency matrices from frozen embeddings. Causal info concentrated in **mid-range layers** | Decodability, not use. Never tests whether the model *uses* that information |
| McCarter, *What exactly has TabPFN learned to do?* (2502.08978) | Black-box study of learned inductive biases, "brilliant or baffling" | Purely behavioural, no internals |
| Zheng et al., *From Tables to Signals* (2511.18278) | Frequency analysis; "Spectral Adaptivity" - capacity adapts to in-context sample count, not training epochs | Descriptive; no mechanism for *how* |
| Ye et al., *A Closer Look at TabPFN v2* (2502.17361) | Infers attribute relationships even with randomised attribute tokens | Reports the capability, does not explain it |

**The frontier in one sentence:** exactly one causal mechanistic study exists, it covers one
of the two attention types on two datasets, and nobody has applied dictionary learning to a
tabular foundation model at all.

---

## 3. Compute tiers

| Tier | Hardware | Realistic budget | Ask first? |
|---|---|---|---|
| **T0** | This laptop, CPU | Minutes. TabPFN v2 inference on n<=1000, d<=20 is seconds per fit | No |
| **T1** | This laptop, CPU | Hours to overnight. Sweeps of hundreds of small fits, activation caching | Mention it |
| **T2** | One Colab T4 session | 2-4 h. SAE training on cached activations, nanoTabPFN pre-training runs, GPU inference at scale | Yes - quota |
| **T3** | Colab L4/A100, several sessions | Large SAE suites, hundreds of nanoTabPFN pre-trains, full sweeps | Yes |
| **T4** | Out of reach here | Full TabPFN v2/2.5/3 pre-training from scratch | N/A - use nanoTabPFN instead |

Two things make this cheaper than LLM interpretability. TabPFN is small, and its activations
are low-dimensional, so **caching every activation for a whole experiment fits on disk** and
most analysis is then a pure CPU job. Per `CLAUDE.md`, anything at T2+ or any parallel
multi-core sweep gets flagged before running.

---

## 4. Tier A - grounded extensions

Low risk, clear result, workshop-paper scale. Each is a gap left by a specific existing paper.

### A1. Map the sample-wise attention heads [T0-T1]
The only causal study (2606.12917) analysed feature-wise heads. **Sample-wise attention is
where in-context learning has to live** - it is the only path by which a test cell sees
training rows. Run the same protocol (activation patching, ablation, attention entropy) over
sample-wise heads, across more than two datasets.

*Honest risk:* this is the obvious next paper for the group that wrote 2606.12917, published
June 2026. Assume it is in progress. Worth doing anyway as a skills-building exercise, but do
not plan a publication around it without checking arXiv first.

### A2. Is sample-attention doing retrieval, kernel smoothing, or inference? [T0-T1]
Construct datasets where k-NN, kernel regression and the exact Bayes predictor **disagree by
construction**, then patch sample-attention heads to see which behaviour tracks. Cheap,
decisive, and it discriminates between hypotheses rather than describing one. Connects to the
retrieval-flavoured findings in the fine-tuning literature (LoCalPFN, TuneTables).

### A3. Diagnose the steering failure properly [T0-T1]
2606.12917 reports that contrastive activation steering **fails to transfer across samples**
and attributes this to ICL encoding task structure in-context. That attribution is a
hypothesis stated in a discussion section, not a result.

Test it directly: does steering transfer *within* a fixed in-context set but not across sets?
If so, the natural reading is that TabPFN's features are **context-relative** - the direction
encoding a concept is defined relative to the current in-context distribution, so there is no
fixed "refusal direction" analogue. That would be a real and quotable contrast with LLM
steering results (Arditi et al., 2024), and it is a negative result with a mechanism, which is
worth more than a positive result without one.

### A4. Layer-wise algorithm phase diagram against context size [T1]
Spectral Adaptivity says effective capacity varies with the number of in-context samples.
Combine with per-layer decoding (tuned-lens style) and ask whether the model **switches
strategy** at particular n - simple smoothing at small n, something structured at large n. A
phase diagram over (layer, n) would be a genuinely informative artifact and is pure sweep work.

### A5. How is feature identity recovered from statistics alone? [T0-T1]
Ye et al. found attribute relationships are inferred even with randomised attribute tokens.
The Nature paper says random feature embeddings are added purely to break symmetry between
identically-distributed features. So the model reconstructs "what this column means" from
distributional evidence during the forward pass. Trace where and how. This is a well-posed
mechanistic question with a known-interesting answer.

---

## 5. Tier B - new mechanism claims

Full paper scale. Each needs causal evidence, not just probing.

### B1. Sparse autoencoders on TabPFN cell representations [T2]
**Nobody has applied dictionary learning to a tabular foundation model.** Open questions that
are novel rather than derivative:

- Is there superposition in a model whose inputs are continuous values rather than a discrete
  vocabulary? The superposition argument (Elhage et al., 2022) is about representing more
  features than dimensions. What counts as a "feature" when there is no token vocabulary?
- What do the features turn out to be? Plausible candidates: "this column is categorical",
  "this row is an outlier", "these two columns interact", "the target is monotone in this
  feature". If SAE latents correspond to *statistical properties of the dataset*, that is a
  qualitatively different kind of feature from anything in the LLM SAE literature.
- Cell representations are indexed by (row, column). Do you train one SAE over all cells, or
  separate dictionaries per attention type? The answer is not obvious and is itself a result.

Cheap: the model is small, activations are low-dimensional, and Gemma-Scope-scale
infrastructure is not needed. This is the single best ratio of novelty to compute in the doc.

### B2. Test Bayesian-ness against the exact posterior [T1-T2]
On problems small enough for the exact posterior predictive under TabPFN's own prior to be
computed (or MCMC-approximated), compare both the output *and the internal computation*.
Distinguish three hypotheses: it approximates inference; it runs a heuristic that coincides
with Bayes on-distribution; it interpolates memorised prior samples. McCarter's "brilliant or
baffling" behavioural findings are exactly what you would expect from hypothesis two, and the
mechanistic version of that test has not been run.

### B3. Recover a circuit for a known structural causal model [T2]
Generate data from a specific, known SCM. Find the circuit TabPFN uses to predict the target.
Because **you know the true causal graph**, you can check whether the recovered circuit
corresponds to it. This is *Interpretability in the Wild* (Wang et al., 2022) with an answer
key - and the answer key is precisely what that line of work has always lacked.

### B4. Prior ablation with nanoTabPFN [T2-T3] - strongest experimental design here
Re-pre-train nanoTabPFN with **systematically modified priors**: remove causal structure,
remove feature interactions, change the SCM family, vary noise. Then ask whether a given
mechanism appears **only** when the prior contains the corresponding structure.

This is a controlled experiment on the training distribution. It is the thing that is
impossible for LLMs and it converts correlational mechanism claims into causal ones at the
level of *why the mechanism exists*, not just what it does. At roughly one minute per
pre-training run, a sweep of hundreds of priors is an afternoon on a T4.

If only one idea in this document gets done, it should be this one.

### B5. Universality across the model line [T2]
Universality was a founding claim of the circuits programme (Olah et al., 2020) and remains
barely tested. TabPFN offers an unusually clean test: v1, v2, 2.5, 3 and nanoTabPFN are a
family trained on closely related priors at different scales. Do the same head roles and SAE
features recur? A negative result here is as publishable as a positive one, and it directly
addresses a gap flagged in the review.

### B6. Is the decodable causal information actually used? [T2]
Swelam et al. showed causal structure is decodable from mid-layer embeddings by a learned
adapter. **Decodable is not used.** This is precisely the trap Hase et al. (2023) documented
for factual localisation. Ablate or corrupt the mid-layer causal representation and test
whether predictions change in the way the causal-information hypothesis predicts. If they do
not, a prominent claim in the TabPFN literature needs qualifying - and demonstrating that is
a contribution.

---

## 6. Tier C - field-level bets

High risk. Each could fail entirely. Each would matter if it worked.

### C1. A ground-truth benchmark for mechanistic explanations [T3] - highest value
Build a benchmark of synthetic tabular tasks where **the correct mechanistic explanation is
known by construction**, because you specified both the data-generating process and the
training prior. Then score existing MI methods - SAEs, activation patching, attribution
patching, probing, automated interpretability - on whether they recover it.

This attacks the field's central problem head on. Heap et al. (2025) showed current automated
interpretability metrics cannot distinguish trained from random transformers, which means the
field is scoring explanations with instruments that have never themselves been validated.
Nothing validates an instrument except ground truth, and TabPFN is where ground truth exists.

The strongest version tests whether the *ranking* of methods on this benchmark predicts their
usefulness elsewhere. If it does, this becomes infrastructure the whole field uses. If it does
not, that is a substantive finding about the transferability of interpretability evaluation.

### C2. Does in-context learning here implement gradient descent? [T2-T3]
There is a standing literature claiming transformer ICL implements implicit gradient descent
or ridge regression. TabPFN is the cleanest available test case: ICL is its only mode, it was
trained explicitly to approximate Bayesian prediction, and it is small enough to analyse
exhaustively. Test mechanistically whether anything isomorphic to iterative parameter updates
is carried in the residual stream across layers. A clear negative would be valuable, since
much of that literature rests on architectures chosen for analytic convenience.

### C3. Transfer validated methods back to LLMs [T3]
If a mechanism can be established with ground truth in TabPFN, and the same analysis finds a
corresponding mechanism in LLM in-context learning, you have bootstrapped from a domain where
verification is possible into one where it is not. Speculative, and the disanalogies may
dominate, but this is the payoff that would justify the whole programme.

### C4. Design a prior for interpretability, then verify [T3]
Invert the usual approach: instead of interpreting a model you were given, design the training
prior so the resulting model is interpretable **by construction**, then check mechanistically
whether it worked. Related in spirit to Softmax Linear Units (Elhage et al., 2022) and
Transformer Programs (Friedman et al., 2023), but acting on the data distribution rather than
the architecture, which is the lever TabPFN uniquely exposes. nanoTabPFN makes the iteration
loop short enough to actually search this space.

---

## 7. The matrix

| | **T0-T1** laptop | **T2** one T4 session | **T3** L4/A100, multi-session |
|---|---|---|---|
| **A** grounded | A1 sample-wise heads · A2 retrieval vs Bayes · A3 steering diagnosis · A5 feature identity | A4 phase diagram | |
| **B** new claims | B2 Bayesian-ness (small n) | B1 **SAEs on cells** · B3 known-SCM circuit · B5 universality · B6 decodable vs used | B4 **prior ablation** |
| **C** field-level | | C2 ICL as gradient descent | C1 **ground-truth benchmark** · C3 transfer to LLMs · C4 interpretable-by-design prior |

Bold: highest value for effort.

---

## 8. Honest assessment

**Crowded.** A1 is the obvious follow-up to 2606.12917 by a group that is clearly active and
publishing quickly. Check arXiv before investing. TabPFN interpretability generally went from
almost nothing to several papers within about a year, so treat any idea here as having a live
competitor.

**Likely to disappoint.** A3 (steering) starts from a method that has already failed once in
this setting. Frame it as explaining a negative result, not as making steering work; if you
need steering to succeed for the project to be worth writing up, pick something else.
C3 (transfer to LLMs) is the kind of idea that reads well and usually dies on disanalogies
between the settings.

**Underrated.** B1 (SAEs) and B4 (prior ablation). Both are cheap, neither has been done, and
both ask questions that cannot be asked anywhere else. B6 is underrated in a different way: it
is a debunking-shaped result, which is less fun to write but is exactly what the review found
the field to be short of.

**The real reason to do this.** Not that TabPFN is important, though it is. It is that
mechanistic interpretability currently has no way to check its own answers, and this is the
one place where it could. That argument is the strongest framing for any paper coming out of
this list, and it is worth putting in the introduction rather than the discussion.

**Caveat on my own confidence.** The claims about what has and has not been done rest on the
30 papers in `papers/tabpfn/`, which cover the core line and methodological work but exclude
about 40 domain-application papers and anything published outside DBLP's index. Given that the
literature review found this field's foundational work is frequently *not* indexed anywhere
(`_learnings/2026-08-06-0030_*`), treat "nobody has done X" here as "nobody in the searched
corpus has done X" and check directly before committing to a project.

---

## 9. Concrete first week

Assumes the goal is a result, not a survey.

1. **Day 1-2, T0.** Get TabPFN v2 running on CPU on small synthetic tables. Cache activations
   for both attention types. Confirm you can reproduce one number from 2606.12917. If you
   cannot reproduce their setup, everything downstream is unanchored.
2. **Day 3, T0.** Build the A2 discriminating datasets, where k-NN, kernel regression and
   exact Bayes give different predictions. This is a small amount of code and it is reusable
   by B2, B3 and C1.
3. **Day 4-5, T1.** Run A1/A2 patching over sample-wise heads. This is the cheapest path to a
   real finding and it de-risks the tooling for everything in Tier B.
4. **Day 6-7, T2** (ask first). Stand up nanoTabPFN and do one modified-prior pre-training run
   end to end. Getting that loop working is the gate on B4, which is the best idea here.

The order is deliberate: every step produces infrastructure the next one needs, and the
expensive commitment comes last, after the cheap steps have shown whether the tooling holds.
