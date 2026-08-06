# Review: *Feature Effects as Behavioral Specifications for Mechanistic Analysis of Tabular Foundation Models*

Reviewed 2026-08-07 against the 39-paper collection in `papers/tabpfn/` and the field map in
`lit_review/mechanistic_interpretability_review.md`.

**Overall: a good proposal with one serious missing citation and one fixable methodological
hole.** The core bet is well aimed, the scoping is realistic, and the citation hygiene is
better than most published work. The two substantive problems below are both addressable
before the thesis starts, and fixing them would materially improve it.

---

## 1. Citation check

All eight verifiable references resolve correctly, with exact metadata
(`reviews/proposal_verification.json`):

| Claimed | Verified |
|---|---|
| Hollmann et al. 2025, *Nature* 637: 319-326 | Correct, volume and page range exact, 8 authors |
| Apley & Zhu 2020, *JRSS-B* 82(4): 1059-1086 | Correct, volume/issue/pages exact |
| Biloš et al., Wilson, Schneider, **Nevmyvaka** | Correct, all four authors |
| Gupta, Kumar, Mandal, Deshpande | Correct, exact author list and order |
| Rundel et al., DePass, ShapPFN, nanoTabPFN | All resolve, titles match |

8/8 verified, 0 title mismatches. Two very recent papers I had not previously encountered
(DePass 2510.18462, ShapPFN 2603.29946) are real and correctly described. One trivial note:
Rundel et al. is cited as "In *Explainable Artificial Intelligence*, 465-476"; Crossref lists
the container as *Communications in Computer and Information Science*, the series the xAI 2024
proceedings appear in. Both are defensible; naming the series is more standard.

---

## 2. The serious gap: a missing paper that threatens RQ2

**Balef, Koshil & Eggensperger (2026), "Is One Layer Enough? Understanding Inference Dynamics
in Tabular Foundation Models", arXiv:2605.06510.**

This is not cited and it should be the first thing the student reads. It is the *first
large-scale mechanistic study of layerwise dynamics across six tabular ICL models*, TabPFN
among them, and its central finding is **substantial depthwise redundancy**: iterative
refinement with overlapping computations across layers. They demonstrate this constructively
by building a looped single-layer model using **20% of the original parameters** at comparable
performance. Code is released.

Why this matters directly for this proposal:

- **RQ2 asks "at which layers and in which components does a selected main effect or
  interaction become represented".** If computation is redundantly distributed across depth,
  that question may have no clean answer for structural reasons rather than methodological
  ones. The proposal's fallback ("if component-level localization proves unstable, a
  defensible fallback is a coarser layer- or module-level account", §6) is the right instinct,
  but Balef et al. suggests this is the *expected* outcome, not a contingency.
- **It weakens single-component ablation as evidence.** With redundant pathways, ablating
  component C may leave \(B(g_D)\) nearly unchanged because another component recomputes the
  same quantity. A null Δ_B would then be evidence of redundancy, not of C being irrelevant.
  Distinguishing those two requires ablating *sets* of components, which changes the search
  design.
- **It supplies a prediction and a codebase.** Starting from their released instrumentation is
  far cheaper than building layerwise tooling from scratch, and "does the redundancy story hold
  for a specified feature interaction?" is a sharper framing of RQ2 than the current one.

Recommendation: cite it, read it before finalizing RQ2, and consider reframing RQ2 as *given
known depthwise redundancy, is a specified interaction localizable at all, and at what
granularity?* That is a better question and it has a genuinely uncertain answer.

---

## 3. The methodological hole: Δ_B has no null model

The evaluation quantity is

\[\Delta_B(C,D)=d\!\left(B(g_D),B(g_D^{(-C)})\right)\]

and the proposal says "a useful mechanistic account should explain a substantial share of the
target behavior while preserving relevant non-target behavior" (§2). **Substantial relative to
what is never specified, and without that the criterion is unfalsifiable.** Ablating any
component changes behavior. Some components matter more than others by construction, for
reasons that have nothing to do with the target interaction.

The fix is small and standard:

1. **Random-component null.** Report the distribution of \(\Delta_B(C',D)\) over random
   component sets \(C'\) of matched size and type. The claim is then "C is in the upper tail of
   the null", which is falsifiable.
2. **Estimator-variability floor.** \(B(g_D)\) is itself an estimate whose value depends on ALE
   bin count, reference distribution, and context sample. Recompute \(B(g_D)\) under resampling
   with *no* ablation and report that spread. Any \(\Delta_B\) below this floor is
   indistinguishable from estimator noise. The proposal notes that "changing them changes the
   behavior being explained" (§5) but does not build the consequence into the metric.
3. **Non-target control, quantified.** §5 correctly says a mechanism that destroys the
   predictor should not count. Operationalize it: report \(\Delta_B\) on a *designated
   non-target effect* alongside the target, and treat the ratio as the headline number. That is
   the selectivity claim, and it is what distinguishes this work from ordinary ablation studies.

This is the single highest-value change to the document. It converts the central contribution
from "we located a mechanism" to "we located a mechanism and can show it is specific", which
is precisely the evidentiary standard the mechanistic interpretability literature currently
lacks (see §5 below).

---

## 4. Missed opportunity: nanoTabPFN is treated only as "small and fast"

§5 says the synthetic design "supplies a known functional structure at the data level, but not
a ground-truth neural mechanism". That honest caveat is the proposal's main epistemic
limitation, and **nanoTabPFN can partly lift it**, which the proposal does not exploit.

nanoTabPFN pre-trains in roughly **one minute on a single GPU** (Pfefferle et al. report
160,000x faster than TabPFN v2 pre-training). That makes the *pre-training prior* an
experimental variable, not a fixed given. Concretely:

- Pre-train one model on a prior containing feature interactions and one on a prior without
  them, holding architecture and everything else fixed.
- If the interaction mechanism you localize exists in the first and not the second, you have
  causal evidence that the mechanism is a response to the prior, not an artifact of your
  localization method or of the context dataset.
- If it exists in both, the interaction is being constructed in-context rather than recalled
  from pre-training, which is a substantive and publishable finding about what in-context
  learning does.

This is the closest thing to a ground-truth mechanism available anywhere in mechanistic
interpretability, it costs minutes of compute, and it directly answers the limitation the
proposal already identifies about itself. Worth at least a paragraph in §5, and it is a natural
extension in §3.

---

## 5. What the proposal gets right, and why it matters more than it claims

Worth stating plainly because the document undersells its own contribution.

**The core bet addresses a real, current gap.** The mechanistic interpretability literature's
binding constraint is evaluation: Heap et al. (2025) found automated interpretability metrics
do not distinguish trained transformers from randomly initialised ones; Hase et al. (2023)
found that localizing a fact does not tell you where to edit it. Both are failures of
*verification*. This proposal's move, specifying behavior externally and precisely before
looking inside, is a direct response to that, and the framing "external explanations as
evaluation instruments" is stronger than the document's own modest presentation of it. That
argument belongs in §1, not implicitly in §7.

**§6's first bullet is the right instinct.** "Recovering the known synthetic interaction shows
that the model learned a behavior, not how it implemented it" is exactly the distinction most
papers in this area blur.

**The scoping is realistic.** One model, one behavioral specification, one localization method,
one validation method, with a stated minimum viable core. Better bounded than most proposals.

**Negative results are pre-authorized** (§6, §7). Given §2 above, this is not just good practice,
it is likely load-bearing.

---

## 6. Smaller points

**The two attention types are noted but not used.** §2 correctly says TabPFN "alternates
attention across observations and features". That factorisation gives a *pre-registerable,
architecture-specific hypothesis* that the current RQ2 does not state: a pairwise **feature
interaction** should be carried by feature-wise (within-row) attention, while sample-wise
attention performs estimation from context. Stating that up front is much stronger than "at
which layers and in which components". Note also that Gupta et al. already studied feature-wise
heads specifically, so this builds on them rather than duplicating.

**RQ1 is the weakest of the three.** "Which estimands provide sufficiently precise and stable
targets" is a survey question without a falsifiable answer; a thesis could satisfy it by
tabulating properties. Sharper: *does the choice of estimand change which components are
identified?* That has a yes/no answer, it is cheap to test once the pipeline exists, and a
"yes" would be an important warning to the field.

**RQ3 already has partial prior evidence.** Gupta et al. report that contrastive activation
steering **fails to transfer across samples**, attributing it to in-context encoding of task
structure. RQ3 asks about reuse across contexts, so this is not merely related, it is a partial
prior answer and a strong prior in favour of the "different contexts recruit partially
different components" hypothesis already floated in §4. Cite it there.

**The PFN theory literature is absent.** Two omissions are conspicuous:
- Nagler, *Statistical Foundations of Prior-Data Fitted Networks* (2305.11097, ICML 2023),
  which argues a purely **frequentist** reading of PFNs as pre-tuned but untrained predictors
  explains their behaviour. Nagler is a co-author on two works the proposal *does* cite
  (Rundel et al., Herbinger et al.), so its absence will be noticed.
- Müller, Hollmann & Hutter, *Bayes' Power for Explaining In-Context Learning Generalizations*
  (2410.01565), which argues the opposite, **Bayesian**, reading.

These bear on the proposal more than they might appear. What counts as "the mechanism" of a PFN
is contested at the theory level, neither side has mechanistic evidence, and Biloš et al. find
an implemented readout (similarity-weighted vote) that matches neither cleanly. A thesis that
specifies behavior precisely and then tests implementation is well positioned to say something
about that dispute, which is a much larger claim than "we localized an interaction". At minimum
it is a strong framing sentence for §2.

**Scoop risk is real but the angle is distinctive.** Three groups are visibly active on TFM
internals: Gupta et al., Biloš et al., and Balef/Eggensperger. The IML-as-specification framing
is not obviously on any of their trajectories, so the risk is to RQ2's descriptive parts rather
than to the core contribution. Worth checking arXiv monthly rather than at the end.

---

## 7. Priority list

1. **Cite and read Balef et al. (2605.06510).** Reframe RQ2 around known depthwise redundancy.
   Highest impact, costs nothing.
2. **Add a null model and estimator-variability floor to Δ_B.** Two sentences in §2, and it is
   what makes the central claim falsifiable.
3. **Report target/non-target Δ_B as a ratio.** Makes selectivity the headline number.
4. **Add the prior-ablation extension using nanoTabPFN.** Cheap, and it partly answers the
   proposal's own stated limitation about ground-truth mechanism.
5. **State the feature-attention hypothesis explicitly** as a pre-registered prediction.
6. **Add Nagler 2305.11097 and Müller et al. 2410.01565** to the starting literature.
7. **Sharpen RQ1** from "which estimands are suitable" to "does the estimand change the answer".

Items 1-3 I would treat as required before starting. Items 4-7 improve the thesis but the core
survives without them.
