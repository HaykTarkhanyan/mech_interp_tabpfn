# A relevance screen that reads abstracts silently penalises databases that have none

**Symptom.** ROME (*Locating and Editing Factual Associations in GPT*, 105 citations, clearly
in scope) failed the automated relevance screen. So did *Transcoders Beat Sparse Autoencoders
for Interpretability*. Meanwhile a 1978 paper titled *Introduction* and a review of the
cognitive neuroscience of working memory passed.

**Cause.** Two independent defects, both invisible without inspecting individual records.

*Missing abstracts.* The screen scored `title + abstract` against a term vocabulary. **729 of
1,489 deduplicated records (49%) had no abstract at all** - Crossref frequently omits them and
DBLP never carries them:

```
records with no abstract: 729 / 1489
by source: Crossref 524, DBLP 167, OpenAlex 56
```

ROME came from Crossref with an empty abstract, so only one term matched from its title and it
fell under the threshold. The screen was rejecting records for *missing data*, not for being
off-topic - and doing so disproportionately to the peer-reviewed venue records that DBLP and
Crossref uniquely contribute.

*Over-broad context gate.* Requiring a "transformer"/"language model" context word alongside
domain terms rejected `Transcoders Beat Sparse Autoencoders for Interpretability`: two
unambiguous mech-interp terms, but no context word in the title and no abstract to supply one.
Strong domain vocabulary does not need corroboration; the context gate exists only to
disambiguate *weak* terms.

**Fixes.**
1. Include on `>=1 strong term` OR `>=3 distinct medium terms AND >=1 context term`. Recovered
   99 records immediately (243 -> 342 passing).
2. Backfill abstracts by DOI before screening, rather than loosening thresholds to compensate.
   Fixing the data beats weakening the filter.
3. Drop `interpretab` and `explainab` from the term vocabulary entirely - they are the general
   XAI markers and let medical-XAI surveys through. Measured: the XAI survey passed on exactly
   `['interpretab', 'explainab']`, the working-memory paper on `['circuit', 'internal
   representation']` where "circuit" meant a *biological* neural circuit.

**Consequence / rule.** When a screen reads a field, first measure how often that field is
populated, broken down by source. A 49% missing rate turns a relevance filter into a metadata-
completeness filter, and the failure is silent - the excluded records look like ordinary
exclusions in the count.

**Related bug, same root.** Fuzzy title matching with a containment boost
(`if a in b or b in a: ratio = 0.95`) matched a 1978 paper titled *Introduction* against *Zoom
In: An Introduction to Circuits*. Guard containment with a minimum length (`len >= 25`) - short
strings are contained in everything.
