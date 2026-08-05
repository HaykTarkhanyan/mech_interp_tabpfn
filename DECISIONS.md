# Decisions

Newest first. Never delete a superseded entry.

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
