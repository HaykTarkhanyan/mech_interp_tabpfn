# Verify citations by comparing the registry's title, not just that the identifier resolves

**Symptom.** Two arXiv identifiers written from prior belief were wrong, and both would have
produced citations that look completely normal:

```
2501.17727  assumed: "Sparse Autoencoders Can Interpret Randomly Initialized Transformers"
            actual:  "Automated Interpretability Metrics Do Not Distinguish Trained and
                      Random Transformers"                             (match ratio 0.48)

2408.01416  assumed: "The Quest for the Right Mediator: A History, Survey, and Theoretical
                      Grounding of Causal Interpretability"
            actual:  "The Quest for the Right Mediator: Surveying Mechanistic Interpretability
                      Through the Lens of Causal Mediation Analysis"   (match ratio 0.57)
```

**Cause.** Both identifiers *resolve*. Both return HTTP 200. A verifier that checks "does this
DOI/ID exist" passes them without complaint. The papers are real; the pairing of identifier to
title is what was wrong. The second case is the nastier one - same paper, same first clause of
the title, different subtitle, because the work was retitled between versions.

**Fix.** Verification must compare the *cited* title against the *registry's* title and fail on
mismatch:

```python
ratio = SequenceMatcher(None, normalise(cited_title), normalise(registry_title)).ratio()
if ratio < 0.85:
    raise / record failure    # a resolving identifier for the WRONG paper
```

Applied at build time (`build_references.py`) and again at verify time
(`verify_references.py`). Final state: 53/53 verified, 0 title mismatches - but only because
the guard rejected the two bad pairings first.

**Consequence / rule.** "The link works" is not citation verification. An identifier that
resolves to the wrong paper is worse than a broken one, because nothing downstream will ever
flag it - a reader who follows the link finds a real, plausible, topically adjacent paper.
Any pipeline that emits citations should treat identifier-to-title agreement as the check,
and existence as merely necessary.

**Corollary for LLM-authored bibliographies.** The failure mode here is specifically the one
models produce: confident, well-formed, nearly-right metadata. The guard is cheap and it is the
only thing standing between "remembered an arXiv ID" and a silently incorrect reference list.
Build it before writing citations, not after.
