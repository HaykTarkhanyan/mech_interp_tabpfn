# In a brand-new field, keyword search alone misses about a quarter of the included works

**Evidence.** Provenance of the 151 works included in the robotics-foundation-model review
(`robotics_foundation_models_explainability/lit_review/data/processed/provenance.json`):

```
found by arXiv keyword search (16 complete boolean queries, 1,572 records)    111
found only by the web sweep (agent: 73 searches, workshop pages, lab blogs)   31
found only through reference lists of the downloaded PDFs                       5
found by web sweep and reference lists, not by keyword search                   4
unique contribution of Crossref (320 records)                                   0
                                                                        total 151
```

Of the 31 web-only works, 19 are workshop papers with no arXiv version, so no database indexes them yet.
The other 12 are arXiv papers that use different vocabulary ("action memorization", "lookup table",
chess look-ahead, Sokoban planning) or are adjacent literature. For the core VLA work itself, keyword
recall was good. The one core miss was "Demystifying Diffusion Policies", which never says
"interpretability".

**Consequences.**
- For a field less than two years old, budget a web or venue sweep (workshop accepted-paper pages) as a
  first-class retrieval route, not an afterthought. In this case it is where the newest work lives.
- Crossref adds nothing for arXiv-native ML topics; it can be dropped there.
- Backward chaining from PDF reference lists is cheap (no API key) and found 5 works, including one cited by
  7 of the 80 downloaded papers (2503.08558).
- Keyword search missed 40 of 151 included works (26%): the 31 web-only, the 5 reference-list-only, and 4
  found by both of those routes.
