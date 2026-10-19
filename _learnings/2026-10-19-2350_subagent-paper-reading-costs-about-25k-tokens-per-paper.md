# A reader subagent writing structured notes costs about 20-26k tokens per paper

**Evidence.** Three reader agents read pdftotext output selectively (first ~300 lines, then grep for
tables, rollouts, limitations) and wrote one schema-following JSON note per paper:

```
batch A  22 papers  567k tokens   ~25.8k per paper   18.5 min
batch B  24 papers  572k tokens   ~23.8k per paper   22.0 min
batch C  34 papers  688k tokens   ~20.2k per paper   23.1 min  (9 adjacent papers at half budget)
web sweep           203k tokens   73 searches, 65 items, every arXiv id title-checked
```

The estimate before launch was ~350k per reader (about 12k per paper). Actual use was about 60% higher.

**What the cost bought.** The readers found problems a skim would miss: a bug in LIBERO-PRO's released
evaluation code, a paper labelling its own key figures "illustrative pending verification", two
table/text mismatches, and a "ROC-AUC" metric that goes below 0.5. Separating `key_findings` from
`critique` in the schema made the review's claims vs assessment split almost mechanical.

**Rule.** Budget about 25k tokens per paper for full structured reading (about 12k for adjacent papers),
and quote that number, not a lower guess, when asking for subagent approval.
