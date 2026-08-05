# non_essential/

Maintained and reusable, but not required to rebuild the literature review.

## seed_and_enrich.py

**What it does.** Two supplementary retrieval stages against OpenAlex:

1. **Seed search** - looks up 46 canonical mechanistic-interpretability titles explicitly,
   so that foundational works missed by keyword search still enter the corpus. Titles that
   come back not-found are reported rather than dropped.
2. **Abstract enrichment** - 49% of deduplicated records carry no abstract (Crossref often
   omits them, DBLP never has them), and the relevance screen reads title+abstract, so those
   records get excluded for missing metadata rather than for being off-topic. This stage
   batch-refetches abstracts from OpenAlex by DOI, 50 per request.

**Why it is here and not at the top level.** It never completed. OpenAlex began returning
HTTP 429 to this network partway through the session (2026-08-05, after roughly 200 requests)
and did not recover across ~50 minutes of patient retries with exponential backoff. The review
was produced without it: `scripts/screen_papers.py --allow-missing-supplementary` records the
skip in `data/processed/prisma_counts.json`, and the resulting limitation is disclosed in
§4.4 of the review.

The gap it would have filled was covered another way. `scripts/build_references.py` resolves
foundational works from arxiv.org directly (the arXiv *website* serves `citation_*` meta tags
even though the arXiv *API* is 429 to this network), with a title-match guard so a wrong
identifier fails loudly.

**When to re-run it.** When OpenAlex is reachable again. It is resumable - seed lookups are
checkpointed to `data/processed/seed_cache.json` after every title, so an interrupted run
picks up where it stopped. Running it and then re-running `screen_papers.py` *without*
`--allow-missing-supplementary` would strengthen the review: more records would survive
screening on their merits rather than on the presence of an abstract.

Moved here 2026-08-06, reason: blocked on an external rate limit, not on the repo.
