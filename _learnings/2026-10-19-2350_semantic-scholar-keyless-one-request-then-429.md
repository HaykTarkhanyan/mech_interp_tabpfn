# Semantic Scholar's keyless API allows about one request, then 429s for minutes

**Symptom.** A single request to `api.semanticscholar.org/graph/v1/paper/arXiv:2509.00328/citations`
returned 200. The next request, seconds later, returned 429:

```
{"message": "Too Many Requests. Please wait and try again or apply for a key ...", "code": "429"}
```

A probe inside `lit_search.py` passed at 22:09:53, and the first real search at 22:10:00 got 429 and kept
getting 429 through exponential backoff (10, 20, 40, 80, 150, 150, 150 s), more than seven minutes in total.
An earlier isolated test showed the same pattern for 7+ minutes.

**Cause.** The keyless pool is shared by every unauthenticated user worldwide; a success says nothing
about the next request.

**Consequences.**
- A one-request probe is not evidence that Semantic Scholar is usable for a batch. It passed and the batch
  still failed.
- Keyless Semantic Scholar cannot carry citation chaining (26 calls for 13 seeds). Use a Semantic Scholar
  key, the OpenAlex MCP login, or backward chaining from downloaded PDFs' reference lists
  (`robotics_foundation_models_explainability/scripts/lit_refchain.py`: 850 cited arXiv ids from 80 papers,
  no API key, about 90 s).
- Earlier sessions (2026-08-05, and the synthetic-data review on 2026-10-19) saw the same: this is the
  normal state, not bad luck.
