# Bibliographic databases do not index mechanistic interpretability's foundational papers

**Symptom.** A 16-theme, 3-database systematic search returning 1,822 records failed to
retrieve *Toy Models of Superposition*, *A Mathematical Framework for Transformer Circuits*,
*Towards Monosemanticity*, *Scaling Monosemanticity*, *Zoom In: An Introduction to Circuits*,
*Softmax Linear Units*, *Circuit Tracing*, or *On the Biology of a Large Language Model* -
arguably the eight most influential works in the field.

**Cause.** They have no Crossref-registered DOI. They are published on
`transformer-circuits.pub` and `distill.pub`. Explicit title lookups confirm absence rather
than a ranking problem:

```
OpenAlex  search "A Mathematical Framework for Transformer Circuits"  -> no match
DBLP      search "A Mathematical Framework for Transformer Circuits"  -> 0 hits
Crossref  GET /works/10.48550/arXiv.2209.10652                        -> 404
```

The Crossref 404 is a second, separate trap: arXiv registers DOIs with **DataCite**, not
Crossref. A verifier that checks existence via `doi.org/api/handles` (registrar-agnostic, so it
passes) but reads metadata from Crossref will report an arXiv paper as "verified" with an empty
metadata record. That is a false pass, and in this field it would cover most of a bibliography.

Structural reason: NeurIPS, ICML and ICLR historically issue no DOIs; arXiv uses DataCite;
industry labs publish primary research on their own web properties. Bibliographic
infrastructure built around Crossref-registered journal articles has systematically poor
coverage of exactly this literature.

**Consequences.**
- Any purely database-driven review of mech interp silently omits its own foundations. Budget
  for a targeted seed-title stage; do not assume keyword breadth substitutes for it.
- Route citation verification **by registrar**: Crossref for journal/conference DOIs, arXiv or
  DataCite for `10.48550/*`, direct URL resolution for web-native publications.
- Cross-database overlap was near zero - of 1,489 unique records, **1,448 were found by only
  one database**, 4 by all three. Multi-database search here is not redundancy for reliability;
  the databases behave as near-disjoint samples. Never conclude "not in the literature" from
  one database.

**Two workarounds that mattered.**
1. The arXiv **API** was HTTP 429 to this network all session, but the arXiv **website**
   (`arxiv.org/abs/<id>`) stayed reachable and serves `citation_title` / `citation_author` /
   `citation_date` meta tags. That became the preprint metadata path.
2. `transformer-circuits.pub` renders its byline client-side - `curl | grep citation_author`
   returns nothing. Playwright renders the page and the authors are readable from the DOM.
   Without it those references would have carried empty or remembered author lists.

**Disproven theory.** The initial arXiv 429 looked like my own query being malformed or too
complex. It was not: a trivial `?search_query=all:superposition&max_results=2` also returned
429, `http://` returned 301, and a browser User-Agent returned 503. It was IP-level throttling,
independent of the query. Diagnosing with the simplest possible request first saved rewriting
a query builder that was never at fault.
