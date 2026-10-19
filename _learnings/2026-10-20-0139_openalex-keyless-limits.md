# OpenAlex without a key: a shared $0.10 daily budget, no cursor paging, and gaps in indexing

**Symptom (synthetic-data review, 2026-10-19).**
- The keyless daily budget is $0.10 per network IP, shared by every session on the machine. The response
  header `x-ratelimit-remaining-usd` shows what is left; a search costs about $0.001 and a list query $0.0001.
  The review's search and chaining used about $0.08; $0.0194 was left afterwards.
- The second page of a `cursor=*` listing returned HTTP 401 without a key; numbered `page=` paging worked.
- A phrase search (`title.search:"..."`) failed for titles containing "?".
- The AISTATS ARF paper (Watson et al.) has no OpenAlex record at all; XAI-BENCH and Advani et al. exist only
  as preprint records, and Advani's journal version is a separate record with its own 16 citing works.

**Consequences.**
- Check the remaining budget before chaining; another session's run drains the same allowance.
- Page by number; checkpoint the keyword stage so a crash does not re-spend it (`--reuse-keyword`).
- Resolve seeds by DOI when title search fails, and record which version (preprint or journal) was chained.
- With a key (via the OpenAlex MCP login, about $1/day) most of this goes away.
