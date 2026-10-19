# DBLP's search API returned HTTP 202 with an anti-bot page instead of JSON

**Symptom (2026-10-19).** After one HTTP 429, `https://dblp.org/search/publ/api?...&format=json` answered
HTTP 202 with an HTML challenge page. A status-code check alone passes it; JSON parsing then fails.

**Fix.** The search probe in `synthetic_data/scripts/lit_search.py` checks that the body is JSON, records DBLP
as unavailable with the reason in `search_counts.json`, and requires at least three working keyword
databases (arXiv's API replaced DBLP that day). No attempt was made to get past the challenge.

**Consequences.** Probe each database for content, not just status, before a run, and say in the review's
methods which databases were unavailable and why.
