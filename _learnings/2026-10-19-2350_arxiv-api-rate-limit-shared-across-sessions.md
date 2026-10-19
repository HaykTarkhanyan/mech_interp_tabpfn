# The arXiv API rate limit is per IP, so two Claude sessions on this machine throttle each other

**Symptom.** The arXiv API answered a probe query normally (HTTP 200, 205 hits) and then, about ten minutes
later, returned 429, 503 and read timeouts to a test run of `robotics_foundation_models_explainability/scripts/lit_search.py`:

```
2026-10-19 21:56:02 WARNING  arXiv HTTP 429 (attempt 1/4), retry in 10s
2026-10-19 21:56:24 WARNING  arXiv HTTP 429 (attempt 2/4), retry in 20s
...
RuntimeError: arXiv failed after 4 attempts
```

**Cause.** Another Claude session was running `synthetic_data/scripts/lit_references.py` (started 21:55:08)
against the same API from the same IP. Found with:

```
Get-CimInstance Win32_Process -Filter "Name='python.exe'"   ->  ...synthetic_data\.venv\Scripts\python.exe scripts/lit_references.py
curl https://arxiv.org/abs/2509.00328                        ->  200   (website fine)
curl https://export.arxiv.org/api/query?id_list=2509.00328   ->  429   (API throttled)
```

Once that process ended, the API returned 200 again and the full 16-query search completed in about
three minutes with no retries.

**Consequences.**
- Before blaming a query or the network for arXiv 429s, list running Python processes: another session
  may be the cause. Polite per-script delays do not add up across processes.
- Do not run two arXiv-heavy scripts at once, even from different repos or sessions.
- The website (`arxiv.org/abs`, `/pdf`) and the API (`export.arxiv.org`) throttle separately, as already
  noted in `2026-08-06-0030_bibliographic-databases-miss-mech-interp-foundations.md`.
