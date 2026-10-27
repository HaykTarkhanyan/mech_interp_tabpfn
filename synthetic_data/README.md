# Synthetic data built from real data, for evaluating explanation methods

The likely thesis direction (DECISIONS.md #19): a benchmark that learns the feature distribution of
real datasets, fits a closed-form target, and publishes exact reference explanations. Self-contained:
it has its own uv environment and scripts and does not depend on the TabPFN work elsewhere in the repo.

## Contents

```
xai_benchmark_pitch.pdf                    the benchmark proposal (8 slides)
chatgpt_xai_ground_truth_benchmarks.md     saved ChatGPT conversation on ground-truth XAI benchmarks
chatgpt_xai_ground_truth_benchmarks.json   the same, as data (11 visible messages)
chatgpt_conference_submission_plan.md      saved ChatGPT conversation (2026-10-24): venues, related papers, modality
chatgpt_conference_submission_plan.json    the same, as data (6 visible messages)
papers/                    the five core papers, read in full (XAI-BENCH, Islam et al. 2025, ARF, Advani et al., Karim),
                           four more must-reads, also read in full (OpenXAI, ITErpretability, Through the Looking Glass,
                           Miro-Nicolau et al. 2026), four on data-fitted simulation and estimands (Athey et al. WGAN,
                           Credence, RealCause, Chen et al. "True to the model or true to the data?"), and
                           must_read_5_main_parts.pdf (main text of five, for printing)
paper_explainers/          one standalone HTML explainer per paper in papers/ (critique, demo, link to this work)
benchmarks/                the papers' own benchmarks, runnable here (shared env, pinned clones, patches, one runner
                           each); start at benchmarks/report.html and benchmarks/README.md (DECISIONS.md #25)
directions_2026-10-27.md   directions found while running them, with open questions Q13-Q16
results/transfer_pilot/    transfer-pilot runs (<tag>.json) and the summaries and reports derived from them
results/frontier/          realism-tractability probe: frontier.json and frontier_report.html
lit_review/
  synthetic_data_for_xai_evaluation_review.{md,pdf}   scoping review, 81 verified references
  review_source.md          review text with [@key] citation markers (edit this, then re-render)
  included.json             manual screening decisions: the 81 included works, by theme
  references.json           resolved, title-checked references (built from included.json)
  verification_report.json  81/81 verified against arXiv/Crossref
  figures/prisma_flow.png   drawn from the count files
  data/raw/                 search results, keyword-stage checkpoint, search parameters and counts
  data/processed/           deduplicated + screened records, shortlist, screening sheet, PRISMA counts
scripts/
  lit_search.py      keyword search (OpenAlex, Crossref, arXiv) + OpenAlex citation chaining + abstract backfill
  lit_screen.py      merge duplicates, rule-based relevance screen, shortlist and PRISMA counts
  lit_references.py  resolve included.json into references.json, title-checked
  lit_prisma.py      PRISMA figure from the count files
  lit_render.py      citations -> author-year, reference list, house-style checks, PDF via pandoc
  transfer_pilot_v2.py  first study (DECISIONS.md #22, #24): do data-fitted worlds reproduce reality's explainer ranking?
  transfer_pilot_v3_conditional.py  the same question under the conditional Shapley estimand (DECISIONS.md #26)
  transfer_report_v2.py summary JSON + HTML report from a transfer_pilot_v2.py or v3 results file
  transfer_cost.py      wall-time estimate per study tier, from the --timing runs
  frontier_probe.py     how much of a real task exact-SHAP target families capture (~4 min); pairwise_truth
  frontier_report.py    HTML report from results/frontier/frontier.json (~2 s)
  archive/              v1 pilot scripts (see archive/README.md)
tests/test_truth.py  ground-truth shortcuts (pilot and frontier) vs exact enumeration or closed forms (~25 s, mostly imports)
logs/                per-script logs
```

## Re-running the review

From `synthetic_data/` (environment: `uv sync`; Python 3.12, `requests==2.34.2`, `matplotlib==3.11.2`):

```
.venv/Scripts/python.exe scripts/lit_search.py                  # ~5 min; uses ~$0.08 of OpenAlex's free daily allowance
.venv/Scripts/python.exe scripts/lit_search.py --reuse-keyword  # chaining only, ~2-3 min
.venv/Scripts/python.exe scripts/lit_screen.py                  # ~10 s, offline
.venv/Scripts/python.exe scripts/lit_references.py              # ~4-6 min (polite delays to arXiv/Crossref)
.venv/Scripts/python.exe ../scripts/verify_references.py lit_review/references.json --output lit_review/verification_report.json   # ~3 min
.venv/Scripts/python.exe scripts/lit_prisma.py                  # ~3 s
.venv/Scripts/python.exe scripts/lit_render.py --pdf            # ~40 s
```

## Transfer pilot

```
.venv/Scripts/python.exe -m unittest tests.test_truth                                  # ~25 s (1.8 s of tests)
.venv/Scripts/python.exe scripts/transfer_pilot_v2.py --timing --reality rf --tag timing_rf   # ~5 min; 2 threads
.venv/Scripts/python.exe scripts/transfer_cost.py                                      # < 1 s, from the timing JSONs
.venv/Scripts/python.exe scripts/transfer_report_v2.py --tag timing_rf                 # a few s, from the JSON
```

Full runs take hours; see `results/transfer_pilot/cost_estimate.md`. ARF comes from the user's fork of arfpy
(github.com/HaykTarkhanyan/arfpy, pinned by commit in pyproject.toml; DECISIONS.md #23).

Keyless OpenAlex shares a $0.10 daily budget per network IP, and answers HTTP 401 to a second `cursor`
page, so the scripts page by number. DBLP served an anti-bot challenge and Semantic Scholar rate-limited
on 2026-10-19; both are recorded in `search_counts.json` and in the review's methods section.
