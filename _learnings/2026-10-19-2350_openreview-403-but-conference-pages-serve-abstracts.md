# OpenReview blocks scripts, but neurips.cc and icml.cc pages serve workshop abstracts

**Symptom.** Twelve core works in the robotics review exist only as 2026 workshop papers (ICML MI workshop,
NeurIPS RoboPAD, Grounded and Faithful VLMs). Their PDFs could not be fetched:

```
curl -A "<browser UA>" "https://openreview.net/pdf?id=GQEGpBhtYk"   ->  403 text/html
curl https://openreview.net/forum?id=...                            ->  282-character block page
```

**What works.** The conference "virtual" pages are plain HTML with the full abstract and the OpenReview id:

```
curl -L https://neurips.cc/virtual/2026/162921  ->  200 (redirects to /virtual/2026/loc/paris/162921)
curl -L https://icml.cc/virtual/2026/79347      ->  200
```

`robotics_foundation_models_explainability/scripts/fetch_workshop_pages.py` got full abstracts for 9 of 19
workshop items this way. The abstracts were far more informative than the one-line summaries from the web
search. One example: a SmolVLA paper whose 130-site patching sweep "localized" a failure until the same
sweep on a no-failure contrast gave the same profile.

**Trap.** A URL under `/virtual/2026/workshop/...` is the workshop's own page. Its "abstract" describes the
workshop, not the paper. The first version of the script stored it as the paper's abstract, and the
review's reference list then claimed the abstract had been read. Fixed with a `page_is_work` flag.

**Consequences.** For workshop-only papers: listing page first (abstract, OpenReview id), and a real
browser (Playwright) for the PDF. Always label in the output how much of each such work was read.
