# mech_interp_tabpfn

**Mechanistic interpretability of TabPFN and tabular foundation models.**

Starts from a systematic literature review of mech interp on transformer language models
(with a fully reproducible retrieval-to-verification pipeline), then narrows onto TabPFN: a
verified paper collection, research directions, a proposal review, and runnable baselines.

## The review

- **[`lit_review/mechanistic_interpretability_review.md`](lit_review/mechanistic_interpretability_review.md)** - the review (13pp as PDF)
- **[`lit_review/mechanistic_interpretability_review.pdf`](lit_review/mechanistic_interpretability_review.pdf)** - typeset via pandoc/xelatex
- **[`lit_review/references.json`](lit_review/references.json)** - 53 references with provenance
- **[`lit_review/verification_report.json`](lit_review/verification_report.json)** - 53/53 verified, 0 failures

Searched 2026-08-05 across OpenAlex, Crossref and DBLP: 1,822 records retrieved, 1,489 after
deduplication, 342 passing automated relevance screening, 53 cited.

## Paper collections

**[`papers/tabpfn/`](papers/tabpfn/)** - 39 papers (148 MB) on TabPFN and tabular foundation
models: the core model line, analysis, scaling and extension work, two comparison models, and
the PFN theory literature.
Excludes narrow domain applications. Each PDF
was title-verified against the registry before saving; `manifest.json` has full metadata and
`README.md` explains scope and provenance. Rebuild with
`python scripts/download_tabpfn_papers.py`.

## Research directions

**[`research_ideas/tabpfn_mech_interp.md`](research_ideas/tabpfn_mech_interp.md)** - 17 ideas
for mechanistic interpretability on TabPFN, grouped by ambitiousness and tagged by compute
tier (laptop CPU through A100). Grounded in what the six existing TabPFN interpretability
papers already cover, so the ideas are additive. The argument: TabPFN is the one setting where
mech interp's central problem, having no ground truth for what a correct explanation is, can
actually be solved.

## Pipeline

```bash
python scripts/search_literature.py     # 16 themes x 3 databases -> data/raw/
python scripts/screen_papers.py --allow-missing-supplementary   # dedup + relevance -> data/processed/
python scripts/fetch_web_sources.py     # Playwright: metadata for non-indexed publications
python scripts/build_references.py      # assemble references from verified sources only
python scripts/verify_references.py lit_review/references.json
python scripts/plot_prisma.py           # PRISMA figure from the counts file
```

No API keys required. All databases are used on keyless tiers.

## Two things worth knowing before re-running

**The field's foundational literature is not in any bibliographic database.** *Toy Models of
Superposition*, *A Mathematical Framework for Transformer Circuits*, *Towards
Monosemanticity*, *Zoom In* and four others are indexed in none of OpenAlex, Crossref or DBLP -
they are published on `transformer-circuits.pub` and `distill.pub` without conventional DOIs.
A purely database-driven review of this field silently omits its own foundations. See §2.4 of
the review, and `scripts/fetch_web_sources.py` for how they are retrieved instead.

**Rate limits shape what is reachable.** As of 2026-08-05 the arXiv API and the keyless
Semantic Scholar pool both returned HTTP 429 to this network, and OpenAlex began 429-ing
partway through the session. The arXiv *website* stayed reachable and serves `citation_*` meta
tags, which is how preprint metadata is verified. Scripts probe unavailable sources and record
the outcome in their output rather than skipping them silently.

## Layout

```
lit_review/     review, references, verification report, figures
papers/tabpfn/  39 verified PDFs + manifest
paper_explainers/ plain-language HTML walkthroughs of single papers
research_ideas/ research directions, grouped by ambition and compute
reviews/        proposal reviews + their citation-verification output
tabpfn_benchmark/ TabPFN vs GBDT/RF/LogReg on OpenML German Credit, + HTML reports
tabpfn_toy_demo/  minimal TabPFN classification example on Iris
fe_mech_pipeline/ ALE behavior specs + head/MLP ablation on TabPFN v2: Delta_B with null, floor, selectivity; CPU pilot
synthetic_data/ likely thesis direction: semi-synthetic XAI ground-truth benchmark (pitch, scoping review,
                core-paper explainers, transfer pilot; own uv env)
robotics_foundation_models_explainability/ literature review: explainability and mech interp of VLAs and
                other robot foundation models (151 works, standalone HTML review, 83 papers; own uv env)
scripts/        load-bearing pipeline (see above)
  non_essential/  supplementary retrieval, blocked on an external rate limit
data/raw/       raw search results per database + search parameters
data/processed/ deduplicated, screened, PRISMA counts
logs/           per-script run logs (gitignored)
_learnings/     one file per non-obvious lesson
_work_sessions/ session logs
DECISIONS.md    numbered design decisions with rationale
```

`DECISIONS.md` records why the databases, the verification routing, the screening rules and
the figure toolchain are what they are.
