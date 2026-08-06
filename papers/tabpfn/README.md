# TabPFN papers

39 papers on TabPFN and tabular foundation models, retrieved 2026-08-06.
147.9 MB across 39 PDFs. Every PDF verified: title matched against the registry before
saving, `%PDF` magic number checked, and confirmed readable.

`manifest.json` carries full metadata per paper - title, authors, date, DOI, arXiv abs URL,
byte count, and the title-match ratio that admitted it.

## Scope

The **core model line plus methodologically significant work**, plus two comparison models
(TabICL, TabDPT) for cross-architecture work, and the PFN theory literature on what these
models are supposed to be computing. Deliberately excludes the ~40 narrow domain
applications DBLP also indexes under "TabPFN" (IoT intrusion detection, drug discovery, crop
yield, insurance pricing, Alzheimer's prediction), which apply TabPFN rather than developing
or analysing it. DBLP reports 69 papers mentioning TabPFN in total.

**Search-scope warning, twice learned.** The first pass keyed on "TabPFN" in the title and
missed two significant mechanistic papers that say "Tabular Foundation Models" instead
(2605.21288, 2605.06510). A third pass searching the underlying concept, **"prior-data fitted
networks"**, surfaced a further ~25 DBLP hits including the theory papers that define what
TabPFN is supposed to be computing (2305.11097, 2410.01565). Product name, architecture name
and concept name each retrieve a different slice. Search all three.


### Core model line (6)

| Year | ID | First author | Title |
|---|---|---|---|
| 2021 | [2112.10510](https://arxiv.org/abs/2112.10510) | Müller et al. | Transformers Can Do Bayesian Inference |
| 2022 | [2207.01848](https://arxiv.org/abs/2207.01848) | Hollmann et al. | TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second |
| 2025 | [Nature](https://doi.org/10.1038/s41586-024-08328-6) | Hollmann et al. | Accurate predictions on small data with a tabular foundation model |
| 2025 | [2511.03634](https://arxiv.org/abs/2511.03634) | Pfefferle et al. | nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN |
| 2025 | [2511.08667](https://arxiv.org/abs/2511.08667) | Grinsztajn et al. | TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models |
| 2026 | [2605.13986](https://arxiv.org/abs/2605.13986) | Grinsztajn et al. | TabPFN-3: Technical Report |

### Analysis: what the model actually does (11)

| Year | ID | First author | Title |
|---|---|---|---|
| 2025 | [2502.08978](https://arxiv.org/abs/2502.08978) | McCarter et al. | What exactly has TabPFN learned to do? |
| 2025 | [2502.17361](https://arxiv.org/abs/2502.17361) | Ye et al. | A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities |
| 2025 | [2505.16226](https://arxiv.org/abs/2505.16226) | Cheng et al. | Realistic Evaluation of TabPFN v2 in Open Environments |
| 2025 | [2505.20003](https://arxiv.org/abs/2505.20003) | Zhang et al. | TabPFN: One Model to Rule Them All? |
| 2025 | [2511.07236](https://arxiv.org/abs/2511.07236) | Swelam et al. | Does TabPFN Understand Causal Structures? |
| 2025 | [2511.18278](https://arxiv.org/abs/2511.18278) | Zheng et al. | From Tables to Signals: Revealing Spectral Adaptivity in TabPFN |
| 2026 | [2601.08181](https://arxiv.org/abs/2601.08181) | Gupta et al. | TabPFN Through The Looking Glass: An interpretability study of TabPFN and its internal representations |
| 2026 | [2602.04596](https://arxiv.org/abs/2602.04596) | Fortini et al. | Uncertainty Decomposition for Bayes-Filtered Transformers via Bayesian Predictive Inference |
| 2026 | [2605.06510](https://arxiv.org/abs/2605.06510) | Balef et al. | Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models |
| 2026 | [2605.21288](https://arxiv.org/abs/2605.21288) | Biloš et al. | A Mechanistic Study of Tabular Foundation Models |
| 2026 | [2606.12917](https://arxiv.org/abs/2606.12917) | Gupta et al. | Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function |

### Scaling and efficiency (6)

| Year | ID | First author | Title |
|---|---|---|---|
| 2023 | [2311.10609](https://arxiv.org/abs/2311.10609) | Feuer et al. | Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks |
| 2024 | [2402.06971](https://arxiv.org/abs/2402.06971) | Ma et al. | In-Context Data Distillation with TabPFN |
| 2024 | [2402.11137](https://arxiv.org/abs/2402.11137) | Feuer et al. | TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks |
| 2025 | [2502.02527](https://arxiv.org/abs/2502.02527) | Liu et al. | TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems |
| 2025 | [2509.00326](https://arxiv.org/abs/2509.00326) | Sergazinov et al. | Chunked TabPFN: Exact Training-Free In-Context Learning for Long-Context Tabular Data |
| 2025 | [2510.06162](https://arxiv.org/abs/2510.06162) | Kolberg et al. | TabPFN-Wide: Continued Pre-Training for Extreme Feature Counts |

### Extensions and adaptations (9)

| Year | ID | First author | Title |
|---|---|---|---|
| 2024 | [2403.10923](https://arxiv.org/abs/2403.10923) | Rundel et al. | Interpretable Machine Learning for TabPFN |
| 2024 | [2406.05216](https://arxiv.org/abs/2406.05216) | Ma et al. | TabPFGen -- Tabular Data Generation with TabPFN |
| 2024 | [2406.06891](https://arxiv.org/abs/2406.06891) | Liu et al. | Tokenize features, enhancing tables: the FT-TABPFN model for tabular classification |
| 2024 | [2411.10634](https://arxiv.org/abs/2411.10634) | Helli et al. | Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data |
| 2025 | [2501.02945](https://arxiv.org/abs/2501.02945) | Hoo et al. | From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting |
| 2025 | [2507.03971](https://arxiv.org/abs/2507.03971) | Garg et al. | Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data |
| 2026 | [2603.10254](https://arxiv.org/abs/2603.10254) | Tugnoli et al. | Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure |
| 2026 | [2605.20234](https://arxiv.org/abs/2605.20234) | Cureton et al. | TabPFN-MT: A Natively Multitask In-Context Learner for Tabular Data |
| 2026 | [2606.04876](https://arxiv.org/abs/2606.04876) | Tajjar et al. | Towards Pretraining Text Encoders for TabPFN |

### Comparison models (other tabular foundation models) (2)

| Year | ID | First author | Title |
|---|---|---|---|
| 2024 | [2410.18164](https://arxiv.org/abs/2410.18164) | Ma et al. | TabDPT: Scaling Tabular Foundation Models on Real Data |
| 2025 | [2502.05564](https://arxiv.org/abs/2502.05564) | Qu et al. | TabICL: A Tabular Foundation Model for In-Context Learning on Large Data |

### PFN theory: what the model is supposed to be computing (5)

| Year | ID | First author | Title |
|---|---|---|---|
| 2023 | [2305.11097](https://arxiv.org/abs/2305.11097) | Nagler et al. | Statistical Foundations of Prior-Data Fitted Networks |
| 2024 | [2410.01565](https://arxiv.org/abs/2410.01565) | Müller et al. | Bayes' Power for Explaining In-Context Learning Generalizations |
| 2025 | [2503.01256](https://arxiv.org/abs/2503.01256) | Wang et al. | Prior-Fitted Networks Scale to Larger Datasets When Treated as Weak Learners |
| 2025 | [2509.20950](https://arxiv.org/abs/2509.20950) | Sharma et al. | Decoupled-Value Attention for Prior-Data Fitted Networks: GP Inference for Physical Equations |
| 2026 | [2603.12037](https://arxiv.org/abs/2603.12037) | Melnychuk et al. | Frequentist Consistency of Prior-Data Fitted Networks for Causal Inference |

## Provenance and how to reproduce

```bash
python scripts/download_tabpfn_papers.py --dry-run   # verify every title, download nothing
python scripts/download_tabpfn_papers.py             # download; resumable
python scripts/non_essential/render_tabpfn_readme.py # regenerate this file
```

arXiv identifiers come from DBLP's `ee` field or from OpenAlex, not from recall. The exception
is `2207.01848` (the original TabPFN paper): DBLP holds only the ICLR'23 entry with no arXiv
link, so that identifier rests purely on a title match against the arXiv abs page.

The script fetches each abs page and compares `citation_title` against the expected title
before writing anything, so a wrong identifier cannot save the wrong paper under a
right-looking name. Two papers tripped this guard because they were **retitled between arXiv
versions** while DBLP kept the original title:

| arXiv | DBLP title | Current arXiv title |
|---|---|---|
| 2501.02945 | The Tabular Foundation Model TabPFN Outperforms Specialized Time Series... | From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting |
| 2602.04596 | A principled framework for uncertainty decomposition in TabPFN | Uncertainty Decomposition for Bayes-Filtered Transformers via Bayesian Predictive Inference |

Both were confirmed to be the intended work (abstracts still concern TabPFN; 2501.02945 is
TabPFN-TS) before the expected titles were updated. Neither was accepted automatically.

The Nature paper has no arXiv preprint. It is resolved via Crossref
(`10.1038/s41586-024-08328-6`) and its PDF is downloadable because the article is open access.

## Notes

- The arXiv **API** returns HTTP 429 to this network; the **website** serves `citation_*` meta
  tags and `/pdf/` downloads normally, which is what this uses.
- Downloads are rate-limited to one request per 3s and the script is resumable.
- Research directions built on this collection:
  [`research_ideas/tabpfn_mech_interp.md`](../../research_ideas/tabpfn_mech_interp.md).
