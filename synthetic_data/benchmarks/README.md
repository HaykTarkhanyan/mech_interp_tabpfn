# Upstream benchmarks, runnable here

The benchmarks of the papers in `../papers/`, set up so they can be run and explored on this laptop (CPU, no CUDA).
DECISIONS.md #25 records how: one shared uv env, upstream code cloned at pinned commits, small committed patches,
and one runner per benchmark that does a deliberately small run. **`report.html` is the place to start**: what each
run did, its results, and what it confirms or contradicts in the paper (and in our explainer pages).

## Layout

```
pyproject.toml, uv.lock   the shared env (Python 3.12, every dependency pinned with ==)
setup_sources.sh          clones the 8 upstream repos into _src/ at the tested commits and applies patches/
patches/<repo>.patch      compatibility patches (lazy optional imports, removed numpy aliases, CPU trainers)
common.py                 paths, logging to ../logs/bench_<tag>.log, 2-thread caps, a disk guard, result writer
run_openxai.py            OpenXAI: 8 datasets x {LR, ANN} x 7 explainers, OpenXAI's own metrics
run_xaibench.py           XAI-BENCH: Gaussian datasets x rho, 6 explainers, plus scores against interventional SHAP
run_iterpretability.py    ITErpretability: CATE learners on real Twins covariates, with a random-attribution baseline
run_realcause.py          RealCause: re-analysis of its 1,568-estimator CSV + estimators on its pre-sampled data
run_wgan.py               Athey et al.: simulate from the shipped WGANs (LaLonde) and score seven ATE estimators
run_credence.py           Credence-v2: fit to LaLonde PSID with a chosen effect, check it, score the estimators
run_fb2ts_tabular.py      Miro-Nicolau et al.: the tabular notebook on CPU, both ground-truth definitions, random baseline
run_all.sh                every full run one after another (~80 min), then report.py
smoke_all.sh              tiny run of each runner (~5 min), to catch errors after a change
report.py                 builds report.html from results/*.json
results/<tag>.json        the artifact of record for each run
_src/, _data/             upstream clones and downloads (gitignored)
```

## Running

From `synthetic_data/benchmarks/` in Git Bash. Every runner logs progress with elapsed and remaining time, and
its docstring says what it cuts relative to upstream and roughly how long it takes.

```
uv sync                                        # the shared env (~0.8 GB; torch is usually in uv's cache)
bash setup_sources.sh                          # ~2 min, ~620 MB (realcause ships 519 MB of sampled data)
.venv/Scripts/python.exe run_openxai.py --models lr --tag openxai_lr
.venv/Scripts/python.exe run_openxai.py --models ann --tag openxai_ann
.venv/Scripts/python.exe run_xaibench.py
.venv/Scripts/python.exe run_iterpretability.py
.venv/Scripts/python.exe run_realcause.py
.venv/Scripts/python.exe run_wgan.py
.venv/Scripts/python.exe run_credence.py
.venv/Scripts/python.exe run_fb2ts_tabular.py
.venv/Scripts/python.exe report.py             # a few seconds, from results/*.json
```

Not set up: Karim (2026)'s `regmode-code` (cloned, but it is R with a SLURM cluster layout; R 4.4.1 is installed if
you want to try `engine/regular_plasmode.R`), Through the Looking Glass (no public code) and Islam et al. (2025)
(data on Zenodo, no code).

## Caveats

- One env for code written between 2020 and 2026: most benchmarks run on newer libraries than their authors used.
  A gap from a paper's number can come from that, or from the small settings, before it comes from the paper.
  Each results JSON records the package versions.
- The runs are small (fewer points, seeds, epochs). Treat them as a working setup and a first look, not as a
  replication.
