# Archived scripts

Kept as a record; not expected to run against current results and not imported by live code.

| Script | What it did | Archived |
|---|---|---|
| `transfer_pilot.py` | Transfer pilot v1: one task per world, six SHAP-family explainers, 100-row exact-enumeration truths. Produced `results/transfer_pilot/smoke.json`. | 2026-10-20: superseded by `scripts/transfer_pilot_v2.py` (DECISIONS.md #24). The smoke run showed single-task rankings are noise; the independent review found the truth background too small and the oracle not comparable. |
| `transfer_report.py` | Summary and HTML report for v1 results (Kendall tau, top-1, regret per world). Produced `smoke_summary.json` and `smoke_report.html`. | 2026-10-20: superseded by `scripts/transfer_report_v2.py` (resolved-pair agreement, bootstrap intervals). |
