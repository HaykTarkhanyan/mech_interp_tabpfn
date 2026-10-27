# OpenXAI's published numbers changed between the NeurIPS 2022 paper and arXiv v5 (2024), with no erratum

**Symptom:** the NeurIPS 2022 Datasets & Benchmarks paper and arXiv 2206.11104v5 report different results for the
same methods, datasets and metrics. Examples from the two Table 2s (read 2026-10-24):
- the gradient's sign agreement (SA) on HELOC goes from 0.469 to 1.000;
- the random baseline's feature agreement (FA) goes from 0.498 to 0.149;
- the stability and fairness "winners" swap.

Neither version mentions a correction.

**Evidence from rerunning the code** (2026-10-27, `synthetic_data/benchmarks/run_openxai.py`, OpenXAI commit
a182886, 50 test points per dataset, `results/openxai_lr.json`):
- **Gradients score 1.000** on every ground-truth metric (FA, RA, SA, SRA, PRA, RC) on all 8 datasets, HELOC
  included. So did Integrated Gradients and SmoothGrad. For a logistic model each is proportional to the
  coefficient vector, which is the reference.
- **The random baseline's FA** is 0.128 to 0.220 across datasets (HELOC 0.166), close to v5's 0.149 and far from
  0.498.
- **The current code explains the predicted class** and flips the reference's sign to match
  (`evaluate_metrics.py` passes `predictions` to FA/RA/SA/SRA). A version that skipped the flip would get about half
  the signs wrong on a balanced dataset like HELOC. That fits the NeurIPS 0.469, but nothing confirms that this
  was the cause.

**Consequences:**
- Cite one version and say which. The code on GitHub reproduces v5, not the NeurIPS proceedings.
- When comparing with OpenXAI, rerun the code rather than copy numbers from either paper.
- Do not take the published ranking of "most faithful" methods at face value. On the logistic models the
  ground-truth metrics only ask "is the attribution proportional to the coefficients?", so slope-type methods
  win by construction and contribution-type methods (SHAP, input x gradient) lose by construction: SHAP's rank
  correlation is -0.093 on GMSC and -0.018 on German credit.
