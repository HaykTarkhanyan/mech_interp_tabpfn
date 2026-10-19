# Reading-note schema

One JSON object per paper in `lit_review/notes/<arxiv_id or slug>.json`, written while reading the full
text in `papers/text/`. The review is written from these notes, so they must separate what the paper
claims from the reader's own assessment.

```json
{
  "id": "2509.00328",
  "title": "exact title",
  "first_author": "Haon",
  "year": 2025,
  "venue": "CoRL 2025 / arXiv / workshop name",
  "tier": "core | adjacent",
  "models": ["OpenVLA", "pi0-FAST"],
  "action_head": "discrete action tokens | flow/diffusion action expert | diffusion policy | regression head | n/a",
  "methods": ["value-vector projection", "activation steering"],
  "question": "the question the paper asks, one sentence",
  "key_findings": ["finding with the number that supports it, e.g. 'probe AUROC 0.97 at layer 16'"],
  "evidence": {
    "setting": "simulation | real robot | both | offline dataset only",
    "benchmarks": ["LIBERO-Long", "SimplerEnv", "DROID"],
    "scale": "rollouts / seeds / episodes per condition, as reported",
    "statistics": "CIs or tests reported? yes / no / partial, with detail",
    "baselines": "what the method is compared against"
  },
  "causal_validation": "do they test explanations by intervening (ablation, steering, patching)? how, and what changed?",
  "closed_loop": "are effects measured in closed-loop execution, or only on open-loop predictions?",
  "limitations_stated": ["what the authors say"],
  "critique": ["reader's own assessment, not in the paper"],
  "code": "URL or 'not released'",
  "one_line": "plain-language summary in one sentence"
}
```

Rules: numbers come from the paper's text or tables, never from memory. If a field is not reported,
write "not reported", not a guess. ASCII only: straight quotes, hyphens instead of em or en dashes.
