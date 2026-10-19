# Transfer pilot v2: cost estimate

From the measured timing runs (one task per world, 2 threads, this laptop). Parallel = replicates split across processes of 2 threads each; ideal speed-up, so treat it as a lower bound.

- Derived: mlp: no timing run; reality and oracle tasks derived from timing_rf.json (+6.4s for exact MLP truths)

| Tier | Realities | reps x k, k_reality | CPU time (2-thread hours) | Laptop, 1 process | Laptop, 3 processes |
|---|---|---|---|---|---|
| Tier 0: go/no-go | rf | 4 x 4, 30 | 1.3 | 1.3 h | 0.4 h |
| Tier 1: full pilot | rf, mlp | 8 x 5, 30 | 6.1 | 6.1 h | 2.0 h |
| Tier 2: paper scale (x5 datasets, 3 realities) | rf, mlp, mlp | 10 x 5, 30 | 56.5 | 56.5 h | 18.8 h |
