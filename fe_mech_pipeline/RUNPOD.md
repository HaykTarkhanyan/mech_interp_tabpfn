# GPU plan (RunPod)

Nothing here has been run. This file lists what the laptop pilot could not do, how many GPU
hours it is likely to take, and how to run it. Hour figures are planning estimates, not
measurements; the first step on any pod is to measure them (step 2 below).

## Why a GPU

On the laptop (4 threads, Intel CPU), one TabPFN v2 forward pass with a 512-row context and 2,304
query rows took 6.2-7.6 s (`check_model.py`, first run). The pilot itself (256-row context, 1,152 query rows, 293 passes)
took 13.0 minutes at 2.65 s per pass with the CPU shared with other processes. The pilot therefore uses a 256-row
context, 128 reference rows, one context draw and 25 null sets per type (DECISIONS #18). The
main study needs roughly 1,000 times more forward passes.

Planning number for a modern data-center GPU (A100, L40S, H100): about 0.1 s per pass at 512
context + 2,304 query rows. Basis: ~160 GFLOP per pass (12 layers x [~10 GFLOP of projections and
MLP over 8,448 tokens + ~3 GFLOP of item attention]), which the laptop runs at ~25 GFLOP/s; a GPU
running small matrices at a few TFLOP/s effective, plus Python and hook overhead, lands near
0.05-0.1 s. Unverified until measured.

## What the laptop pilot changes in this plan

- With 256 context rows the interaction surface itself moves by about half its size between
  context draws (median 0.47-0.59), so per-context ablation claims are fragile. Prefer contexts of
  512-1,024 rows and many context draws, and consider averaging B over contexts before ablating.
- Mean and zero ablation ranked the layer sets differently (Spearman 0.28), so item A runs a third
  replacement (resample ablation) and reports every claim under all of them.
- The scattered-unit MLP null was beaten by 10-11 of 12 MLP layers; add a within-layer null
  (random subsets of each layer's units) for MLPs.
- One passing set was selective against the X2 main effect but not against X1; use both main
  effects as non-targets (S = Delta_target / max over non-targets).

## Work items

### A. TabPFN v2 main study: Delta_B over many contexts (RQ2, RQ3, H1, H2)

Per context draw and condition, about 1,000 forward passes:

| Piece | Passes |
|---|---|
| Layer scan, mean ablation (12 layers x feature attn / item attn / MLP) | 36 |
| Layer scan, zero ablation | 36 |
| Single heads, mean ablation (72 feature + 72 item) | 144 |
| Null, 200 random sets per type for the layer sets | 600 |
| Noise floor: 50 joint + 20 each of query / grid / context | 110 |
| Resample ablation of the layer scan (third replacement value) | 36 |
| **Total** | **~960** |

Conditions: interaction task at gamma 0.75, 1.5, 3.0 x two placements x context sizes 256, 512,
1024 (18); the flat-main-effect task, two placements, context 512 (2); the gamma = 0 control,
two placements, context 512 (2). With 20 independent contexts each: 22 x 20 x 960 = ~420,000
passes. At 0.1 s per pass that is ~12 GPU-hours; context 1,024 passes cost 2-4 times more and
the reference set should grow to 256-512 rows, so **budget 20-40 GPU-hours** on one A100-class GPU.

Engineering that would cut this by an estimated 5-10x, worth doing first: TabPFN v2 accepts a batch
dimension (several datasets per forward pass), but the hooks in `model.py` ablate the same units
in every batch element. A per-batch-element unit mask would run 16-32 null sets in one pass.

New code needed (not written yet): a `run_main.py` that loops `run_pilot.py`'s per-placement
logic over contexts and conditions and writes one JSON per (condition, context), and report
sections for across-context reuse (RQ3): how often each component passes across the 20 contexts.

Also needed for the flat task: a non-target distance that does not divide by a near-zero norm
(DECISIONS #14), for example the absolute change in the main-effect curve in units of the
unablated interaction norm.

### B. nanoTabPFN prior ablation (review item 4)

Pretrain nanoTabPFN on a prior with feature interactions and on one without, everything else
fixed, then run the same Delta_B pipeline on both. Balef et al. report 11.9 A100-hours to
pretrain a 1-layer nanoTabPFN and 62.3 A100-hours for 6 layers (10,000 steps, batch 512,
TabICL prior; their Table A.10). Estimates, scaled from those numbers:

| Variant | Models | A100-hours |
|---|---|---|
| Feasibility pilot: 2 priors x 1 seed, 6 layers, 2,000 steps | 2 | ~25 |
| Research quality: 2 priors x 3 seeds, 6 layers, 10,000 steps | 6 | ~370 |

Run the feasibility pilot first and measure. New code needed: a prior generator with and without
interaction terms, the nanoTabPFN training loop (from the nanoTabPFN repository), and a model
adapter so `HookedTabPFN` can hook nanoTabPFN's attention output projections and MLP (same sites,
different module names). Whether nanoTabPFN embeds features in pairs determines whether H2 can be
tested on it at all; check this before training.

### C. Not GPU-bound, but deferred

- RQ1 sharpened ("does the estimand change the answer?"): repeat A with a second behavioral
  estimand (pairwise Shapley-Taylor, or GADGET regions). Cheap once A exists.
- A paired bootstrap of Delta_B as a tighter alternative to the floor (DECISIONS #16): free, uses
  existing predictions.

## How to run on a pod

1. Create a pod from a PyTorch template (CUDA 12.x). Any GPU with 24 GB or more is plenty: the
   checkpoint is 29 MB and activations at these sizes are well under 1 GB. Attach a volume for
   `results/`.
2. Set up and measure before planning hours:
   ```bash
   git clone <this repo> && cd <repo>/fe_mech_pipeline
   pip install uv && uv sync            # the lock is cross-platform; Linux torch wheels include CUDA
   uv run python -c "import torch; print(torch.cuda.is_available())"
   uv run python validate_ale.py
   uv run python check_model.py         # downloads the v2 checkpoint, verifies hooks, times a pass
   ```
   `check_model.py` and `run_pilot.py` take `--device cuda`. The thread cap in `common.py`
   (4 threads, set for the laptop) does not limit GPU work.
3. Reproduce the laptop pilot on the GPU first and compare `results/pilot.json` with the laptop's
   numbers (expect agreement to float tolerance, not bitwise):
   ```bash
   uv run python run_pilot.py --device cuda
   uv run python report.py
   ```
4. Scale up with the larger settings already exposed by `run_pilot.py`, for example
   `--n-ctx 512 --n-per-cell 16 --n-null 200 --floor 20,20,20,50`, then the new `run_main.py`.
5. Copy `results/` back and stop the pod.
