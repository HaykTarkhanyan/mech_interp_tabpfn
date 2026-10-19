# fe_mech_pipeline

Experimental core of the thesis proposal "Feature Effects as Behavioral Specifications for
Mechanistic Analysis of Tabular Foundation Models" (`../reviews/proposal_as_reviewed.md`),
built to the requirements of the review (`../reviews/feature-effects-proposal-review.md`,
sections 3 and 6, and section 8 of the HTML version).

The idea in one line: describe a behavior of TabPFN v2 from the outside with ALE (a main-effect
curve, an interaction surface), ablate parts of the network, and measure how much that behavior
changes, against a random-component null, an estimator noise floor, and a non-target behavior.

## Files

| File | What it is |
|---|---|
| `common.py` | paths, logging (console + `logs/<script>.log`), 4 torch threads, seed 509, JSON writer |
| `tasks.py` | synthetic tasks: additive control (gamma = 0), interaction (gamma = 1.5), flat main effects; column placements |
| `ale.py` | first- and second-order ALE in numpy, the fixed-grid distance, grid jitter and interpolation |
| `model.py` | loads the TabPFN v2 checkpoint by name; `HookedTabPFN` with mean / zero / patch ablation of heads and MLP units |
| `delta_b.py` | Delta_B, null percentiles, noise floor, selectivity, and the pre-registered decision rules |
| `validate_ale.py` | script: plain-assert checks of the ALE code on closed-form functions |
| `check_model.py` | script: loads the model, verifies the wrapper and the hooks, times a forward pass |
| `run_pilot.py` | script: the pilot (behavior check, layer scan, null, floor, placements) -> `results/pilot.json` |
| `report.py` | script: builds `results/pilot_report.html` from the JSON files |
| `RUNPOD.md` | what needs a GPU, estimated hours, and how to run it there |
| `results/` | `validate_ale.json`, `check_model.json`, `pilot.json` (raw pilot results), `pilot_report.html`; `pilot_smoke.json` is the wiring test, not results |

## Run

```bash
uv sync                                  # exact pins in pyproject.toml / uv.lock
uv run python validate_ale.py            # ~2 s
uv run python check_model.py             # ~30 s (first run also downloads the 29 MB checkpoint)
uv run python run_pilot.py               # ~5 min target on an idle laptop (measured below)
uv run python report.py                  # ~5 s, writes results/pilot_report.html
```

Measured runtimes are in the "Pilot results" section at the end.

## Design in brief

- **Model:** the original released TabPFN v2 classifier, `tabpfn-v2-classifier.ckpt`, loaded by
  name (tabpfn 8.2.0 would otherwise load v3, or a fine-tuned v2 file). One estimator, wrapper
  preprocessing off, so input column j is model column j. DECISIONS #9, #10.
- **Task:** Pr(Y=1|X) = sigmoid(X1 + (X2^2 - 4/3) + 1.5 X1 X2 1{X3 > 0}), with X1, X2, X3 and
  one noise column iid Uniform(-2, 2). Context 256 rows. Also an additive control (gamma = 0)
  and a flat-main-effect stress task (h1 = h2 = 0).
- **Behavior B(g_D):** on the log-odds scale (DECISIONS #11). Target = second-order ALE of
  (X1, X2), the pure interaction surface, 4 x 4 cells. Non-target = first-order ALE of X2,
  8 bins. The first-order ALE of X1 is reported too. Grid fixed by the population, reference
  data = 128 fresh rows from the task distribution, 8 per (X1, X2) cell (DECISIONS #12).
- **Ablation:** mean ablation of per-head slices before each attention output projection
  (feature attention and item attention) and of MLP hidden units; zero ablation as contrast
  (DECISIONS #13).
- **Delta_B:** relative integrated L2 on the fixed grid (DECISIONS #14), with a random-component
  null of matched size and type (#15), a no-ablation noise floor (#16), and selectivity
  S = Delta_target / max(Delta_nontarget, floor_nontarget).

## Pre-registration

Written 2026-10-19 at 14:00 local time, before `run_pilot.py` was run for the first time.
Seen before registering: the output of `validate_ale.py` and of `check_model.py` (including the
prediction-level effect of ablating layer 0). No Delta_B, null or floor value had been computed.
The same text is DECISIONS #17; the rules are implemented in `delta_b.py` and applied by
`report.py` without manual judgment.

**Pilot design (fixed in advance).** Task "interaction" (gamma = 1.5, t = 0), one context of 256
rows, seed 509. Candidate sets for every layer L = 0..11: all 6 feature-attention heads of L, all
6 item-attention heads of L, all 768 MLP units of L; mean ablation. Null: 25 random sets per type
of matched size (6 heads of the same attention type, or 768 MLP units, anywhere in the network).
Floor: 7 joint no-ablation resamples (new context, new reference rows, jittered grid), plus 3
each of the single sources. Both placements use the same context rows, reference rows, null sets
and floor draws, with columns permuted.

**Pass rule.** A candidate set passes iff
1. Delta_target > floor_target (the largest joint-resample Delta of the interaction surface),
2. its Delta_target is at or above the 95th null percentile (it must beat at least 24 of the 25
   matched random sets), and
3. selectivity S > 1 (the interaction changes more, relatively, than the X2 main effect or its floor).

**H1. The X1*X2 interaction is carried mainly by feature-attention heads, not item-attention
heads.** Rationale: feature attention mixes features within a row, which is where a pairwise
feature interaction must be computed; item attention moves information between rows (the
estimation from context). Rule, per placement: if no attention set passes, H1 is inconclusive;
otherwise the passing attention set with the highest S decides: a feature-attention set means
supported, an item-attention set means contradicted.

**H2. It matters whether X1 and X2 sit in the same feature pair (columns 0, 1) or in different
pairs (columns 0, 2).** Rationale: TabPFN v2 embeds features in consecutive pairs, and pair
tokens let a single layer represent XOR-type interactions (Bilos et al.), so the interaction may
be implemented differently when X1 and X2 share a token. Rule: if no attention set passes in
either placement, inconclusive; if the sets of passing attention components are identical, not
supported; if they differ and at least one set that flips changes its Delta_target by more than
the larger of the two placements' floors, supported; otherwise inconclusive.

**Non-binding expectations,** written down so that they cannot be claimed as predictions later
if they fail: ablating layer 0 will damage everything (Balef et al.), so it should show a large
Delta_target with S near 1; single middle layers may show small Delta because later layers
repair the damage, which would make H1 inconclusive rather than false.

**What these verdicts are not.** One context draw, one task, 25 null sets. A verdict here is a
pilot signal for designing the main study, not evidence for or against the hypotheses in general.

## Pilot results

Run 2026-10-19, 14:07-14:20 local time, `results/pilot.json`, report `results/pilot_report.html`.
Everything below is one context draw of 256 rows. Read it as a test of the pipeline that also
produced some numbers, not as findings about TabPFN.

**Verdicts by the pre-registered rules:** H1 same pair **supported**, H1 different pairs
**contradicted**, H2 **supported**. Each is weaker than its label:

- **H1, same pair.** The only passing attention set is feature attention, layer 3:
  Delta_target 0.437 against a floor of 0.405, null percentile 100, S = 1.12. Both margins are thin.
  Item attention layer 3 changes the interaction more (0.756, S = 1.13) but sits at the 92nd
  percentile of a very wide item-attention null (p95 0.96).
- **H1, different pairs.** Item attention layers 2 and 3 pass (Delta_target 1.05 and 1.03, S 2.08
  and 1.55, both at the 96th percentile). Item layer 2 also changes the X1 main effect by 1.14, more
  than the interaction, so it is selective against X2 (the pre-registered non-target) but not
  against X1. Feature attention layer 9 comes closest among feature sets (0.548 vs floor 0.556).
  With the verdict flipping between placements, there is no consistent support for H1.
- **H2.** The passing sets differ between placements, and item attention layer 2 differs by
  0.64 between them, above the floor of 0.556. But both placements use the same single context, and
  the floor is dominated by context variability (next point), so this is weak.

**What matters more than the verdicts:**

1. **The behavior itself moves a lot between contexts.** With no ablation, the interaction surface
   changes by a median 0.47 (same pair) / 0.59 (different pairs) of its own size when the 256-row
   context is redrawn, against 0.06 / 0.01 for a new reference sample and 0.05 for grid jitter. The
   floor in the pass rule (largest of 7 joint resamples) is 0.405 / 0.556, and only 5 of 24
   attention sets exceed it in the same-pair placement. At this context size, context variability of
   g_D, not ALE estimation, is the binding constraint.
2. **Mean and zero ablation disagree.** Spearman rank correlation of Delta_target over the 36
   same-pair sets is 0.28. Examples: feature attention layer 0 gives 0.01 (mean) vs 0.45 (zero); MLP
   layer 7 gives 0.84 vs 0.15. Which components "matter" depends on the replacement value, as the
   proposal's section 6 warned.
3. **The MLP null measures concentration.** 768 MLP units scattered across layers barely move the
   behavior (null p95 0.15 / 0.20), so 10 of 12 (same) and 11 of 12 (different) whole MLP layers
   beat it. The expected failure mode from DECISIONS #15 happened; MLPs need a within-layer null.
4. **Layer 0.** Mean-ablating the layer-0 MLP is catastrophic (mean |change in log-odds| 6.5,
   accuracy against the Bayes rule 0.52, chance level), in line with Balef et al.'s first-layer
   result. Feature attention in layers 0-2 does almost nothing to the interaction (Delta_target <= 0.04).
5. **Behavior check.** The model recovers the interaction roughly (||ALE_12|| 0.73 vs 0.80 for the
   truth, relative distance 0.38). In the gamma = 0 control it still shows an interaction of 0.23,
   about a third of the interaction task's. In the flat task the interaction is 0.56 vs 0.80 and the
   main effects stay near zero, as designed.
6. **Unexplained:** in the different-pair placement a new reference sample moves the interaction
   surface by only 0.006, against 0.06 in the same-pair placement. One reading is that the model's
   interaction there barely depends on X3 (the data's gating variable). Not checked.

**Implications for the main study** (also in RUNPOD.md): estimate B over several contexts, or use
larger contexts, before making ablation claims; report every result under at least two
replacement values; give MLPs a within-layer null; use both main effects as non-targets.

## Measured runtimes

Laptop, Windows 11, 4 torch threads, 2026-10-19. The CPU was shared with other processes
(antivirus scan, other tools) for most of the session, which made forward passes 2-4 times slower
than on the idle machine.

| Script | Runtime | Notes |
|---|---|---|
| `validate_ale.py` | 0.2-0.9 s | numpy only |
| `check_model.py` | 31-42 s | +13 s on the first run for the checkpoint download |
| `run_pilot.py` | 782 s (13.0 min) | 293 forward passes, 2.65 s per pass; an idle machine should manage ~1 s per pass, so ~5 min (estimate, not measured) |
| `run_pilot.py --smoke` | 131 s | 45 passes; wiring check, output `results/pilot_smoke.json` |
| `report.py` | 1-17 s | the first build includes cold imports |

Forward pass at the pilot size (256 context + 1,152 query rows): 2.1-3.9 s while loaded; not
measured idle, estimated ~1 s from 0.76-0.89 s at 256 + 864 rows (idle). At 512 context + 2,304
query rows: 6.2-7.6 s.
