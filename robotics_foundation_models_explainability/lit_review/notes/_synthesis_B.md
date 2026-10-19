# Synthesis B: steering and editing; attention, layers and causal structure

24 papers read (10 steering, 14 computation). Per-paper evidence is in the JSON notes; this file only draws the cross-paper picture.

## 1. Activation steering and editing

**What looks established**
- Interventions on internal states change closed-loop behavior of VLAs. FFN-neuron upweighting shifts speed and height in discrete-token models (2509.00328); subspace or probe-direction steering of flow-matching action experts raises task success (2605.17144, 2610.04681); optimal-transport maps shift speed and height in pi0.5 and SmolVLA (2607.14280); weight-space task-vector negation switches skills off (2608.04692).
- Strong fixed-strength steering costs task success. Static C=20 drops LIBERO success from 71% to 27% (height) and 2% (speed) (2606.00269), and DiMaS needs an interpolation knob for the same reason (2607.14280).
- Steering toward a policy's default tendency is easier than against it: "slow" and "low" work on a real UR5 while "fast" and "high" do not (2509.00328), and decreasing speed or height is the reliably significant direction in DiMaS.

**Contested or contradictory**
- Linear steering. DiMaS reports that mean-difference and regression steering fail on flow-matching experts ("linearly decodable but not linearly steerable"). Yet CAA gives partial gains in COAST (+0.04 to +0.10) and is competitive at matched sites in RoboIRS (56.5% vs 66.2%). The likely reconciliation is target (continuous kinematic features vs binary success) and site choice, but nobody has tested this directly.
- Meaning of steering directions. RoboIRS's best sign pushes along the failure-classifier direction in 6 of 15 tasks; CTRL-STEER's vocabulary-chosen "up" neurons lower the arm (2606.00269); Robotic Steering's appendix finds random heads almost as task-discriminative as selected ones (2511.22697). Control works more often than the semantic label does.
- Locality. Only 2608.04692 measures what an edit does to other skills: targets go to 0/20, but mean control retention is 52% and most edits zero at least one unrelated skill. No activation-steering paper reports this.

**Common weaknesses**: LIBERO-only or sim-only evaluation; hyperparameters (layer, strength, sign) tuned on rollouts of the target task; undertrained "early checkpoints" as baselines (COAST); random-direction controls rare (RoboIRS, LAE/CLAE, 2608.04692 have them); real-robot claims on 10-15 trials without tests. 2605.30695 labels its own key figures "illustrative pending verification" and should not be cited as evidence yet. The two multirobot papers (2509.20623, 2606.11489) steer 30-dim RL controllers, not foundation models.

**Open questions**: where to steer in flow experts (late expert layers in DiMaS, layer 11 of 18 in COAST; early-middle expert layers in DiMaS steer the wrong way); whether success-labelled steering is interpretability or few-shot test-time adaptation; whether steering, like negation, masks rather than removes behavior.

## 2. Attention, layers and causal structure

**What looks established**
- On LIBERO-style benchmarks, language is a weak and shallow input. VLAs complete tasks under contradictory instructions at over 90% (2603.06001); most language-backbone blocks can be dropped and recovered (2606.27755); action experts are insensitive to rephrasing and absent objects (2610.10178); four VLAs fail all 24 feasible new instructions and substitute trained behaviors (2610.05878). 2609.39971 sharpens this into "instruction-action binding": the instruction retrieves a memorized trajectory family (69% of 381 probes), the task-conditioned KV state encodes the executed task (34/45 top-1), and injecting another task's state redirects the arm (109/200 vs 0/200 controls, replicated on GR00T).
- The backbone-to-expert channel is narrow. Interchange interventions in pi0.5, MolmoAct2 and MolmoBot place the conditioning in 1-3 mid-to-late layers carrying target location, and pruning to them keeps closed-loop success (2610.20750). EWAM finds a similar depth-ordered handoff with masking and content injection in its own architecture (2609.39973).

**Contested**
- "Deep layers ignore vision" (2603.15618) rests on masking visual tokens per layer with open-loop MSE; 2610.20750 shows location information migrates into text and, in pi0.5, image positions, so low late-layer sensitivity is expected from routing and is not a deficit.
- Attention as signal vs explanation. Attention heads work as monitors (2603.13782 navigation deviation, 2606.09749 target identification with success AUC 0.89, 2610.05878 failure stopping at 87-100%), but none of these test the heads causally. Attribution patching is uncorrelated with activation patching for pi0.5 (r = 0.089, 2610.10178), and attention saliency is less faithful than masking-based attribution (2605.00321).

**Common weaknesses**: open-loop metrics (flow loss against demonstrations, action MSE) used to make claims about behavior; single-frame or episode-start analyses; LIBERO saturation; tiny failure counts behind detection rates. 2609.39971 documents that LIBERO-PRO's released code delivered the original instruction in Semantic and Task cells, which affects any paper reporting those cells.

**Open questions**: do the binding and narrow-interface findings hold for generalist checkpoints rather than LIBERO fine-tunes; which components inside the interface implement the task key; can monitors built on attention be validated causally.

## 3. Cross-cutting patterns
- pi0.5 is now the default subject (about 15 of 24 papers), then OpenVLA/OFT and GR00T. Flow-matching experts dominate, so tools built on vocabulary projection (2509.00328, 2606.00269) do not carry over; X-VLA use in 2606.00269 is unexplained.
- Decodable is not controllable: 100% linear separability without linear steerability (2607.14280), probe AUC 0.88 with sign-ambiguous directions (2610.04681), success-predictive attention with no causal test (2606.09749).
- Closed-loop causal tests are becoming normal but remain coarse (whole-cache swaps, layer-band masks); component-level circuits in VLAs are still missing.
- Simulation dominates; real-robot results are small and untested statistically. Only 2608.04692, 2609.15005 and 2609.39971 report intervals seriously.

## 4. Five most important papers
1. 2610.20750 - hypothesis-driven interchange interventions across three VLAs with predicted null cells, plus a closed-loop architectural consequence and an off-interface control.
2. 2609.39971 - explains benchmark-vs-counterfactual failure as language-keyed trajectory retrieval, with closed-loop state injection in two architectures, a targeted fix, and a benchmark bug report.
3. 2608.04692 - the most rigorous closed-loop audit: edits suppress targets but locality is fragile and the effect is masking, a warning for all steering claims.
4. 2605.17144 - steering across flow, token and diffusion heads against CAA, SAE and SFT baselines in sim and on a real robot (with the undertrained-baseline caveat).
5. 2607.14280 - direct evidence that linear steering fails in flow-matching experts and that steering sign depends on layer.
