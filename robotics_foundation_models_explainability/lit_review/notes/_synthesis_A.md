# Synthesis A: retention, probing, sparse autoencoders (22 papers)

## Retention: what action training does to the VLM

**Looks established.** Action fine-tuning lowers how decodable the inherited VLM information is. Evidence: an ImageNet-100 probe drop in OpenVLA (79.9 to 77.5%, 2510.25616), lower depth decodability at every layer plus a late-layer "cliff" in MolmoAct2 against its weight-matched base (2608.08904), and near-chance action-based answers on richer semantic categories (2606.19297). Simple perceptual concepts such as colour survive (2510.25616, 2606.19297). VQA co-training (Magma) goes with better retention (2606.19297), but this is an association across about 7 checkpoints.

**Contested.** Is the information lost, or just not read out? 2606.19297 finds answer information above chance in middle layers that fades toward the action layers. 2608.08904 recovers most of the depth cliff by zeroing the last MLP writes, which only returns the model to its layer-32 level. Both VLM-vs-VLA comparisons pit text answers against cube placement. The headline gap also fails for pi0: its base PaliGemma is at chance on everything, including colour, where pi0 scores 86%. Repairs disagree too. An alignment loss improved simulated OOD success (2510.25616), but injecting VGGT geometry into GR00T closed the depth-probe gap without significant success gains under standard fine-tuning (2605.24642).

**Weaknesses.** One model per paper, simulation only, a single training seed, and readout channels that do not match.

**Open.** Does the late-layer drop matter for behavior? Do knowledge-insulated models (pi0.5) retain more?

## Probing: what VLAs encode

**Looks established.** Almost everything people probe for is linearly decodable at high accuracy: symbolic state (2502.04558), progress (2608.13474), eventual success (2605.28527), position, contact and distance (2610.20840), and robot state (2603.05487). But decodability is cheap and often not specific to the policy:
- Progress is readable from layer 0 and from SigLIP (2608.13474).
- Success is predicted almost as well from DINOv2 and CLIP as from the VLAs (task-split R2 0.51 vs 0.55, 2605.28527).
- Time and arm pose alone predict object state on natural trajectories (2609.34684).

Decodable is also not the same as used:
- The instructed target is 96-100% decodable, yet pi0.5, GR00T and two world-action models go to the scene-implied object in 86-98% of trials, including on a real robot (2610.06235).
- Prompt category is 99.3% decodable with no behavioral effect (2603.19233).
- History is decodable but used only when the current frame is blacked out (2607.03372).

**Contested.**
- **Steerability of probe directions.** It works for action-adjacent quantities: lift, grasp choice and object choice on hardware (2603.05487), and gradient-through-head directions (2609.30996). It fails for progress at layer 0 (2608.13474), and static probe directions can even reverse the effect (2609.30996). 2610.20840 explains much of this. Edits to "belief" geometry, such as the wrist arc, make the policy compensate against the edit. Edits derived from the action side change the command.
- **Language sensitivity of pi0.5.** It ignores prompts in 2603.19233 (LIBERO-Goal 83.3 to 80.0% with a null prompt, ANOVA p=0.247), but collapses under random sentences in 2610.06235 (98.0 to 11.7%). The baselines differ, so checkpoint and protocol matter.
- **World-model claims.** 2509.24559's claim is confounded by task identity and gives R2 of only 0.09-0.18 at K=1.

**Weaknesses.** Probes trained only on successful demonstrations, often without a control task or non-VLA baseline. Mean pooling. Tiny datasets (50 episodes in 2502.04558, 400 frames in 2608.08904). Controlled counterfactual tests are rare; 2609.34684 is the exception.

## Sparse autoencoders and dictionary features

**Looks established.**
- Single features can change closed-loop behavior, including on real robots (2603.19183, 2605.17204, 2610.20767).
- Practical constraints recur. Mean-pooled SAE reconstructions destroy pi0.5 success (2603.19233). Inserting an SAE into discrete-token OpenVLA gives 0% success except at the last layer (2603.19233, 2605.17204). Offline FVE does not predict closed-loop fidelity (2605.17204).
- In pi0.5, single-feature edits in the backbone barely matter, while the action expert is hypersensitive (2605.17204, 2603.19233).

**Contested.** Claims of interpretability and generality rest on author labelling (79% interpretable, 2603.19183) and on ablation tiers that are not matched on activation norm or frequency. 2605.17204 shows that random action-expert features also collapse the policy. Dictionary choices vary widely and confound comparisons: 1x width (2603.19183, 2605.17204 for pi0.5), 4-8x (2603.19233), and block-sparse features trained on 52M unpooled tokens (2610.20767). Block-sparse features are more compact than SAEs, but SAE edits transfer better for visual steering (2610.20767).

**Weaknesses.** 5-25 trials per condition, hand-tuned steering strengths, almost no statistics. Two papers in this section are not SAEs: 2602.01834 uses supervised PCA concept vectors and 2603.05487 uses probes. The unlearning result in 2610.09496 lacks a plain fine-tuning baseline.

**Open.** Norm-matched random-feature controls, transfer across checkpoints, and a principled choice of intervention site.

## Cross-cutting patterns

- **Models.** pi0.5 appears in 12 of 22 papers and OpenVLA in 13 (one only in an appendix). Action-head type matters: SAE insertion fails on discrete tokens, the flow expert is hypersensitive to edits, and OpenVLA collapses when its prompt template is knocked out (2605.30117). Few papers compare head types directly (2603.19233, 2603.05487, 2605.17204).
- **Simulation.** LIBERO appears in 16 papers. Its fixed scenes reward memorizing the scene, which confounds both language and state probes. Real-robot closed-loop evidence comes from 7 papers; 2610.20840 uses DROID offline only.
- **Accuracy vs control.** Probe accuracy and causal control come apart again and again. Interventions succeed when they are action-proximal (late layers, the action expert, gradient-derived directions) and fail or trigger compensation when they target perceived state.
- **Statistics.** Only a minority report CIs or tests: 2607.03372, 2609.34684, 2605.28527, 2603.19233, 2605.24642 and 2510.25616 (ablations only). Single training seeds are the norm.
- **Baselines.** Generic vision features carry much of what gets decoded, so non-VLA encoder baselines should be standard.

## Five most important papers

1. **2610.20840:** controlled sweeps plus interventions show concept geometry, and separate belief edits (which the policy compensates against) from action edits. Replicated across three models on DROID.
2. **2609.34684:** the key methodological result. Natural probe accuracy can hide weak responsiveness to the target variable, which undercuts most probe claims in this batch.
3. **2603.19233:** six architectures and 394k closed-loop episodes. Vision dominates, language matters only when the scene is ambiguous, and it sets practical SAE constraints.
4. **2610.06235:** clean "encoded but not in control" demonstration, extended to world-action models and a real robot.
5. **2610.20767:** the most rigorous dictionary-learning work. Unpooled 52M-token corpus, matched SAE baselines, closed-loop steering, and feature-guided training-data masking.
