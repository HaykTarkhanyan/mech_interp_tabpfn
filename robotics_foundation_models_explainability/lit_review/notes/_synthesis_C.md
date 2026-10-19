# Synthesis, batch C (34 papers: failure, faithfulness, security, generative, by-design, methodology, adjacent)

## Failure prediction from internals (11 papers)

Established: last-layer VLA features carry linearly or shallowly decodable failure information across OpenVLA, pi0, pi0-FAST and pi0.5 (2506.09937, 2605.30834, 2609.04277, 2606.29699). Recurrent detectors beat per-step MLPs (2605.30834, Table 4), and replacing uniform trajectory labels with contrastive or sparse timestep supervision helps (2605.30834, 2609.04277). Token-uncertainty baselines are near chance everywhere they are tested.

Contested: whether internals are needed at all. SAFE's own ablation shows DINOv2+CLIP image features match VLA features for the LSTM on a real Franka (2506.09937, Table 7). Tri-Info gets the same in-domain accuracy from executed actions and proprioception as from embeddings (0.880 vs 0.881), with internals helping only under shift (2606.19998). VLA-Scope gains only 0.06 ROC-AUC from hidden states over simple action statistics (2609.21246). GUARD (2608.04510) and VLA-FAIL (2606.21386) suggest the useful internal signal is grounding or novelty, not "knowledge of failure".

Weaknesses: offline evaluation on recorded rollouts (no paper except ProbeAct and a Hide-and-Seek appendix closes the loop); "unseen tasks" are usually other tasks from the same LIBERO suite; seeds resample splits of one rollout pool; episode-length shortcuts (failures run to the time limit) are guarded only in some papers (SAFE, Tri-Info, FIPER warns about it). Metrics are incomparable: SAFECAST's "ROC-AUC" is an alpha-sweep area with values below 0.5 (2608.04246). 2606.29699 is the most honest: AUROC 0.97 retrospectively, but over 3 false warnings per clean episode.

Open: where in depth failure information appears (layer sweeps exist only for OpenVLA: flat across layers in 2605.30834, three layers in 2606.29699); whether any failure direction is causally used.

## Faithfulness of reasoning (5 papers)

Established: text placed before the action is causally load-bearing. Injecting a clean chain against a corrupted instruction recovers 47.8 points on LIBERO-Goal (pre-registered, 2603.12717); corrupting a driving CoT collapses F1 from 72% to 12% (2609.34794). Equally established: the link is partial. The pre-registered residual was 38 points, not 5-15; reasoning VLAs often write the right plan and execute the wrong action (2510.16281); a pi0.5-derived action expert ignores planned waypoints almost entirely (0.6% of noise-level sensitivity, erasure costs 0 points) until a deep conditioning path is added (2609.30833). 2607.04681 supplies the conceptual split between functional and faithful reasoning, but its own method only optimizes LLM-judged consistency.

Contested: whether CoT should sit before the action at all (2609.34794 finds decide-then-explain slightly better). Weakness: most evidence is text- or input-level intervention; none localizes where instruction and chain compete inside the network.

## Security through internals (3 papers)

Internal signatures of triggers are real and usable for detection: an SAE feature tracks patch presence (2610.03498); triggered states show homogenized layerwise evidence and deep-layer attention promotion (2607.12571); 2602.03153 shows the attention shift only qualitatively. But every effective fix ends in input-side inpainting, and SAE suppression yields only 1-6 points with large clean-performance costs when applied unconditionally. TrustVLA is the only one with counterfactual validation, paired 500-episode cells and intervals.

## Generative policies (4) and interpretable-by-design (1)

Small-data diffusion policies behave like action lookup tables (2505.05787) - a useful null model. Co-training success tracks sim-real feature alignment that preserves domain separability (2604.13645). The initial noise of flow and DDIM policies is a causal handle on chunk-boundary artifacts, but not under DDPM sampling (2603.11642). WAM vs VLA SAE comparisons are explicitly correlational (2606.01095). LM-X (2608.25757) exposes progress, event and uncertainty signals but does not test whether they drive the actions, the gap the faithfulness papers show is real.

## Methodology and adjacent

2606.30686 argues success rate cannot separate semantic from physical competence; LIBERO-Plus (2510.13626) supplies the behavioral evidence (camera and initial-state collapse, language ignored on LIBERO-Object). The RL, chess and video-model papers (2504.01871, 2406.00877, 2310.08043, 2602.07050) set the standard batch C rarely meets: hypothesis-driven concepts, causal edits with random or norm-matched controls, and warnings that physical variables can be high-dimensional population codes that single-direction steering cannot move (2602.07050). The driving papers add an action-space logit lens (2608.07361) and an SAE ablation that improves EPDMS 0.50 to 0.59 (2607.06328).

## Cross-cutting patterns

1. Models: OpenVLA and pi0/pi0.5 dominate; findings rarely transfer cleanly between discrete-token and flow-expert heads, and several methods are head-specific (GUARD and VLA-FAIL need flow heads; 2603.11642 needs a deterministic noise path).
2. Simulation dominance: LIBERO variants carry most results; real-robot studies use 20-30 trials per cell without intervals, except a few (TrustVLA, Hide-and-Seek counts).
3. Probe accuracy vs control: high decodability repeatedly fails to imply use (failure probes vs image features, early command decodability vs planner compatibility in 2608.07361, waypoints ignored in 2609.30833). Papers that intervene (2603.12717, 2609.30833, 2607.12571, 2603.11642) are far more informative than those that only probe.
4. Sensitivity is not faithfulness: naive goal injection gives 348% sensitivity and 15% success (2609.30833); unconditional SAE suppression hurts clean behavior (2610.03498); steering effects are often one-directional (2310.08043).
5. Almost nothing in the batch identifies a circuit inside a VLA; the closest are attention-level signatures in backdoored models.

## Top 5 papers in batch C

1. 2603.12717 - pre-registered, closed-loop causal test that a VLA's text reasoning steers actions, with the honest finding that it does so only partially.
2. 2609.30833 - best intervention design: shows a flow action expert ignores explicit plans and that high sensitivity can signal a shortcut rather than faithful use.
3. 2607.12571 - most rigorous internals-based security work: shared backdoor signature across two attacks, counterfactual validation, paired 500-episode cells, OpenVLA and pi0.5 including real robot.
4. 2506.09937 - the reference internal-feature failure detector that later work builds on, including its own evidence that generic image features nearly match VLA features.
5. 2606.19998 - key negative control: in-domain failure prediction needs no model internals, which reframes what failure-from-internals papers actually show.
