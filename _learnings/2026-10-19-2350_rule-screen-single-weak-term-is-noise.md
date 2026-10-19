# A relevance screen that passes on one weak term is mostly noise; split terms into strong and medium

**Symptom.** The first robot-term AND interpretability-term screen passed 669 of 1,192 records. Of those,
456 matched a single interpretability term, and a random sample was mostly off-topic:

```
'steer'        -> "A Multi-Vine Soft Robot Enabling ... Steering", inference-time output steering
'interpretab'  -> VLA papers that call themselves "interpretable" in passing
'\bprob(e|es|ing)\b' + robot term 'VLA' -> "A VLA-GMRT Look at 11 Powerful FR II Quasars"
```

"VLA" also means the Very Large Array radio telescope: 28 astronomy records matched in this first run, 51 in
the full search.

**Fix.** Strong terms (mechanistic, sparse autoencoder, linear probe, activation patching, steering vector,
neurons, attention heads, internal representations, circuits) pass alone. Medium terms (interpretable,
explanation, probe, steer, attribution, counterfactual, concept) need two distinct hits or one in the
title. Plus an astronomy exclusion. Result: 371 of the same 1,192 records pass (413 of 1,520 in the full
search, 437 of 2,245 after reference-list records were added).

**Recall check, by hand.** Reading all 402 titles rejected as "too weak" rescued 17 relevant works (11 core),
for example "From Recovery to Drop-off: How Action Post-training Reduces a VLM's Late-Layer Depth
Decodability" and "TrustVLA: Mechanism-Guided ... Defense". So the tightened rule is still not safe to use
without a manual pass over its rejects.

**Rule.** Inspect a sample of single-term passes before trusting a screen's count, and always title-screen
the "too weak" bucket by hand.
