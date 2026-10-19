#!/usr/bin/env python3
"""Write lit_review/included.json from the manual screening decisions below.

Decisions are made by hand from the screening sheet (titles for all 413 rule passes plus the 402
rejected-as-weak titles, abstracts for borderline ones). Titles are taken from the search records
or the web sweep, never retyped, so download_papers.py can title-check them against arXiv.

Usage (from robotics_foundation_models_explainability/), ~1 s, offline:
    .venv/Scripts/python.exe scripts/build_included.py
"""

import json
import logging
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "lit_review" / "data" / "processed"
RAW = ROOT / "lit_review" / "data" / "raw"

SECTIONS = {
    "retention": "What robot training does to the pretrained VLM",
    "probing": "Probing what VLAs encode",
    "sae": "Sparse autoencoders and dictionary features",
    "steering": "Activation steering and editing",
    "computation": "Attention, layers and causal structure",
    "failure": "Failure prediction from internals",
    "faithfulness": "Faithfulness of reasoning and self-explanation",
    "security": "Security through internals",
    "generative": "Diffusion, flow and world-action policies",
    "by_design": "Interpretable by design",
    "methodology": "Methodology, positions and tools",
    "behavioral": "Behavioral diagnostics (adjacent)",
    "nl_explanations": "Language explanations of robot behavior (adjacent)",
    "classic_xai": "Pre-foundation-model XAI for robot policies (adjacent)",
    "world_models": "World models (adjacent)",
    "rl_agents": "Mechanistic interpretability of RL agents (adjacent)",
    "driving": "Driving models (adjacent)",
    "background": "Background: models, benchmarks and interpretability methods",
}

# Background references cited for context (models, benchmarks, LLM-interpretability methods). Not screened
# works of the review. Each arXiv id is title-checked against the arXiv API before it is written; ids were
# chosen as the works the core papers cite most (lit_refchain.py) plus the model papers they build on.
BACKGROUND = [
    ("2307.15818", "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"),
    ("2406.09246", "OpenVLA: An Open-Source Vision-Language-Action Model"),
    ("2405.12213", "Octo: An Open-Source Generalist Robot Policy"),
    ("2410.24164", "π0: A Vision-Language-Action Flow Model for General Robot Control"),
    ("2501.09747", "FAST: Efficient Action Tokenization for Vision-Language-Action Models"),
    ("2504.16054", "π0.5: a Vision-Language-Action Model with Open-World Generalization"),
    ("2303.04137", "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"),
    ("2503.14734", "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"),
    ("2506.01844", "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"),
    ("2502.19645", "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success"),
    ("2306.03310", "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning"),
    ("2310.08864", "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"),
    ("2403.12945", "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset"),
    ("2405.05941", "Evaluating Real-World Robot Manipulation Policies in Simulation"),
    ("2407.08693", "Robotic Control via Embodied Chain-of-Thought Reasoning"),
    ("2012.14913", "Transformer Feed-Forward Layers Are Key-Value Memories"),
    ("2203.14680", "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space"),
    ("2308.10248", "Steering Language Models With Activation Engineering"),
    ("2312.06681", "Steering Llama 2 via Contrastive Activation Addition"),
    ("2310.01405", "Representation Engineering: A Top-Down Approach to AI Transparency"),
    ("1610.01644", "Understanding intermediate layers using linear classifier probes"),
    ("1909.03368", "Designing and Interpreting Probes with Control Tasks"),
    ("2102.12452", "Probing Classifiers: Promises, Shortcomings, and Advances"),
    ("2210.13382", "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task"),
    ("2309.00941", "Emergent Linear Representations in World Models of Self-Supervised Sequence Models"),
    ("2311.03658", "The Linear Representation Hypothesis and the Geometry of Large Language Models"),
    ("2311.17030", "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching"),
    ("2202.05262", "Locating and Editing Factual Associations in GPT"),
    ("2309.08600", "Sparse Autoencoders Find Highly Interpretable Features in Language Models"),
    ("2307.13702", "Measuring Faithfulness in Chain-of-Thought Reasoning"),
    ("2305.04388", "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting"),
]

# (arXiv id or "web:<title prefix>", section, tier, download). download=True only for arXiv ids.
DECISIONS = [
    # retention
    ("2510.25616", "retention", "core", True), ("2608.08904", "retention", "core", True),
    ("2606.19297", "retention", "core", True), ("2605.15735", "retention", "core", False),
    ("2606.21496", "retention", "adjacent", False),
    # probing
    ("2502.04558", "probing", "core", True), ("2509.24559", "probing", "core", True),
    ("2608.13474", "probing", "core", True), ("2605.28527", "probing", "core", True),
    ("2607.03372", "probing", "core", True), ("2609.34684", "probing", "core", True),
    ("2610.06235", "probing", "core", True), ("2609.02634", "probing", "core", True),
    ("2605.30117", "probing", "core", True), ("2610.20840", "probing", "core", True),
    ("2609.30996", "probing", "core", True), ("2605.24642", "probing", "core", True),
    ("2602.04880", "probing", "adjacent", False),
    ("web:Reaction Without Attribution", "probing", "core", False),
    ("web:Hijacking a 7B robot policy", "probing", "core", False),
    ("web:Layer-Wise Structural and Mechanistic", "probing", "core", False),
    # sae
    ("2603.19183", "sae", "core", True), ("2603.19233", "sae", "core", True), ("2603.05487", "sae", "core", True),
    ("2605.17204", "sae", "core", True), ("2602.01834", "sae", "core", True), ("2610.09496", "sae", "core", True),
    ("2610.20767", "sae", "core", True),
    ("web:When Internal Feedback Goodharts", "sae", "core", False),
    ("web:Goodfire Silico", "sae", "adjacent", False),
    # steering
    ("2509.00328", "steering", "core", True), ("2605.17144", "steering", "core", True),
    ("2606.00269", "steering", "core", True), ("2607.14280", "steering", "core", True),
    ("2511.22697", "steering", "core", True), ("2610.04681", "steering", "core", True),
    ("2606.11489", "steering", "core", True), ("2509.20623", "steering", "adjacent", True),
    ("2608.04692", "steering", "core", True), ("2605.30695", "steering", "core", True),
    ("web:Reading and Steering a Vision Language Action Policy", "steering", "core", False),
    # computation
    ("2610.10178", "computation", "core", True), ("2609.15005", "computation", "core", True),
    ("2610.20750", "computation", "core", True), ("2605.00321", "computation", "core", True),
    ("2604.18000", "computation", "core", True), ("2603.13782", "computation", "core", True),
    ("2606.09749", "computation", "core", True), ("2603.06001", "computation", "core", True),
    ("2610.05878", "computation", "core", True), ("2609.39973", "computation", "core", True),
    ("2606.27755", "computation", "core", True), ("2603.15618", "computation", "core", True),
    ("2609.39971", "computation", "core", True), ("2609.38401", "computation", "core", True),
    ("2610.02813", "computation", "adjacent", False), ("2607.20771", "computation", "adjacent", False),
    ("2606.31382", "computation", "adjacent", False),
    ("web:Block the Heads", "computation", "core", False),
    ("web:Partial Composition, Not Blind Substitution", "computation", "core", False),
    # failure
    ("2506.09937", "failure", "core", True), ("2606.29699", "failure", "core", True),
    ("2608.04246", "failure", "core", True), ("2606.09740", "failure", "core", True),
    ("2606.19998", "failure", "core", True), ("2605.30834", "failure", "core", True),
    ("2609.21246", "failure", "core", True), ("2606.21386", "failure", "core", True),
    ("2609.04277", "failure", "core", True), ("2608.04510", "failure", "core", True),
    ("2510.09459", "failure", "core", True),
    # found only through the reference lists of the downloaded papers (lit_refchain.py)
    ("2503.08558", "failure", "core", True), ("2606.08508", "failure", "adjacent", True),
    ("2607.14943", "generative", "core", True), ("2410.00371", "nl_explanations", "adjacent", False),
    ("2310.16410", "rl_agents", "adjacent", False),
    ("2606.20754", "failure", "adjacent", False), ("2607.27933", "failure", "adjacent", False),
    ("2603.18342", "failure", "adjacent", False), ("2606.23085", "failure", "adjacent", False),
    ("web:Interpretable Signals Reveal Failure-Relevant", "failure", "core", False),
    ("web:Predictable Before Action", "failure", "core", False),
    ("web:Structured Velocity-Field Disagreement", "failure", "adjacent", False),
    # faithfulness
    ("2607.04681", "faithfulness", "core", True), ("2603.12717", "faithfulness", "core", True),
    ("2510.16281", "faithfulness", "core", True), ("2609.34794", "faithfulness", "core", True),
    ("2609.30833", "faithfulness", "core", True),
    ("web:Reasoning We Can TRUST", "faithfulness", "core", False),
    # security
    ("2610.03498", "security", "core", True), ("2602.03153", "security", "core", True),
    ("2607.12571", "security", "core", True), ("2610.09462", "security", "adjacent", False),
    # generative policies
    ("2505.05787", "generative", "core", True), ("2604.13645", "generative", "core", True),
    ("2603.11642", "generative", "core", True), ("2606.01095", "generative", "core", True),
    ("web:How Do World Action Models Use Their Imagined Futures", "generative", "core", False),
    # by design
    ("2608.25757", "by_design", "core", True), ("2607.21049", "by_design", "adjacent", False),
    ("2607.17861", "by_design", "adjacent", False), ("2606.30101", "by_design", "adjacent", False),
    ("2407.08065", "by_design", "adjacent", False), ("2412.03293", "by_design", "adjacent", False),
    ("2609.27753", "by_design", "adjacent", False), ("2210.06650", "by_design", "adjacent", False),
    ("web:Between pixels and policies", "by_design", "adjacent", False),
    # methodology
    ("web:The Hitchhiker's Guide", "methodology", "core", False), ("2606.30686", "methodology", "core", True),
    ("2604.23775", "methodology", "adjacent", True), ("2504.19475", "methodology", "adjacent", False),
    # adjacent literatures
    ("2510.13626", "behavioral", "adjacent", True), ("2602.17659", "behavioral", "adjacent", False),
    ("2506.09930", "behavioral", "adjacent", False), ("2610.06078", "behavioral", "adjacent", False),
    ("2505.03500", "behavioral", "adjacent", False), ("2610.19438", "behavioral", "adjacent", False),
    ("2306.15724", "nl_explanations", "adjacent", True), ("2309.10346", "nl_explanations", "adjacent", False),
    ("2311.18062", "nl_explanations", "adjacent", False), ("2504.05625", "nl_explanations", "adjacent", False),
    ("2402.04206", "nl_explanations", "adjacent", False), ("2503.17703", "nl_explanations", "adjacent", False),
    ("2510.16435", "nl_explanations", "adjacent", False), ("2304.12958", "nl_explanations", "adjacent", False),
    ("2310.19432", "classic_xai", "adjacent", False), ("2105.02658", "classic_xai", "adjacent", False),
    ("2202.08434", "classic_xai", "adjacent", False),
    ("web:Beyond performance: Explaining generalisation failures", "classic_xai", "adjacent", False),
    ("2602.07050", "world_models", "adjacent", True), ("2603.20327", "world_models", "adjacent", False),
    ("2605.24322", "world_models", "adjacent", False), ("2606.28455", "world_models", "adjacent", False),
    ("2610.18637", "world_models", "adjacent", False), ("2606.03685", "world_models", "adjacent", False),
    ("2605.20299", "world_models", "adjacent", False), ("2603.15248", "world_models", "adjacent", False),
    ("2504.11419", "world_models", "adjacent", False),
    ("web:Decision-Level Auditing of Latent Robot World Models", "world_models", "adjacent", False),
    ("2504.01871", "rl_agents", "adjacent", True), ("2406.00877", "rl_agents", "adjacent", True),
    ("2505.21552", "rl_agents", "adjacent", False), ("2310.08043", "rl_agents", "adjacent", True),
    ("2411.00867", "rl_agents", "adjacent", False), ("web:MultiSTEVE-1s", "rl_agents", "adjacent", False),
    ("2607.06328", "driving", "adjacent", True), ("2608.07361", "driving", "adjacent", True),
    ("2609.37046", "driving", "adjacent", False), ("2608.17095", "driving", "adjacent", False),
    ("2411.18714", "driving", "adjacent", False),
    ("web:Reacting to Text Is Not Following It", "driving", "adjacent", False),
]


def background_works(screened_ids: set) -> list:
    """Title-check BACKGROUND against the arXiv API; cached in data/raw/background_arxiv.json."""
    cache_path = RAW / "background_arxiv.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    need = [aid for aid, _ in BACKGROUND if aid not in cache]
    if need:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from lit_search import arxiv_by_ids
        cache.update(arxiv_by_ids(need))
        cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    out, bad = [], []
    for aid, expected in BACKGROUND:
        if aid in screened_ids:
            raise ValueError(f"{aid} is both a screened work and a background reference")
        got = cache.get(aid)
        ratio = SequenceMatcher(None, norm(expected), norm(got["title"] if got else "")).ratio()
        if ratio < 0.85:
            bad.append(f"{aid}: expected {expected!r}, arXiv has {(got or {}).get('title')!r} ({ratio:.2f})")
            continue
        out.append({"title": got["title"], "arxiv_id": aid, "url": f"https://arxiv.org/abs/{aid}",
                    "year": got["year"], "venue": "", "authors": got["authors"], "verified": True, "abstract": "",
                    "key": aid, "section": "background", "section_title": SECTIONS["background"],
                    "tier": "background", "download": False})
    if bad:
        raise RuntimeError("Background id/title mismatch:\n" + "\n".join(bad))
    return out


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", stream=sys.stdout)
    records = {r["arxiv_id"]: r for r in json.loads((PROC / "deduplicated.json").read_text(encoding="utf-8"))
               if r["arxiv_id"]}
    web = json.loads((RAW / "web_sweep.json").read_text(encoding="utf-8"))["items"]
    works, missing, seen = [], [], set()
    for ref, section, tier, download in DECISIONS:
        if ref in seen:
            raise ValueError(f"duplicate decision for {ref}")
        seen.add(ref)
        if ref.startswith("web:"):
            hits = [it for it in web if it["title"].lower().startswith(ref[4:].lower())]
            if len(hits) != 1:
                missing.append(f"{ref} ({len(hits)} web matches)")
                continue
            it = hits[0]
            rec = {"title": it["title"], "arxiv_id": it.get("arxiv_id") or "", "url": it["url"],
                   "year": it.get("year"), "venue": it.get("venue") or "", "authors": "",
                   "verified": bool(it.get("verified")), "abstract": it.get("why_relevant", "")}
        else:
            r = records.get(ref)
            if not r:
                missing.append(ref)
                continue
            rec = {"title": r["title"], "arxiv_id": ref, "url": f"https://arxiv.org/abs/{ref}", "year": r["year"],
                   "venue": r["venue"], "authors": r["authors"], "verified": True, "abstract": r["abstract"]}
        # Cite keys: the arXiv id, or "w_" + the first four title words for works without one.
        key = rec["arxiv_id"] or "w_" + "_".join(re.findall(r"[a-z0-9]+", rec["title"].lower())[:4])
        works.append(dict(rec, key=key, section=section, section_title=SECTIONS[section], tier=tier,
                          download=download and bool(rec["arxiv_id"])))
    if missing:
        raise RuntimeError(f"{len(missing)} decisions could not be resolved: {missing}")
    works += background_works(seen)
    out = ROOT / "lit_review" / "included.json"
    out.write_text(json.dumps({"sections": SECTIONS, "works": works}, indent=1, ensure_ascii=False), encoding="utf-8")
    by_tier = {t: sum(w["tier"] == t for w in works) for t in ("core", "adjacent")}
    logging.info(f"Wrote {len(works)} works to {out}: {by_tier}, {sum(w['download'] for w in works)} to download")


if __name__ == "__main__":
    main()
