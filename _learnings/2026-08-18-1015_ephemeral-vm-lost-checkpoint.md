# A finished checkpoint died with its Colab VM - always push artifacts off the box immediately

> Copied 2026-10-19 from the ipr-ate-3 repo (Armenian text embeddings project), where it was recorded. Project-specific names (ate3 repos, datasets) refer to that project; the lessons apply to any RunPod or Colab GPU job here.

**Symptom** `repro_hy` trained to completion on a Colab T4 (50/50 steps, 63 min, loss 5.0 -> 1.58, "REPRO_HY DONE - final at /content/runs/repro_hy/final"). Minutes later the next `colab exec` returned:

```
[colab] Session 'ate3' appears to be lost (404/401). Cleaning up.
```

The VM was gone and took `/content/runs/repro_hy/final` with it. 63 minutes of T4 training lost, and Colab's free-tier GPU quota was exhausted by then, so it could not simply be re-run.

**Cause** Two compounding mistakes, both mine:
1. The artifact was left on ephemeral storage. Colab VMs can vanish at any moment (idle reclaim, quota, backend rotation) - `/content` is not storage, it is scratch.
2. I chained the *next* training run instead of first securing the finished one. The download step was planned for "after both runs", which is one failure away from losing everything.

**Consequences / fix**
- `scripts/train_ate.py` now takes `--push-to-hub <repo_id>`: it calls `create_repo(private=True, exist_ok=True)` + `upload_folder` immediately after `save_pretrained`, inside the same process. Use it on EVERY remote run.
- Rule: on any ephemeral box (Colab, RunPod, spot GCE), an artifact is not real until it is off the machine. Secure the artifact before starting the next job, never after.
- Colab free-tier GPU quota is real and it bit us the same day: ~2h of T4 (35 min benchmark + 63 min training + smoke tests) exhausted it. `colab new --gpu T4` then fails with `Precondition Failed` on the assign endpoint, while a plain CPU `colab new` still succeeds - that contrast is the quick diagnostic for "GPU quota exhausted" vs "account/auth broken".
