# RunPod: three separate traps between "API key works" and "GPU is training"

> Copied 2026-10-19 from the ipr-ate-3 repo (Armenian text embeddings project), where it was recorded. Project-specific names (ate3 repos, datasets) refer to that project; the lessons apply to any RunPod or Colab GPU job here.

Getting the first RunPod training job running took four failed deploys. Each failure mode is
invisible from the error message you get, so they are recorded here in the order we hit them.

## 1. "This machine does not have the resources to deploy your pod" can mean "wrong GPU model"

Symptom: every `POST /v2/pods` returned 400 with that message - 8 attempts across A40 and RTX
4090, SECURE and COMMUNITY, via REST v2, GraphQL `podFindAndDeployOnDemand`, and RunPod's own
MCP tool - while `list-gpu-types`/`get-capacity` reported **HIGH** stock for both.

Not capacity, not the payload, not the key. Proof: deploying an A40 from the **web console**
succeeded instantly (pod `5bky2z0g1ftre6`, CA-MTL-1), and re-issuing that pod's *exact* config
through the API (same image, datacenter, disk 30, 50GB persistent mount) failed again seconds
later. A config sweep isolated it:

| variant | result |
|---|---|
| A40 + exact UI config (mounts + dataCenterIds) | 400 no resources |
| A40, no mounts | 400 no resources |
| A40, no datacenter, no mounts | 400 no resources |
| A40 via `templateId: runpod-torch-v280` | 400 no resources |
| **RTX 4090, `cloud: SECURE`, `disk: 30`, no mounts, no datacenter** | **SUCCESS** |

So: the API refused A40 under every configuration while accepting 4090 immediately. When you
hit this, **sweep GPU models before believing the error text** - and do not waste time on the
account (our key was Read & Write and enabled, balance $20, ToS signed, and `DELETE /v2/pods`
worked fine, which proves write access).

## 2. `args` is not passed to a shell verbatim - nested quotes are stripped

`args: "bash -c 'python -c \"...\"'"` arrived in the container with the inner quotes **gone**:

```
from huggingface_hub import snapshot_download; snapshot_download(HaykTarkhanyan/ate3-code, local_dir=/root/code)
SyntaxError: invalid syntax
```

Write bootstraps with **zero nested quotes**. Use CLIs that take bare arguments
(`hf download REPO --local-dir /root/code`) instead of `python -c "..."`.

## 3. A crash-looping container bills at full rate and looks "RUNNING"

Both failed bootstraps left the pod `status: RUNNING` while the container restarted every ~13s.
`get-pod` alone will not tell you; `stream-pod-logs --source container` will. First failure was
`git clone` on our **private** GitHub repo:

```
fatal: could not read Username for 'https://github.com': No such device or address
```

Fix used: ship code via a private HF model repo (`HaykTarkhanyan/ate3-code`, uploaded with
`upload_folder`) and pull it on the pod with `hf download` + `HF_TOKEN` - no GitHub credential
needed on the box, and the same token already handles `--push-to-hub` for the artifacts.

**Always verify a new pod with `stream-pod-logs --source container` within a minute of
creating it, and delete on sight of a loop.** Four bad pods cost about $0.10 total only
because they were caught fast.

## Working recipe (verified 2026-08-18)

```json
{"name":"ate3-training","image":"runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
 "cloud":"SECURE","gpu":{"id":"NVIDIA GeForce RTX 4090","count":1},"disk":30,
 "args":"bash -c 'set -e; ... no nested quotes ...'",
 "env":{"WANDB_API_KEY":"...","HF_TOKEN":"...","HF_HUB_DISABLE_PROGRESS_BARS":"1"}}
```

Note the 4090 reports `bf16` support, so `train_ate.py --precision auto` selects bf16 there,
versus fp16 on Colab's T4 - a real (if small) numerical difference between the two lanes.
