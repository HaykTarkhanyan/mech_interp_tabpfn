"""TabPFN v2 as g_D, with hooks that ablate attention heads and MLP units.

Model: the original released TabPFN v2 classifier checkpoint `tabpfn-v2-classifier.ckpt`
(Hugging Face Prior-Labs/TabPFN-v2-clf; 12 blocks, d_model 192, 6 heads of 32 dims per
attention, MLP hidden 768, features embedded in pairs). This is the file Balef et al.'s
released code loads (it vendors tabpfn 2.0.9, whose v2 default is this file). It is
named explicitly because tabpfn 8.2.0's `create_default_for_version(V2)` would load
`tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` instead.

Prediction: g_D(x) = log-odds of class 1 = logit_1 - logit_0 from the raw model output,
temperature 1, one forward pass with the context D and all query rows. Query rows do not
attend to each other in TabPFN v2 (item attention: queries attend to context rows only;
feature attention and MLP are per row), so batching every ALE query into one pass is
exact. check_model.py verifies that this raw path equals the sklearn wrapper's output.

Hook locations (all hooks registered once; inactive hooks return None and change nothing):
  feature attention, layer L, head h:
      forward pre-hook on blocks[L].per_sample_attention_between_features.out_projection;
      its input is the concatenated per-head attention output, shape (rows, tokens, 6*32);
      head h is the slice [h*32:(h+1)*32], i.e. the head's output before the output
      projection mixes heads. out_projection has no bias, so zeroing the slice removes
      exactly that head's contribution to the residual stream.
  item attention, layer L, head h:
      the same on blocks[L].per_column_attention_between_cells.out_projection; input
      shape (tokens, rows, 6*32). Note: in TabPFN v2 query rows use multi-query attention
      (all 6 query heads read key/value head 0), so for query rows "head h" is query h
      reading the shared key/value head; for context rows it is a normal head.
  MLP, layer L, hidden unit u:
      forward hook on blocks[L].mlp[1] (the GELU), output shape (rows*tokens, 768). The
      second MLP linear has no bias, so mean-ablating all 768 units equals replacing the
      MLP output by its mean, and zeroing them removes the MLP. A whole-layer MLP is
      therefore "768 units", which lets the null sample 768 random units across layers.

Ablation types:
  mean  replace the unit's activation by its mean over a reference forward pass
        (context D plus the unmodified reference query rows), computed separately for
        each token column and for context vs query rows (the target token holds the label
        in context rows and no label in query rows, so pooling them would inject or
        remove label information).
  zero  replace by 0 (sanity contrast).
  patch replace by activations captured from another pass (used to test the hooks).

Not a script; imported by the others.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import prepend_cache_path
from tabpfn.preprocessing import PreprocessorConfig

from common import SEED

CKPT_NAME = "tabpfn-v2-classifier.ckpt"
HF_REPO = "Prior-Labs/TabPFN-v2-clf"
N_LAYERS, N_HEADS, HEAD_DIM, D_MLP = 12, 6, 32, 768
SITES = ("feat", "item", "mlp")
SITE_LABEL = {"feat": "feature attention", "item": "item attention", "mlp": "MLP"}
UNITS_PER_LAYER = {"feat": N_HEADS, "item": N_HEADS, "mlp": D_MLP}
PERF = PerformanceOptions(save_peak_memory_factor=None, force_recompute_layer=False, use_chunkwise_inference=False)

# Units to ablate: (site, layer) -> unit indices (heads 0..5, or MLP hidden units 0..767)
Units = dict[tuple[str, int], np.ndarray]


def pinned_inference_config() -> dict:
    """Wrapper preprocessing switched off so that model column j == input column j."""
    return {
        # identity transform; the transformer's own z-scoring still runs inside the model
        "PREPROCESS_TRANSFORMS": [
            PreprocessorConfig("none", categorical_name="numeric", append_original=False, global_transformer_name=None)
        ],
        "FEATURE_SHIFT_METHOD": None,  # no column shuffling (would move features between pairs)
        "CLASS_SHIFT_METHOD": None,  # class 0 stays output 0
        "FINGERPRINT_FEATURE": False,  # would append a hash column and change the pairing
        "POLYNOMIAL_FEATURES": "no",  # would append x_i*x_j columns, i.e. hand the model the interaction
        "OUTLIER_REMOVAL_STD": None,
        "SUBSAMPLE_SAMPLES": None,
    }


def make_classifier(device: str = "cpu") -> TabPFNClassifier:
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        model_path=prepend_cache_path(CKPT_NAME),
        n_estimators=1,
        auto_scale_n_estimators=False,
        softmax_temperature=1.0,
        device=device,
        inference_precision=torch.float32,
        random_state=SEED,
        inference_config=pinned_inference_config(),
        n_preprocessing_jobs=1,
    )


def checkpoint_path() -> Path:
    return Path(prepend_cache_path(CKPT_NAME))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v2(x_ctx: np.ndarray, y_ctx: np.ndarray, device: str = "cpu") -> tuple[TabPFNClassifier, torch.nn.Module]:
    """Fit the wrapper once (this loads, and on first use downloads, the checkpoint)."""
    clf = make_classifier(device)
    clf.fit(x_ctx, y_ctx)
    model = clf.model_
    model.eval()
    cfg = model.blocks[0]
    if len(model.blocks) != N_LAYERS or cfg.per_sample_attention_between_features.num_heads != N_HEADS:
        raise RuntimeError("loaded model does not have the TabPFN v2 shape (12 blocks, 6 heads)")
    if model.features_per_group != 2:
        raise RuntimeError(f"features_per_group = {model.features_per_group}, expected 2")
    return clf, model


def unit_slice(site: str, idx: np.ndarray) -> torch.Tensor:
    """Indices into the hooked tensor's last dim for the given heads or MLP units."""
    if site == "mlp":
        return torch.as_tensor(np.asarray(idx), dtype=torch.long)
    return torch.as_tensor(np.concatenate([np.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in idx]), dtype=torch.long)


class HookedTabPFN:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.n_ctx = 0
        self.n_tokens = 0
        self._mode: str | None = None  # None | capture_means | capture_full | ablate
        self._kind = "mean"
        self._units: Units = {}
        self._source: dict = {}
        self._store: dict = {}
        self._handles = []
        for layer, block in enumerate(model.blocks):
            fa = block.per_sample_attention_between_features.out_projection
            ia = block.per_column_attention_between_cells.out_projection
            self._handles.append(fa.register_forward_pre_hook(self._pre_hook("feat", layer)))
            self._handles.append(ia.register_forward_pre_hook(self._pre_hook("item", layer)))
            self._handles.append(block.mlp[1].register_forward_hook(self._post_hook("mlp", layer)))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # canonical view of every hooked tensor: (rows, tokens, dim)
    def _view(self, site: str, x: torch.Tensor) -> torch.Tensor:
        if site == "feat":
            return x
        if site == "item":
            return x.transpose(0, 1)
        return x.view(-1, self.n_tokens, x.shape[-1])

    def _process(self, site: str, layer: int, x: torch.Tensor) -> torch.Tensor | None:
        if self._mode is None:
            return None
        key = (site, layer)
        if self._mode == "capture_means":
            v = self._view(site, x)
            self._store[key] = (v[: self.n_ctx].mean(0).clone(), v[self.n_ctx :].mean(0).clone())
            return None
        if self._mode == "capture_full":
            self._store[key] = self._view(site, x).clone()
            return None
        if key not in self._units:
            return None
        out = x.clone()
        v = self._view(site, out)
        if v.shape[0] <= self.n_ctx:
            raise RuntimeError(f"hook at {key}: {v.shape[0]} rows but n_ctx = {self.n_ctx}")
        idx = unit_slice(site, self._units[key]).to(v.device)
        n = self.n_ctx
        if self._kind == "zero":
            v[:, :, idx] = 0.0
        elif self._kind == "mean":
            m_ctx, m_q = self._source[key]
            v[:n, :, idx] = m_ctx[None, :, idx]
            v[n:, :, idx] = m_q[None, :, idx]
        elif self._kind == "patch":
            v[:, :, idx] = self._source[key][:, :, idx]
        else:
            raise ValueError(f"unknown ablation kind {self._kind!r}")
        return out

    def _pre_hook(self, site: str, layer: int):
        def hook(_module, args):
            new = self._process(site, layer, args[0])
            return None if new is None else (new,)

        return hook

    def _post_hook(self, site: str, layer: int):
        def hook(_module, _args, output):
            return self._process(site, layer, output)

        return hook

    def _forward(self, x_ctx: np.ndarray, y_ctx: np.ndarray, x_q: np.ndarray) -> np.ndarray:
        self.n_ctx = x_ctx.shape[0]
        self.n_tokens = math.ceil(x_ctx.shape[1] / self.model.features_per_group) + 1
        dev = next(self.model.parameters()).device
        x = torch.from_numpy(np.vstack([x_ctx, x_q]).astype(np.float32))[:, None, :].to(dev)
        y = torch.from_numpy(y_ctx.astype(np.float32)).to(dev)
        with torch.inference_mode():
            out = self.model(x, y, only_return_standard_out=True, performance_options=PERF)
        logits = out[:, 0, :2].double().cpu().numpy()
        if logits.shape[0] != x_q.shape[0]:
            raise RuntimeError(f"got {logits.shape[0]} outputs for {x_q.shape[0]} query rows")
        return logits

    @contextmanager
    def _mode_ctx(self, mode: str | None, **kw) -> Iterator[None]:
        self._mode = mode
        self._store = {}
        for k, v in kw.items():
            setattr(self, f"_{k}", v)
        try:
            yield
        finally:
            self._mode, self._units, self._source = None, {}, {}

    def logits(self, x_ctx, y_ctx, x_q) -> np.ndarray:
        """Raw 2-class logits, hooks inactive."""
        with self._mode_ctx(None):
            return self._forward(x_ctx, y_ctx, x_q)

    def logodds(self, x_ctx, y_ctx, x_q, units: Units | None = None, kind: str = "mean", source: dict | None = None) -> np.ndarray:
        """g_D(x_q) on the log-odds scale, optionally with `units` ablated."""
        if not units:
            with self._mode_ctx(None):
                lg = self._forward(x_ctx, y_ctx, x_q)
        else:
            if kind in ("mean", "patch") and source is None:
                raise ValueError(f"{kind} ablation needs a source (means or captured activations)")
            with self._mode_ctx("ablate", kind=kind, units=units, source=source or {}):
                lg = self._forward(x_ctx, y_ctx, x_q)
        return lg[:, 1] - lg[:, 0]

    def capture_means(self, x_ctx, y_ctx, x_ref) -> dict:
        """Per (site, layer): (context-row mean, query-row mean), each (tokens, dim)."""
        with self._mode_ctx("capture_means"):
            self._forward(x_ctx, y_ctx, x_ref)
            store = self._store
        if len(store) != 3 * N_LAYERS:
            raise RuntimeError(f"captured {len(store)} sites, expected {3 * N_LAYERS}")
        return store

    def capture_full(self, x_ctx, y_ctx, x_q) -> dict:
        with self._mode_ctx("capture_full"):
            self._forward(x_ctx, y_ctx, x_q)
            return self._store


def layer_units(site: str, layer: int) -> Units:
    return {(site, layer): np.arange(UNITS_PER_LAYER[site])}


def random_units(site: str, size: int, rng: np.random.Generator) -> Units:
    """`size` units of one site drawn uniformly without replacement across all layers."""
    per = UNITS_PER_LAYER[site]
    flat = rng.choice(N_LAYERS * per, size=size, replace=False)
    out: Units = {}
    for layer in range(N_LAYERS):
        sel = np.sort(flat[flat // per == layer] % per)
        if sel.size:
            out[(site, layer)] = sel
    return out


def describe_units(units: Units) -> dict:
    return {f"{s}:{l}": len(v) for (s, l), v in sorted(units.items())}
