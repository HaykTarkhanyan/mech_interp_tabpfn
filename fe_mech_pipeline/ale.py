"""Accumulated local effects (Apley and Zhu 2020), first and second order, in numpy.

Workflow, so that one TabPFN forward pass serves every ALE at once:
  1. ALEPlan(X_ref, specs) builds all query rows the estimators need (each reference row
     moved to the edges of its bin or cell).
  2. The caller predicts those rows with any model, on the chosen prediction scale.
  3. plan.compute(preds) returns the ALE objects; plan.compute(preds, rows=idx) does
     the same for a bootstrap resample of reference rows, at no extra model cost.

Estimators:
  first order   f_j(z_k) = sum_{m<=k} mean_{i in bin m} [g(z_m, x_i\\j) - g(z_{m-1}, x_i\\j)],
                centred so that its bin-count-weighted mean is 0 (Apley and Zhu, eq. 15-16).
  second order  the pure (j, l) interaction: accumulated mean second differences per cell,
                minus their own first-order effects, minus the weighted mean. This follows
                ALEPlot's ALE.2D code (Apley's R reference implementation): lower-order
                effects are removed using differences averaged over the two cell edges.

Empty bins or cells raise ValueError instead of being filled in silently.

Distance (the d in Delta_B): integrated L2 on the fixed grid with the population mass of
each bin or cell as weight, after centring both objects, divided by the norm of the
reference object. With iid uniform features and equal-width bins every bin (cell) has
equal mass, so the weights are uniform. Values on the grid are bin midpoints (cell
corner averages).

Not a script; imported by the others. Validated by validate_ale.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ALESpec:
    name: str
    cols: tuple[int, ...]  # one column (first order) or two (second order)
    edges: tuple[np.ndarray, ...]  # bin edges per column, length K + 1, increasing


@dataclass
class ALEResult:
    name: str
    edges: list[np.ndarray]
    values: np.ndarray  # (K+1,) or (K1+1, K2+1), at bin edges / cell corners
    counts: np.ndarray  # (K,) or (K1, K2)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "edges": [e.tolist() for e in self.edges],
            "values": self.values.tolist(),
            "counts": self.counts.tolist(),
        }


def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin k in 1..K with edges[k-1] < x <= edges[k]; x == edges[0] goes to bin 1."""
    if np.any(x < edges[0]) or np.any(x > edges[-1]):
        raise ValueError(f"values outside the ALE grid [{edges[0]}, {edges[-1]}]")
    return np.clip(np.searchsorted(edges, x, side="left"), 1, len(edges) - 1)


@dataclass
class ALEPlan:
    x_ref: np.ndarray
    specs: list[ALESpec]
    queries: np.ndarray = field(init=False)
    _blocks: list[tuple[ALESpec, int, tuple[np.ndarray, ...]]] = field(init=False)

    def __post_init__(self) -> None:
        n = self.x_ref.shape[0]
        parts = [self.x_ref.copy()]  # block 0: the unmodified reference rows
        offset = n
        self._blocks = []
        for spec in self.specs:
            bins = tuple(bin_index(self.x_ref[:, c], e) for c, e in zip(spec.cols, spec.edges))
            if len(spec.cols) == 1:
                (c,), (e,), (k,) = spec.cols, spec.edges, bins
                for kk in (k - 1, k):  # lower edge, upper edge
                    q = self.x_ref.copy()
                    q[:, c] = e[kk]
                    parts.append(q)
            elif len(spec.cols) == 2:
                (c1, c2), (e1, e2), (k, l) = spec.cols, spec.edges, bins
                for kk, ll in ((k - 1, l - 1), (k, l - 1), (k - 1, l), (k, l)):
                    q = self.x_ref.copy()
                    q[:, c1] = e1[kk]
                    q[:, c2] = e2[ll]
                    parts.append(q)
            else:
                raise ValueError(f"ALE order {len(spec.cols)} not supported")
            self._blocks.append((spec, offset, bins))
            offset += n * (2 if len(spec.cols) == 1 else 4)
        self.queries = np.vstack(parts)

    @property
    def n_ref(self) -> int:
        return self.x_ref.shape[0]

    def reference_preds(self, preds: np.ndarray) -> np.ndarray:
        """Predictions on the unmodified reference rows (block 0)."""
        return preds[: self.n_ref]

    def compute(self, preds: np.ndarray, rows: np.ndarray | None = None) -> dict[str, ALEResult]:
        if preds.shape != (self.queries.shape[0],):
            raise ValueError(f"preds shape {preds.shape} != ({self.queries.shape[0]},)")
        n = self.n_ref
        rows = np.arange(n) if rows is None else rows
        out: dict[str, ALEResult] = {}
        for spec, off, bins in self._blocks:
            if len(spec.cols) == 1:
                lo, hi = preds[off : off + n], preds[off + n : off + 2 * n]
                out[spec.name] = first_order(hi[rows] - lo[rows], bins[0][rows], spec)
            else:
                b = [preds[off + i * n : off + (i + 1) * n] for i in range(4)]
                d2 = (b[3] - b[2] - b[1] + b[0])[rows]
                out[spec.name] = second_order(d2, bins[0][rows], bins[1][rows], spec)
        return out


def first_order(diffs: np.ndarray, k: np.ndarray, spec: ALESpec) -> ALEResult:
    K = len(spec.edges[0]) - 1
    counts = np.bincount(k, minlength=K + 1)[1:]
    if np.any(counts == 0):
        raise ValueError(f"{spec.name}: empty ALE bins {np.flatnonzero(counts == 0) + 1}")
    local = np.bincount(k, weights=diffs, minlength=K + 1)[1:] / counts
    acc = np.concatenate([[0.0], np.cumsum(local)])
    acc -= np.sum(counts * (acc[:-1] + acc[1:]) / 2.0) / counts.sum()
    return ALEResult(spec.name, [spec.edges[0]], acc, counts)


def second_order(d2: np.ndarray, k: np.ndarray, l: np.ndarray, spec: ALESpec) -> ALEResult:
    K1, K2 = len(spec.edges[0]) - 1, len(spec.edges[1]) - 1
    cell = (k - 1) * K2 + (l - 1)
    counts = np.bincount(cell, minlength=K1 * K2).reshape(K1, K2)
    if np.any(counts == 0):
        raise ValueError(f"{spec.name}: {int((counts == 0).sum())} empty ALE cells")
    delta = np.bincount(cell, weights=d2, minlength=K1 * K2).reshape(K1, K2) / counts
    f = np.zeros((K1 + 1, K2 + 1))
    f[1:, 1:] = np.cumsum(np.cumsum(delta, axis=0), axis=1)
    # remove the first-order effects of f itself (ALEPlot ALE.2D)
    d1 = f[1:, :] - f[:-1, :]  # (K1, K2+1) differences along x1
    d1 = (d1[:, :-1] + d1[:, 1:]) / 2.0  # averaged over the two x2 edges of each cell
    f1 = np.concatenate([[0.0], np.cumsum(np.sum(d1 * counts, axis=1) / counts.sum(axis=1))])
    d2b = f[:, 1:] - f[:, :-1]  # (K1+1, K2) differences along x2
    d2b = (d2b[:-1, :] + d2b[1:, :]) / 2.0
    f2 = np.concatenate([[0.0], np.cumsum(np.sum(d2b * counts, axis=0) / counts.sum(axis=0))])
    f = f - f1[:, None] - f2[None, :]
    f -= np.sum(counts * cell_average(f)) / counts.sum()
    return ALEResult(spec.name, [spec.edges[0], spec.edges[1]], f, counts)


def cell_average(v: np.ndarray) -> np.ndarray:
    """Midpoint (1D) or corner average (2D) of values given at edges / corners."""
    if v.ndim == 1:
        return (v[:-1] + v[1:]) / 2.0
    return (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:]) / 4.0


def _centred_cells(v: np.ndarray) -> np.ndarray:
    c = cell_average(v)
    return c - c.mean()  # uniform population weights (equal-mass bins / cells)


def l2_norm(v: np.ndarray) -> float:
    """Integrated L2 norm of a centred ALE object on its grid (uniform cell weights)."""
    return float(np.sqrt(np.mean(_centred_cells(v) ** 2)))


def relative_l2(reference: np.ndarray, other: np.ndarray) -> float:
    """||other - reference|| / ||reference||, both on the same fixed grid."""
    if reference.shape != other.shape:
        raise ValueError(f"grid mismatch {reference.shape} vs {other.shape}")
    norm = l2_norm(reference)
    if norm <= 0.0:
        raise ValueError("reference ALE has zero norm; relative distance undefined")
    diff = _centred_cells(other) - _centred_cells(reference)
    return float(np.sqrt(np.mean(diff**2)) / norm)


def interp_to_grid(res: ALEResult, target_edges: list[np.ndarray]) -> np.ndarray:
    """Linear (1D) or bilinear (2D) interpolation of an ALE onto another grid."""
    for src, tgt in zip(res.edges, target_edges):
        if tgt[0] < src[0] - 1e-12 or tgt[-1] > src[-1] + 1e-12:
            raise ValueError("target grid extends beyond the source grid")
    if res.values.ndim == 1:
        return np.interp(target_edges[0], res.edges[0], res.values)
    along1 = np.stack(
        [np.interp(target_edges[0], res.edges[0], res.values[:, j]) for j in range(res.values.shape[1])],
        axis=1,
    )
    return np.stack(
        [np.interp(target_edges[1], res.edges[1], along1[i, :]) for i in range(along1.shape[0])],
        axis=0,
    )


def uniform_edges(low: float, high: float, k: int) -> np.ndarray:
    return np.linspace(low, high, k + 1)


def jitter_edges(edges: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    """Shift each interior edge by U(-frac, frac) bin widths; the outer edges stay fixed."""
    if not 0.0 <= frac < 0.5:
        raise ValueError("jitter fraction must be in [0, 0.5) to keep edges ordered")
    width = np.diff(edges).min()
    out = edges.copy()
    out[1:-1] += rng.uniform(-frac, frac, size=len(edges) - 2) * width
    return out
