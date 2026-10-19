"""Synthetic binary-classification tasks from the proposal (section 5).

    Pr(Y=1 | X) = sigmoid( h1(X1) + h2(X2) + gamma * X1 * X2 * 1{X3 > t} )

Features are iid Uniform(-2, 2): X1, X2, X3 and one pure-noise column N1. Uniform
features make equal-width bins equal-mass, so the ALE grid can be fixed by the
population quantiles instead of a sample (see ale.py).

Three task variants:
  additive     gamma = 0      control, no interaction in the data-generating logit
  interaction  gamma = 1.5    regional interaction (only where X3 > 0)
  flat         h1 = h2 = 0    stress case from review section 8 item 4: main effects are
                              exactly flat, only the pairwise/regional object carries signal

With X1, X2 centered and independent of X3, the first-order ALE of X2 equals h2 and is
untouched by the interaction term, which is what makes "main effect of X2" a clean
non-target behavior.

Column placement (review section 8 item 5): TabPFN v2 embeds features in consecutive
pairs, columns (0,1), (2,3). "same_pair" puts X1, X2 in columns 0, 1 (one token);
"diff_pair" puts them in columns 0, 2 (two tokens).

Not a script; imported by the others.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

LOW, HIGH = -2.0, 2.0
SEMANTIC_NAMES = ("X1", "X2", "X3", "N1")
# model column j holds semantic feature PLACEMENTS[name][j]
PLACEMENTS: dict[str, tuple[int, ...]] = {
    "same_pair": (0, 1, 2, 3),  # columns: X1 X2 | X3 N1
    "diff_pair": (0, 2, 1, 3),  # columns: X1 X3 | X2 N1
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    gamma: float
    h1: str  # "linear" or "zero"
    h2: str  # "quadratic" or "zero"
    threshold: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


TASKS: dict[str, TaskSpec] = {
    "additive": TaskSpec("additive", gamma=0.0, h1="linear", h2="quadratic"),
    "interaction": TaskSpec("interaction", gamma=1.5, h1="linear", h2="quadratic"),
    "flat": TaskSpec("flat", gamma=1.5, h1="zero", h2="zero"),
}


def h1_fn(spec: TaskSpec, x: np.ndarray) -> np.ndarray:
    if spec.h1 == "linear":
        return x
    if spec.h1 == "zero":
        return np.zeros_like(x)
    raise ValueError(f"unknown h1 {spec.h1!r}")


def h2_fn(spec: TaskSpec, x: np.ndarray) -> np.ndarray:
    if spec.h2 == "quadratic":
        return x**2 - 4.0 / 3.0  # centered: E[X^2] = 4/3 under U(-2, 2)
    if spec.h2 == "zero":
        return np.zeros_like(x)
    raise ValueError(f"unknown h2 {spec.h2!r}")


def true_logit_semantic(spec: TaskSpec, xs: np.ndarray) -> np.ndarray:
    """Data-generating log-odds for rows in semantic column order (X1, X2, X3, N1)."""
    x1, x2, x3 = xs[:, 0], xs[:, 1], xs[:, 2]
    inter = spec.gamma * x1 * x2 * (x3 > spec.threshold)
    return h1_fn(spec, x1) + h2_fn(spec, x2) + inter


def sample_labels(spec: TaskSpec, xs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli labels for given rows (semantic order)."""
    p = 1.0 / (1.0 + np.exp(-true_logit_semantic(spec, xs)))
    y = (rng.uniform(size=len(xs)) < p).astype(np.int64)
    if y.min() == y.max():
        raise ValueError(f"degenerate labels: all {y[0]} in {len(xs)} rows of task {spec.name}")
    return y


def sample_semantic(spec: TaskSpec, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw n rows (semantic order) and Bernoulli labels from the task."""
    xs = rng.uniform(LOW, HIGH, size=(n, len(SEMANTIC_NAMES)))
    return xs, sample_labels(spec, xs, rng)


def to_model_columns(xs: np.ndarray, placement: str) -> np.ndarray:
    return xs[:, list(PLACEMENTS[placement])]


def to_semantic_columns(xm: np.ndarray, placement: str) -> np.ndarray:
    perm = PLACEMENTS[placement]
    xs = np.empty_like(xm)
    xs[:, list(perm)] = xm
    return xs


def model_column_of(semantic_name: str, placement: str) -> int:
    """Which model column holds a semantic feature under a placement."""
    s = SEMANTIC_NAMES.index(semantic_name)
    return PLACEMENTS[placement].index(s)
