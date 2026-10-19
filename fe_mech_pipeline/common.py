"""Shared setup for every script in fe_mech_pipeline: paths, logging, threads, seed.

Not a script; imported by the others.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 509
N_THREADS = 4  # the laptop freezes under sustained all-core load; 4 of 8 cores

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"


def setup(script_name: str) -> logging.Logger:
    """Configure torch threads and a logger writing to console and logs/<script>.log."""
    torch.set_num_threads(N_THREADS)
    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    log = logging.getLogger(script_name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file = logging.FileHandler(LOG_DIR / f"{script_name}.log", mode="w", encoding="utf-8")
    file.setFormatter(fmt)
    log.addHandler(stream)
    log.addHandler(file)
    log.propagate = False
    log.info(f"torch {torch.__version__}, threads={torch.get_num_threads()}, seed={SEED}")
    return log


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=_jsonable), encoding="utf-8")
