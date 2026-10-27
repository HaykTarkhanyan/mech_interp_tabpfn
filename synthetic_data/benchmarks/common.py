"""Shared helpers for the benchmark runners: paths, logging, thread caps, a disk guard and result files.

Import this module first in every runner: it caps BLAS threads at 2 before numpy or torch load.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib.metadata import PackageNotFoundError, version  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE / "_src"            # upstream clones, made by setup_sources.sh (gitignored)
DATA = HERE / "_data"          # downloads and upstream working dirs (gitignored)
RESULTS = HERE / "results"
LOGS = HERE.parent / "logs"
MIN_FREE_GB = 2.0              # a run writes a few MB at most; C: had 4.9 GB free at setup, 2.8 GB later (2026-10-27)
MIN_COMMIT_HEADROOM_GB = 1.5   # one run commits ~0.5 GB; below this Windows grows the pagefile into the scarce disk
SEED = 509


def setup(name: str) -> logging.Logger:
    """Log to logs/bench_<name>.log and stdout, cap torch threads, refuse to start on a nearly full disk."""
    LOGS.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.FileHandler(LOGS / f"bench_{name}.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    try:
        import torch
        torch.set_num_threads(2)
    except ImportError:
        pass
    check_disk()
    log = logging.getLogger(name)
    log.info(f"start {name}")
    return log


def check_disk() -> None:
    free = shutil.disk_usage(HERE).free / 1e9
    if free < MIN_FREE_GB:
        raise RuntimeError(f"only {free:.1f} GB free on the disk; the runners stop below {MIN_FREE_GB} GB")
    headroom = commit_headroom_gb()
    if headroom is not None and headroom < MIN_COMMIT_HEADROOM_GB:
        raise RuntimeError(f"only {headroom:.1f} GB of memory commit left (RAM + pagefile); the runners stop below "
                           f"{MIN_COMMIT_HEADROOM_GB} GB. Close other programs first.")


def commit_headroom_gb() -> float | None:
    """Available commit (RAM + pagefile) on Windows via GlobalMemoryStatusEx; None on other systems."""
    if platform.system() != "Windows":
        return None
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MemoryStatusEx()
    st.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        raise OSError("GlobalMemoryStatusEx failed")
    return st.ullAvailPageFile / 1e9


def workdir(name: str) -> Path:
    """Upstream code often writes relative to the working directory; give each benchmark its own."""
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def pkg_versions(*names: str) -> dict:
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def save(name: str, payload: dict, started: float) -> Path:
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    payload = {"benchmark": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "runtime_s": round(time.time() - started, 1), **payload}
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logging.getLogger(name).info(f"wrote {out} after {payload['runtime_s']} s")
    return out
