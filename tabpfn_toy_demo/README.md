# TabPFN toy demo

A minimal TabPFN classification example using scikit-learn's built-in Iris dataset.

## Run

Requires Python 3.10 or newer. From this directory:

```powershell
uv run run_demo.py
```

`uv` creates and manages the local `.venv` automatically. To install the
environment without running the script, use `uv sync`.

## One-time TabPFN access setup

The first local run downloads TabPFN's pretrained checkpoint. Current TabPFN
releases require you to sign in and accept the local-inference license first:

1. Sign in or register at <https://ux.priorlabs.ai/account>.
2. Accept the license at <https://ux.priorlabs.ai/account/licenses>.
3. Add your access token to a local `.env` file as `TABPFN_TOKEN=...`.
4. Run `uv run run_demo.py` again.

The `.env` file is already ignored by Git. Keep it local and do not commit or
share the token.

The script deliberately uses `device="cpu"` and a 150-row dataset, so it works on
a CPU-only machine after that one-time setup.

For a fuller-quality prediction ensemble, remove `n_estimators=1` and use TabPFN's
default setting.
