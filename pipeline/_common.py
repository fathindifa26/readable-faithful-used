"""Shared path helper for the analysis scripts under pipeline/<stage>/.

Every script in this repo resolves paths through here instead of a
hardcoded absolute path, so `python pipeline/08_.../probe_v2.py` works
the same on your machine as it did on ours.

Convention:
  - Small, final CSV/JSON/PNG outputs (the ones this repo ships) live in
    `results/<stage>/`. Scripts read their cross-stage inputs from there
    and write their own outputs back into their own stage's folder.
  - Large intermediate arrays (`*.npz` / `*.npy` -- raw per-head
    activations, on the order of hundreds of MB to a few GB per stage)
    are produced by the Kaggle notebooks in `pipeline/<stage>/` but are
    NOT included in this repo. To re-run a script that needs them,
    re-run the notebook yourself and drop its `.npz`/`.npy` output into
    `results/<stage>/` next to the shipped CSVs -- the scripts look for
    them there by default.
  - The demographic ground-truth table (`opinionqa_intersectional.csv`)
    is derived from restricted Pew ATP microdata and is likewise not
    shipped; see `data/README.md` to regenerate it into `data/`.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def results_dir(stage: str) -> Path:
    """Path to results/<stage>/ (created if it doesn't exist yet)."""
    d = REPO_ROOT / "results" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_path(relative: str) -> Path:
    """Path to a file under data/ (see data/README.md to populate it)."""
    return REPO_ROOT / "data" / relative
