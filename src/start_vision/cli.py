"""Command-line entry points (sv-inventory, sv-stage, sv-configs, sv-analyze).

Each command is filled in by its build step in plans/PHASE1_BLUEPRINT.md §8.
"""

from __future__ import annotations

import sys


def _not_built(name: str, step: int) -> None:
    print(f"{name}: not built yet (PHASE1_BLUEPRINT §8 step {step})", file=sys.stderr)
    raise SystemExit(2)


def inventory() -> None:
    """Write results/phase1/inventory.tsv from the S3 listing of ds006171."""
    from pathlib import Path

    from start_vision.inventory import list_remote

    df = list_remote()
    out = Path("results/phase1/inventory.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"{len(df)} subjects, {df['bytes'].sum() / 1e9:.1f} GB -> {out}")
    print(df["n_tasks"].value_counts().sort_index().to_string())


def stage() -> None:
    _not_built("sv-stage", 4)


def configs() -> None:
    _not_built("sv-configs", 5)


def analyze() -> None:
    _not_built("sv-analyze", 9)
