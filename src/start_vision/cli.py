"""Command-line entry points (sv-inventory, sv-stage, sv-configs, sv-analyze).

Each command is filled in by its build step in plans/PHASE1_BLUEPRINT.md §8.
"""

from __future__ import annotations

import sys


def _not_built(name: str, step: int) -> None:
    print(f"{name}: not built yet (PHASE1_BLUEPRINT §8 step {step})", file=sys.stderr)
    raise SystemExit(2)


def inventory() -> None:
    _not_built("sv-inventory", 3)


def stage() -> None:
    _not_built("sv-stage", 4)


def configs() -> None:
    _not_built("sv-configs", 5)


def analyze() -> None:
    _not_built("sv-analyze", 9)
