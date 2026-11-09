"""Figure helpers (blueprint §9): Phase 2 style and palette (decoding.figures)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from start_vision.decoding.figures import BAND, GRID, INK, MUTED, SLOTS, SURFACE, _style

__all__ = ["BAND", "GRID", "INK", "MUTED", "SLOTS", "axes", "save", "style"]


def axes(n: int = 1, w: float = 4.2, h: float = 3.2) -> tuple[Figure, list[Axes]]:
    fig, ax = plt.subplots(1, n, figsize=(w * n, h), squeeze=False)
    return fig, list(ax[0])


def style(ax: Axes, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    _style(ax)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    if title:
        ax.set_title(title, fontsize=8, color=INK)


def save(fig: Figure, d: Path) -> Path:
    fig.tight_layout()
    p = d / "figure.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p
