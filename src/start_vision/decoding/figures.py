"""Phase 2 figures F1-F10 and tables T1-T3 (PHASE2 §10; build step 12).

Every function takes arrays/tables (never files) and writes ``<out>/<name>.png`` and ``.svg``.

Colours: dataviz reference palette, categorical slots 1-4 in fixed order (blue, orange, aqua,
yellow), validated with the skill's validator on the light surface: all checks pass; aqua and
yellow are below 3:1 contrast, so every series is named in a legend (the user asked 2026-11-13 not
to repeat legend names on the lines). Series colour follows the entity (a task keeps its colour in every
figure). Matrices use a diverging map centred on chance (AUC 0.5). Text is ink, never series
colour; grids and axes are recessive.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats as sps

from start_vision.decoding import stats

SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#dcdbd6", "#efeeea"
SURFACE = "#ffffff"  # pure white: matches the page in the paper
ENTITY = {  # fixed colour per entity across figures
    "pooled": INK, "nocue": SLOTS[0], "noninformative": SLOTS[1], "informative": SLOTS[2],
    "seen": SLOTS[0], "unseen": SLOTS[1], "catch FA": SLOTS[2], "catch CR": SLOTS[3],
}


def _color(name: str, i: int) -> str:
    return ENTITY.get(name, SLOTS[i % len(SLOTS)])


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _cbar(cb: Any, label: str = "AUC") -> None:
    cb.set_label(label, fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_edgecolor(GRID)


def _save(fig: Figure, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, facecolor=SURFACE)
    fig.savefig(out / f"{name}.svg", facecolor=SURFACE)
    plt.close(fig)
    return out / f"{name}.png"


def _ci(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = a.mean(0)
    if len(a) < 2:
        return m, np.zeros_like(m)
    return m, sps.t.ppf(0.975, len(a) - 1) * a.std(0, ddof=1) / np.sqrt(len(a))


def _dodge(ys: Sequence[float], gap: float) -> list[float]:
    """Spread label y positions so neighbours are at least ``gap`` apart (order kept)."""
    order = np.argsort(ys)
    out = np.asarray(ys, float).copy()
    for a, b in itertools.pairwise(order):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return list(out)


def series_panel(ax: Axes, times: np.ndarray, curves: Mapping[str, np.ndarray],
                 masks: Mapping[str, np.ndarray] | None = None, ylabel: str = "AUC",
                 chance: float = 0.5) -> None:
    """Group mean ± 95 % CI per series (rows = subjects), significant samples as a bar under
    the curves, chance line, legend (no direct labels)."""
    _style(ax)
    t = times * 1000
    ax.axhline(chance, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    lo_all: list[float] = []
    for i, (name, a) in enumerate(curves.items()):
        a = np.atleast_2d(a)
        m, h = _ci(a)
        c = _color(name, i)
        ax.fill_between(t, m - h, m + h, color=c, alpha=0.15, lw=0)
        ax.plot(t, m, color=c, lw=2, label=f"{name[:1].upper() + name[1:]} (N = {len(a)})")
        lo_all.append((m - h).min())
    if masks:
        y0 = min(min(lo_all), chance) - 0.01
        for i, (name, mk) in enumerate(masks.items()):
            yy = y0 - 0.006 * i
            ax.plot(np.where(mk, t, np.nan), np.full_like(t, yy), color=_color(name, i),
                    lw=3, solid_capstyle="butt")
    ax.set_xlabel("Time from onset (ms)", fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="upper left")


def f1_sliding(times: np.ndarray, auc: Mapping[str, np.ndarray],
               masks: Mapping[str, np.ndarray], out: Path) -> Path:
    """F1: within-subject AUC(t) per scope (pooled + tasks), clusters as bars."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    series_panel(ax, times, auc, masks)
    ax.set_title("F1  Within-subject decoding of seen vs unseen (bars: significant clusters)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return _save(fig, out, "F1_sliding")


def f2_generalization(times: np.ndarray, auc: np.ndarray, mask: np.ndarray | None,
                      out: Path) -> Path:
    """F2: group-mean temporal generalization (train rows x test columns), cluster contour."""
    m = auc.mean(0) if auc.ndim == 3 else auc
    lim = max(0.02, float(np.abs(m - 0.5).max()))
    t = times * 1000
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
    im = ax.imshow(m, origin="lower", cmap="RdBu_r", vmin=0.5 - lim, vmax=0.5 + lim,
                   extent=(t[0], t[-1], t[0], t[-1]), aspect="auto")
    if mask is not None and mask.any():
        ax.contour(t, t, mask.astype(float), levels=[0.5], colors=INK, linewidths=1)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Test time (ms)", fontsize=8, color=MUTED)
    ax.set_ylabel("Train time (ms)", fontsize=8, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    _cbar(fig.colorbar(im, ax=ax, shrink=0.8))
    ax.set_title("F2  Temporal generalization (contour: cluster)", fontsize=9, color=INK,
                 loc="left")
    fig.tight_layout()
    return _save(fig, out, "F2_generalization")


def f3_patterns(patterns: np.ndarray, times: np.ndarray, info: mne.Info,
                windows: Sequence[tuple[float, float]], out: Path) -> Path:
    """F3: Haufe pattern topographies, window means of the subject-averaged patterns."""
    fig, axes = plt.subplots(1, len(windows), figsize=(2.6 * len(windows), 2.8),
                             facecolor=SURFACE, squeeze=False)
    p = patterns.mean(0) if patterns.ndim == 3 else patterns
    vals = [p[:, (times >= a) & (times < b)].mean(1) for a, b in windows]
    lim = float(max(np.abs(v).max() for v in vals))
    for ax, v, (a, b) in zip(axes[0], vals, windows, strict=True):
        mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim),
                             contours=0)
        ax.set_title(f"{a * 1000:.0f}-{b * 1000:.0f} ms", fontsize=8, color=INK)
    fig.suptitle("F3  Haufe patterns (seen vs unseen)", fontsize=9, color=INK, x=0.02,
                 ha="left")
    return _save(fig, out, "F3_patterns")


def f_compare(times: np.ndarray, curves: Mapping[str, np.ndarray], title: str, name: str,
              out: Path, masks: Mapping[str, np.ndarray] | None = None) -> Path:
    """F4, F5, F6, F8a: several AUC(t) series on one axis."""
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    series_panel(ax, times, curves, masks)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return _save(fig, out, name)


def f7_paired(table: pd.DataFrame, columns: Sequence[str], out: Path) -> Path:
    """F7: per-subject window AUC for several models, lines join a subject's values."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0.5, color=GRID, lw=1, zorder=0)
    x = np.arange(len(columns))
    v = table[list(columns)].to_numpy(float)
    for row in v:
        ax.plot(x, row, color=GRID, lw=1, zorder=1)
    for i in range(len(columns)):
        ax.scatter(np.full(len(v), i), v[:, i], s=24, color=MUTED, edgecolors=SURFACE,
                   linewidths=1, zorder=2)
        m, h = _ci(v[:, i : i + 1])
        ax.errorbar(i + 0.15, m[0], yerr=h[0], fmt="o", color=INK, capsize=3, zorder=3)
        ax.text(i + 0.22, m[0], f"{m[0]:.3f}", fontsize=7, color=INK, va="center")
    ax.set_xticks(x, [c.replace("_", " ") for c in columns], fontsize=8, color=INK)
    ax.set_ylabel("Window AUC (per subject)", fontsize=8, color=MUTED)
    ax.set_title(f"F7  Prestimulus alpha vs post-stimulus (N = {len(v)})", fontsize=9,
                 color=INK, loc="left")
    fig.tight_layout()
    return _save(fig, out, "F7_alpha")


def f8_loso(times: np.ndarray, auc: np.ndarray, mask: np.ndarray, window_auc: pd.DataFrame,
            out: Path) -> Path:
    """F8: LOSO held-out AUC(t) and per-subject window AUC (sorted) with the group CI."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    series_panel(a1, times, {"pooled": auc}, {"pooled": mask})
    a1.set_title("F8  Leave-one-subject-out AUC(t)", fontsize=9, color=INK, loc="left")
    _style(a2)
    w = window_auc.sort_values("auc", ignore_index=True)
    a2.axhline(0.5, color=GRID, lw=1, zorder=0)
    a2.scatter(np.arange(len(w)), w["auc"], s=18, color=MUTED, zorder=2)
    r = stats.window_test(w["auc"].to_numpy())
    a2.axhspan(r["ci_low"], r["ci_high"], color=BAND, zorder=0)
    a2.axhline(r["mean"], color=INK, lw=1.5)
    a2.text(0, r["ci_high"] + 0.006, f"Mean {r['mean']:.3f} [{r['ci_low']:.3f}, "
            f"{r['ci_high']:.3f}]", fontsize=7, color=INK, va="bottom", ha="left")
    a2.set_xticks([])
    a2.set_xlabel("Subjects (sorted)", fontsize=8, color=MUTED)
    a2.set_ylabel("Held-out window AUC", fontsize=8, color=MUTED)
    fig.tight_layout()
    return _save(fig, out, "F8_loso")


def f9_cross(matrix: np.ndarray, tasks: Sequence[str], cross: Mapping[str, np.ndarray],
             out: Path) -> Path:
    """F9: cross-task 3x3 window AUC (group mean, train rows) and cross-condition dots."""
    m = np.nanmean(matrix, 0) if matrix.ndim == 3 else matrix
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE)
    lim = max(0.02, float(np.nanmax(np.abs(m - 0.5))))
    im = a1.imshow(m, cmap="RdBu_r", vmin=0.5 - lim, vmax=0.5 + lim)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            strong = abs(m[i, j] - 0.5) > 0.6 * lim  # dark cell: light text
            a1.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if strong else INK)
    names = [t[:1].upper() + t[1:] for t in tasks]
    a1.set_xticks(range(len(tasks)), names, fontsize=8)
    a1.set_yticks(range(len(tasks)), names, fontsize=8)
    a1.set_xlabel("Test task", fontsize=8, color=MUTED)
    a1.set_ylabel("Train task", fontsize=8, color=MUTED)
    _cbar(fig.colorbar(im, ax=a1, shrink=0.8))
    a1.set_title("F9  Cross-task", fontsize=9, color=INK, loc="left")
    _style(a2)
    a2.axhline(0.5, color=GRID, lw=1, zorder=0)
    for i, v in enumerate(cross.values()):
        v = np.asarray(v, float)
        a2.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=16,
                   color=MUTED, zorder=2)
        mm, h = _ci(v[:, None])
        a2.errorbar(i + 0.25, mm[0], yerr=h[0], fmt="o", color=INK, capsize=3, zorder=3)
        a2.text(i + 0.32, mm[0], f"{mm[0]:.3f}", fontsize=7, color=INK, va="center")
    a2.set_xticks(range(len(cross)), [k[:1].upper() + k[1:] for k in cross], fontsize=8)
    a2.set_ylabel("Window AUC", fontsize=8, color=MUTED)
    a2.set_title("Cross-hemifield / orientation", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return _save(fig, out, "F9_cross")


def f10_calibration(calib: pd.DataFrame, scores: Mapping[str, np.ndarray], out: Path,
                    name: str = "F10_calibration", title: str = "") -> Path:
    """F10: reliability diagram (calibration.tsv) and calibrated score distributions."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    _style(a1)
    c = calib.dropna()
    a1.plot([0, 1], [0, 1], color=GRID, lw=1)
    a1.plot(c["mean_p"], c["observed"], "o-", color=INK, lw=1.5, ms=5)
    for _, r in c.iterrows():
        a1.annotate(f"n={int(r['n'])}", (r["mean_p"], r["observed"]), xytext=(0, -9),
                    textcoords="offset points", fontsize=6, color=MUTED, ha="center")
    a1.set_xlim(-0.02, 1.02)
    a1.set_ylim(-0.08, 1.02)
    a1.set_xlabel("Calibrated P(seen)", fontsize=8, color=MUTED)
    a1.set_ylabel("Observed seen rate", fontsize=8, color=MUTED)
    a1.set_title(f"F10  Reliability (LOSO held-out){title}", fontsize=9, color=INK, loc="left")
    _style(a2)
    bins = np.linspace(0, 1, 26)
    styles = ("-", "--", "-.", ":")
    for i, (label, s) in enumerate(scores.items()):  # not `name`: that is the file name
        h, e = np.histogram(s, bins=bins, density=True)
        mid = (e[1:] + e[:-1]) / 2
        a2.plot(mid, h, color=_color(label, i), lw=2, ls=styles[i % 4],
                label=f"{label[:1].upper() + label[1:].replace('_', ' ')} (n = {len(s)})")
    a2.set_xlabel("Calibrated P(seen)", fontsize=8, color=MUTED)
    a2.set_ylabel("Density", fontsize=8, color=MUTED)
    a2.legend(fontsize=7, frameon=False)
    a2.set_title("Score distributions", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return _save(fig, out, name)


# ---------- tables ----------
def t1_descriptives(meta: pd.DataFrame) -> pd.DataFrame:
    """T1: per subject x task trial counts, hit/FA rates, contrast by class and overlap."""
    rows: list[dict[str, Any]] = []
    for (sub, task), g in meta.groupby(["subject", "task"], sort=True):
        pres = g[g["stim_present"].astype(bool)]
        catch = g[~g["stim_present"].astype(bool)]
        seen = pres["seen"].astype(bool)
        cs, cu = pres.loc[seen, "gabor_contrast"], pres.loc[~seen, "gabor_contrast"]
        overlap = (max(cs.min(), cu.min()), min(cs.max(), cu.max())) if len(cs) and len(
            cu) else (np.nan, np.nan)
        rows.append({
            "subject": sub, "task": task, "n_trials": len(g), "n_seen": int(seen.sum()),
            "n_unseen": int((~seen).sum()), "n_catch": len(catch),
            "hit_rate": float(seen.mean()) if len(pres) else np.nan,
            "fa_rate": float(catch["seen"].astype(bool).mean()) if len(catch) else np.nan,
            "contrast_seen_mean": float(cs.mean()), "contrast_seen_sd": float(cs.std()),
            "contrast_unseen_mean": float(cu.mean()), "contrast_unseen_sd": float(cu.std()),
            "contrast_overlap_low": overlap[0], "contrast_overlap_high": overlap[1],
            "share_in_overlap": float(pres["gabor_contrast"].between(*overlap).mean())
            if np.isfinite(overlap[0]) else np.nan,
        })
    return pd.DataFrame(rows)


def t2_tests(rows: Sequence[Mapping[str, Any]], family_col: str = "family") -> pd.DataFrame:
    """T2: every test with statistic, p, effect size, CI; Holm within each family."""
    df = pd.DataFrame(list(rows))
    df["p_holm"] = np.nan
    for _, idx in df.groupby(family_col).groups.items():
        df.loc[idx, "p_holm"] = stats.holm(df.loc[idx, "p"].to_numpy(float))
    return df


def t3_channels_windows(results: Mapping[tuple[int, float], Sequence[float]]) -> pd.DataFrame:
    """T3: held-out window AUC per (channel count, window start s) -> group summary."""
    rows = []
    for (n_ch, start), aucs in sorted(results.items()):
        r = stats.window_test(np.asarray(aucs, float))
        rows.append({"n_channels": n_ch, "window_start_ms": round(start * 1000),
                     "n": r["n"], "mean_auc": r["mean"], "ci_low": r["ci_low"],
                     "ci_high": r["ci_high"], "p": r["p"], "d_z": r["d_z"]})
    return pd.DataFrame(rows)
