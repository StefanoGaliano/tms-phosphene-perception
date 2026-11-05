"""Group figures (PHASE1 §7 analysis/figures.py). Static PNGs for results/phase1/figures/.

Colours: dataviz reference palette, categorical slots 1-2 (seen blue, unseen orange),
validated with the skill's validator (light surface: all checks pass, CVD ΔE 24.7). The
difference wave is drawn in text ink with its 95 % CI band; identity is never colour alone
(legend + direct labels). Topographies use MNE's diverging map (two hues, neutral midpoint).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats

SEEN, UNSEEN = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SURFACE = "#fcfcfb"
WINDOW = "#6250d6"  # palette slot 7 (violet), used only as a pale tint for the component window
LOWPASS_HZ = 30.0  # PHASE1_REFERENCES §2: plots low-passed at 30 Hz

PANELS = {  # name -> (seen wave, unseen wave, channels, window, title)
    "van_contra": (
        "seen_contra",
        "unseen_contra",
        ("O1",),
        (180, 280),
        "VAN, contralateral O1/O2 (primary)",
    ),
    "van_bilateral": (
        "seen_m",
        "unseen_m",
        ("O1", "O2"),
        (180, 280),
        "VAN, O1/O2 bilateral, contrast-matched",
    ),
    "lp": (
        "seen_m",
        "unseen_m",
        ("Pz", "CPP3", "CPP4"),
        (350, 550),
        "LP, Pz/CPP3/CPP4, contrast-matched",
    ),
}


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)


def _roi(ev: mne.Evoked, chans: tuple[str, ...]) -> np.ndarray:
    e = ev.copy().pick(list(chans)).filter(None, LOWPASS_HZ, verbose="error")
    return np.asarray(e.data.mean(0) * 1e6)


def waves(per_subject: dict[str, dict[str, mne.Evoked]], scope: str, out: Path) -> Path:
    """Grand-average seen/unseen and the difference with a 95 % CI, one column per measure."""
    panels = [
        v for v in PANELS.values() if any(v[0] in e and v[1] in e for e in per_subject.values())
    ]
    fig, axes = plt.subplots(
        2,
        len(panels),
        figsize=(4.6 * len(panels), 6.2),
        squeeze=False,
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1]},
        facecolor=SURFACE,
    )
    for j, (sw, uw, chans, win, title) in enumerate(panels):
        subs = [s for s, e in per_subject.items() if sw in e and uw in e]
        if not subs:
            axes[0, j].set_visible(False)
            axes[1, j].set_visible(False)
            continue
        t = per_subject[subs[0]][sw].times * 1000
        s = np.array([_roi(per_subject[x][sw], chans) for x in subs])
        u = np.array([_roi(per_subject[x][uw], chans) for x in subs])
        d = s - u
        n = len(subs)
        half = stats.t.ppf(0.975, n - 1) * d.std(0, ddof=1) / np.sqrt(n)
        top, bot = axes[0, j], axes[1, j]
        comp = title.split(",")[0]  # "VAN" or "LP"
        for ax in (top, bot):
            _style(ax)
            ax.axvspan(*win, color=WINDOW, alpha=0.13, lw=0, zorder=0)
        top.text(
            (win[0] + win[1]) / 2, 0.98, f"{comp}\n{win[0]}-{win[1]} ms",
            transform=top.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, color=INK, fontweight="bold",
        )
        top.plot(t, s.mean(0), color=SEEN, lw=2, label="seen")
        top.plot(t, u.mean(0), color=UNSEEN, lw=2, label="unseen")
        k = int(np.argmin(np.abs(t - 700)))
        top.text(t[k] + 8, s.mean(0)[k], "seen", color=INK, fontsize=8, va="center")
        top.text(t[k] + 8, u.mean(0)[k], "unseen", color=INK, fontsize=8, va="center")
        top.set_title(f"{title}\nN = {n}", fontsize=9, color=INK, loc="left")
        top.legend(fontsize=8, frameon=False, loc="upper left")
        bot.fill_between(t, d.mean(0) - half, d.mean(0) + half, color=GRID, lw=0)
        bot.plot(t, d.mean(0), color=INK, lw=2)
        bot.set_xlabel("ms from Gabor onset (photodiode)", fontsize=8, color=MUTED)
        bot.set_xlim(-100, 800)
        if j == 0:
            top.set_ylabel("µV", fontsize=8, color=MUTED)
            bot.set_ylabel("seen - unseen, µV\n(95 % CI)", fontsize=8, color=MUTED)
    fig.suptitle(
        f"Phase 1 grand averages, scope: {scope} (violet band: component window; "
        f"plots low-passed {LOWPASS_HZ:g} Hz)",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    path = out / f"waves_{scope}.png"
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return path


def dots(per_subject_tsv: pd.DataFrame, group: pd.DataFrame, scope: str, out: Path) -> Path:
    """Per-subject seen - unseen differences with the group mean and its 95 % CI."""
    forms = [
        ("van", "contralateral"),
        ("van", "bilateral_matched"),
        ("van", "double_difference"),
        ("lp", "bilateral_matched"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURFACE)
    _style(ax)
    ax.axvline(0, color=GRID, lw=0)
    rng = np.random.default_rng(0)
    labels = []
    for i, (comp, form) in enumerate(forms):
        x = per_subject_tsv[
            (per_subject_tsv.scope == scope)
            & (per_subject_tsv.component == comp)
            & (per_subject_tsv.form == form)
        ]["diff"].to_numpy()
        if not len(x):
            continue
        g = group[(group.scope == scope) & (group.component == comp) & (group.form == form)]
        ax.scatter(
            i + rng.uniform(-0.12, 0.12, len(x)),
            x,
            s=22,
            color=MUTED,
            alpha=0.6,
            edgecolors=SURFACE,
            linewidths=1,
            zorder=2,
        )
        m = x.mean()
        half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        ax.errorbar(i + 0.3, m, yerr=half, fmt="o", color=INK, ms=6, capsize=3, zorder=3)
        dz = float(g["d_z"].iloc[0]) if len(g) else np.nan
        ax.text(i + 0.38, m, f"{m:+.2f} µV\nd_z {dz:+.2f}", fontsize=7, color=INK, va="center")
        labels.append((i, f"{comp.upper()}\n{form.replace('_', ' ')}"))
    ax.set_xticks([i for i, _ in labels], [lab for _, lab in labels], fontsize=8, color=INK)
    ax.set_ylabel("seen - unseen, µV", fontsize=8, color=MUTED)
    ax.set_title(
        f"Per-subject effects, scope: {scope} (grey: subjects; black: mean ± 95 % CI)",
        fontsize=9,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    path = out / f"subjects_{scope}.png"
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return path


def topomaps(per_subject: dict[str, dict[str, mne.Evoked]], scope: str, out: Path) -> Path:
    """Grand-average seen_minus_unseen_m topographies in the VAN and LP windows."""
    evs = [e["seen_minus_unseen_m"] for e in per_subject.values() if "seen_minus_unseen_m" in e]
    common = [c for c in evs[0].ch_names if all(c in e.ch_names for e in evs)]
    ga = mne.grand_average([e.copy().pick(common) for e in evs])
    ga.pick("eeg")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), facecolor=SURFACE)
    for ax, (name, win) in zip(
        axes, (("VAN 180-280 ms", (0.18, 0.28)), ("LP 350-550 ms", (0.35, 0.55))), strict=True
    ):
        sel = (ga.times >= win[0]) & (ga.times <= win[1])
        data = ga.data[:, sel].mean(1) * 1e6
        lim = float(np.abs(data).max())
        im, _ = mne.viz.plot_topomap(
            data, ga.info, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim), contours=0
        )
        ax.set_title(name, fontsize=9, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.7, label="µV")
    fig.suptitle(f"seen - unseen (matched), scope: {scope}, N = {len(evs)}", fontsize=10, color=INK)
    path = out / f"topo_{scope}.png"
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return path
