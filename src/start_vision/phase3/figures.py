"""Phase 3 figures (PHASE3_BLUEPRINT §11) into ``results/phase3/figures/``.

F1 cleaning QA (GMFP per stage, every recording), F2 pulse-artifact end, F3 per-recording and
per-subject AUC, F4 P(seen) distributions + reliability (S4), F5 controls vs H1, F6 R3 pattern
lateralization, F7 S6 ERPs, F8 S8 within-recording decoding, F9 pipeline robustness (S5).
S7 is not drawn (needs the ds006171 epochs). Style and palette: ``decoding.figures`` (dataviz
reference palette, validated; aqua/yellow below 3:1 contrast, so every series is labelled).
Entities keep one colour: phosphene = seen (blue), none = unseen (orange), O1 = aqua, O2 = yellow.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from start_vision.decoding.figures import BAND, ENTITY, GRID, INK, MUTED, SLOTS, _ci, _save, _style

SITE = {"O1": SLOTS[2], "O2": SLOTS[3]}
PHOS, NONE = ENTITY["seen"], ENTITY["unseen"]
STAGES = (("raw", "Raw"), ("cut", "Pulse cut"), ("decay_ica", "Decay ICA"),
          ("ocular_ica", "Ocular ICA"))


def f1_cleaning(qa_dir: Path, recordings: list[str], out: Path) -> Path:
    n = len(recordings)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 1.7), sharex=True)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax, rid in zip(axs.ravel(), recordings, strict=False):
        d = np.load(qa_dir / f"{rid}_desc-qaerp.npz")
        for i, (k, lab) in enumerate(STAGES):
            t, x = d[f"{k}_times"], d[f"{k}_avg"]
            m = (t >= -0.1) & (t <= 0.6)
            ax.plot(t[m] * 1e3, x[:, m].std(0) * 1e6,
                    color=SLOTS[i], lw=1, label=lab)
        ax.set_yscale("log")
        ax.set_title(rid.replace("_site-", " "), fontsize=7, color=INK)
        _style(ax)
    axs.ravel()[0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.supxlabel("Time after pulse (ms)", fontsize=8, color=MUTED)
    fig.supylabel("GMFP (µV, log)", fontsize=8, color=MUTED)
    fig.tight_layout()
    return _save(fig, out, "F1_cleaning_gmfp")


def f2_artifact_end(qa_dir: Path, recordings: list[str], out: Path) -> Path:
    rows = [json.loads((qa_dir / f"{r}_desc-qa.json").read_text()) for r in recordings]
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(rows))
    end = [r["artifact_end_ms"] for r in rows]
    cut = [r["cut_ms"][1] for r in rows]
    ax.vlines(x, 0, end, color=SLOTS[0], lw=2, label="measured artifact end")
    ax.scatter(x, cut, color=SLOTS[1], s=18, zorder=3, label="cut end used")
    ax.axhline(20, color=MUTED, lw=1, ls="--")
    ax.text(len(x) - 0.5, 20.5, "20 ms cap", fontsize=7, color=MUTED, ha="right")
    ax.set_xticks(x, [r.replace("_site-", " ").removeprefix("sub-") for r in recordings],
                  rotation=90, fontsize=6)
    ax.set_ylabel("ms after pulse", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, "F2_artifact_end")


def f3_auc(per_rec: pd.DataFrame, h1: pd.Series, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    subs = sorted(per_rec["subject"].unique())
    xs = {s: i for i, s in enumerate(subs)}
    ax.axhline(0.5, color=GRID, lw=1)
    for site, off in (("O1", -0.12), ("O2", 0.12)):
        g = per_rec[per_rec["site"] == site]
        ax.scatter([xs[s] + off for s in g["subject"]], g["auc"], color=SITE[site], s=22,
                   label=f"Site {site}", zorder=3)
    sm = per_rec.groupby("subject")["auc"].mean()
    ax.scatter([xs[s] for s in sm.index], sm, marker="_", s=160, color=INK, lw=2,
               label="Subject mean", zorder=4)
    ax.axhspan(h1["ci_low"], h1["ci_high"], color=BAND, zorder=0)
    ax.axhline(h1["mean"], color=INK, lw=1)
    ax.text(-0.6, h1["ci_high"] + 0.002, f"Group mean {h1['mean']:.3f} (95 % CI band)",
            fontsize=7, color=INK, ha="left")
    ax.set_xticks(range(len(subs)), subs, fontsize=7)
    ax.set_xlabel("Subject", fontsize=8, color=MUTED)
    ax.set_ylabel("AUC, phosphene vs none", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left")
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, "F3_auc_per_recording")


def f4_calibration(scores: pd.DataFrame, bins: pd.DataFrame, out: Path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    edges = np.linspace(0, 1, 41)
    for v, c, lab in ((1, PHOS, "phosphene"), (0, NONE, "no phosphene")):
        a.hist(scores.loc[scores["seen"] == v, "p_seen"], bins=edges, histtype="step", lw=2,
               color=c, density=True, label=lab)
    a.set_xlabel("P(seen)", fontsize=8, color=MUTED)
    a.set_ylabel("density", fontsize=8, color=MUTED)
    a.legend(fontsize=7, frameon=False)
    b.plot([0, 1], [0, 1], color=GRID, lw=1)
    b.plot(bins["train_mean_p"], bins["train_observed"], "o-", color=MUTED, lw=2, ms=4,
           label="training (ds006171, seen rate)")
    ok = bins["tms_n"] >= 20
    b.plot(bins.loc[ok, "tms_mean_p"], bins.loc[ok, "tms_observed"], "o-", color=PHOS, lw=2,
           ms=4, label="TMS (phosphene rate), bins n >= 20")
    b.set_xlabel("mean P(seen) in bin", fontsize=8, color=MUTED)
    b.set_ylabel("observed rate", fontsize=8, color=MUTED)
    b.legend(fontsize=7, frameon=False)
    for ax in (a, b):
        _style(ax)
    fig.tight_layout()
    return _save(fig, out, "F4_pseen_calibration")


def f5_controls(t2: pd.DataFrame, t3: pd.DataFrame, out: Path) -> Path:
    rows = [("H1 primary", t2[t2["test"] == "H1 primary"].iloc[0])]
    for name in ("R1 motor channels removed", "R4a early bins only (< 0.30 s)",
                 "E1 follow-up: H1 on trials with HEOG step < 16 uV",
                 "A1 H1 with P(seen) residualized on alpha", "R5 C3-C4 alone", "E1 HEOG alone",
                 "E1 VEOG alone", "A1 prestim alpha alone"):
        m = t3[t3["control"] == name]
        if len(m):
            rows.append((name, m.iloc[0]))
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 0.8))
    for i, (name, r) in enumerate(rows):
        y = len(rows) - 1 - i
        has_ci = "ci_low" in r and pd.notna(r.get("ci_low"))
        if has_ci:
            ax.hlines(y, r["ci_low"], r["ci_high"], color=INK if i == 0 else SLOTS[0], lw=2)
        ax.scatter(r["mean"], y, color=INK if i == 0 else SLOTS[0], s=30, zorder=3)
        ax.text(0.425, y, name, fontsize=7, color=INK, va="center", ha="left")
    ax.axvline(0.5, color=GRID, lw=1)
    ax.set_xlim(0.42, 0.58)
    ax.set_yticks([])
    ax.set_xlabel("AUC (dots: mean over subjects; lines: 95 % CI where tested)", fontsize=8,
                  color=MUTED)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, "F5_controls")


def f6_r3(r3: pd.DataFrame, out: Path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3))
    ms = (r3["bin_start_s"] * 1e3).to_numpy()
    a.bar(ms, r3["c3_minus_c4_rel"], width=40, color=SLOTS[0])
    a.axhline(0, color=GRID, lw=1)
    a.set_ylabel("(C3 - C4) / mean |posterior|", fontsize=8, color=MUTED)
    b.fill_between(ms, -r3["null95_abs_r"], r3["null95_abs_r"], color=BAND, step="mid",
                   label="95 % of permuted maps")
    b.plot(ms, r3["spatial_r"], "o-", color=SLOTS[0], lw=2, ms=4,
           label="r(pattern, TMS lateralization map)")
    b.axhline(0, color=GRID, lw=1)
    b.set_ylabel("spatial r", fontsize=8, color=MUTED)
    b.legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.35))
    for ax in (a, b):
        ax.set_xlabel("detector bin start (ms)", fontsize=8, color=MUTED)
        _style(ax)
    fig.tight_layout()
    return _save(fig, out, "F6_r3_pattern_lateralization")


def f7_erps(curves: dict[str, np.ndarray], s6c: pd.DataFrame, out: Path) -> Path:
    t = curves["erp_times"] * 1e3
    m = (t >= -100) & (t <= 800)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for ax, roi, title, win in ((axs[0], "contra", "Contralateral to phosphene (O1/PO7 or O2/PO8)",
                                 (180, 280)),
                                (axs[1], "lp", "Pz", (350, 550))):
        ax.axvspan(*win, color=BAND, zorder=0)
        for k, c, lab in (("seen", PHOS, "Phosphene"), ("unseen", NONE, "No phosphene")):
            mu, h = _ci(curves[f"{roi}_{k}"])
            ax.fill_between(t[m], (mu - h)[m], (mu + h)[m], color=c, alpha=0.15, lw=0)
            ax.plot(t[m], mu[m], color=c, lw=2, label=lab)
        name = "VAN contralateral" if roi == "contra" else "LP Pz"
        sig = s6c[(s6c["roi"] == name) & (s6c["p"] < 0.05)]
        for t0, t1 in zip(sig["t_start"].astype(float), sig["t_end"].astype(float), strict=True):
            ax.axvspan(t0 * 1e3, t1 * 1e3, ymin=0, ymax=0.03, color=INK)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("Time after pulse (ms)", fontsize=8, color=MUTED)
        _style(ax)
    axs[0].set_ylabel("Amplitude (µV, baseline -100 to 0 ms)", fontsize=8, color=MUTED)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _save(fig, out, "F7_s6_erps")


def f8_s8(curves: dict[str, np.ndarray], s8c: pd.DataFrame, out: Path) -> Path:
    t = curves["s8_times"] * 1e3
    mu, h = _ci(curves["s8_auc"])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0.5, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    ax.fill_between(t, mu - h, mu + h, color=SLOTS[0], alpha=0.15, lw=0)
    n = len(curves["s8_auc"])
    ax.plot(t, mu, color=SLOTS[0], lw=2, label=f"within-recording AUC (N = {n})")
    sig = s8c[s8c["p"] < 0.05]
    for t0, t1 in zip(sig["t_start"].astype(float), sig["t_end"].astype(float), strict=True):
        ax.axvspan(t0 * 1e3, t1 * 1e3, ymin=0, ymax=0.03, color=INK)
    ax.set_xlabel("ms after pulse", fontsize=8, color=MUTED)
    ax.set_ylabel("AUC", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, "F8_s8_within_decoding")


def f9_pipelines(per_a: pd.DataFrame, per_b: pd.DataFrame, out: Path) -> Path:
    a = per_a.groupby("subject")["auc"].mean()
    b = per_b.groupby("subject")["auc"].mean().reindex(a.index)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lo, hi = min(a.min(), b.min()) - 0.01, max(a.max(), b.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=1)
    ax.axhline(0.5, color=GRID, lw=1, ls=":")
    ax.axvline(0.5, color=GRID, lw=1, ls=":")
    ax.scatter(a, b, color=SLOTS[0], s=26, zorder=3)
    ax.set_xlabel("Subject AUC, primary pipeline", fontsize=8, color=MUTED)
    ax.set_ylabel("Subject AUC, paperlike pipeline", fontsize=8, color=MUTED)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, "F9_pipeline_robustness")


def make_all(root: Path, pipeline: str = "primary") -> list[Path]:
    res = root / "results" / "phase3"
    out = res / "figures"
    inc = pd.read_csv(res / f"inclusion_{pipeline}.tsv", sep="\t", dtype={"subject": str})
    recs_all = sorted(inc["recording"])
    keep = set(inc.loc[inc["included"], "recording"])
    per = pd.read_csv(res / f"per_recording_{pipeline}.tsv", sep="\t", dtype={"subject": str})
    per = per[per["recording"].isin(keep)]
    scores = pd.read_csv(res / f"scores_{pipeline}.tsv", sep="\t", dtype={"subject": str})
    scores = scores[scores["recording"].isin(keep)]
    t2 = pd.read_csv(res / f"t2_tests_{pipeline}.tsv", sep="\t")
    t3 = pd.read_csv(res / f"t3_controls_{pipeline}.tsv", sep="\t")
    curves = dict(np.load(res / f"secondary_curves_{pipeline}.npz"))
    qa = root / "derivatives" / "sv_p3_clean" / pipeline
    paths = [f1_cleaning(qa, recs_all, out), f2_artifact_end(qa, recs_all, out),
             f3_auc(per, t2[t2["test"] == "H1 primary"].iloc[0], out),
             f4_calibration(scores, pd.read_csv(res / f"s4_bins_{pipeline}.tsv", sep="\t"), out),
             f5_controls(t2, t3, out),
             f6_r3(pd.read_csv(res / f"r3_{pipeline}.tsv", sep="\t"), out),
             f7_erps(curves, pd.read_csv(res / f"s6_clusters_{pipeline}.tsv", sep="\t"), out),
             f8_s8(curves, pd.read_csv(res / f"s8_clusters_{pipeline}.tsv", sep="\t"), out)]
    other = res / "per_recording_paperlike.tsv"
    if pipeline == "primary" and other.exists():
        pb = pd.read_csv(other, sep="\t", dtype={"subject": str})
        paths.append(f9_pipelines(per, pb[pb["recording"].isin(keep)], out))
    return paths
