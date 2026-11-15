"""Redraw every figure of paper/main.tex from saved results into paper/figures/ (300 dpi, white).

No analysis is rerun: inputs are the saved Phase 1 evokeds (runs/), Phase 2 group arrays
(derivatives/sv_decode_group/, derivatives/sv_decode/), detector held-out scores
(results/phase2/), Phase 3 tables (results/phase3/) and Q40 arrays
(results/questions_to_solve/prereg_q40/). Cluster masks are recomputed with the same seed and
permutation count as the group run, so they match it.

Usage: ~/miniconda3/envs/eegunify/bin/python scripts/paper_figures.py [p1 p2 p3 q40 q42_46]
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from start_vision.decoding import config, stats  # noqa: E402
from start_vision.decoding import figures as f2  # noqa: E402

OUT = ROOT / "paper" / "figures"


def _copy(src: Path, name: str) -> None:
    shutil.copy(src, OUT / name)
    print("wrote", OUT / name)


def phase1() -> None:
    from start_vision.analysis import figures, load, pool
    from start_vision.analysis.report import _exclusions
    from start_vision.codes import TASKS

    runs, staged = ROOT / "runs", ROOT / "sourcedata" / "staged" / "ds006171"
    excl = _exclusions(runs, staged, TASKS)
    keep = excl[(excl["task_rule"] == "") & (excl["subject_rule"] == "")]
    by_sub: dict[str, dict[str, dict]] = {}
    for sub, task in zip(keep["subject"].astype(str), keep["task"].astype(str), strict=True):
        by_sub.setdefault(sub, {})[task] = load.evokeds(runs / task, sub, task)
    pooled = {s: pool.pool_subject(t) for s, t in by_sub.items()}
    per = pd.read_csv(ROOT / "results/phase1/per_subject.tsv", sep="\t", dtype={"subject": str})
    grp = pd.read_csv(ROOT / "results/phase1/group.tsv", sep="\t")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _copy(figures.waves(pooled, "pooled", tmp), "p1_waves_pooled.png")
        _copy(figures.topomaps(pooled, "pooled", tmp), "p1_topo_pooled.png")
        _copy(figures.dots(per, grp, "pooled", tmp), "p1_subjects_pooled.png")


def phase2() -> None:
    cfg = config.load(ROOT / "configs" / "decoding.yaml")
    kw = {"n_permutations": cfg.n_permutations, "p": cfg.cluster_p, "seed": cfg.seed}
    g = ROOT / "derivatives" / "sv_decode_group"
    tmp = Path(tempfile.mkdtemp())
    a1 = np.load(g / "a1_sliding_pooled.npz", allow_pickle=True)
    times = a1["times"]
    f2.f1_sliding(times, {"pooled": a1["auc"]}, {"pooled": f2_mask(a1["auc"], kw)}, tmp)
    _copy(tmp / "F1_sliding.png", "p2_F1_sliding.png")
    lo = np.load(g / "a7_loso_sliding.npz", allow_pickle=True)
    table = pd.read_csv(ROOT / "results/phase2/a7_loso_window_per_subject.tsv", sep="\t")
    f2.f8_loso(times, lo["auc"], f2_mask(lo["auc"], kw), table, tmp)
    _copy(tmp / "F8_loso.png", "p2_F8_loso.png")
    # F9: per-subject 3x3 cross-task window AUC (same reduction as decoding.group)
    from start_vision.decoding.analyses import WIN_S
    from start_vision.decoding.group import _win

    order = list(cfg.tasks)
    mats = []
    for p in sorted((ROOT / "derivatives" / "sv_decode").glob("sub-*/eeg/*a8_cross_task_scores.npz")):
        z = np.load(p, allow_pickle=True)
        if "matrix" not in z.files:
            continue
        tk = [str(t) for t in z["tasks"]]
        m = np.full((3, 3), np.nan)
        wm = _win(z["matrix"].reshape(-1, len(times)), times, WIN_S).reshape(len(tk), len(tk))
        for i, ti in enumerate(tk):
            for j, tj in enumerate(tk):
                m[order.index(ti), order.index(tj)] = wm[i, j]
        mats.append(m)
    w9 = {}
    for key, label in (("hemifield", "cross-hemifield"),
                       ("hemifield_mirrored", "cross-hemifield, mirrored"),
                       ("orientation", "cross-orientation")):
        f = g / f"a9_cross_condition_{key}.npz"
        if f.exists():
            w9[label] = _win(np.load(f, allow_pickle=True)["auc"], times, WIN_S)
    f2.f9_cross(np.array(mats), order, w9, tmp)
    _copy(tmp / "F9_cross.png", "p2_F9_cross.png")
    h = np.load(ROOT / "results/phase2/detector_tms-robust_held_out.npz")
    calib = pd.read_csv(ROOT / "detectors/sv-perception-tms-robust-v1.0.0/calibration.tsv",
                        sep="\t")
    scores = {k.replace("_", " "): h[k] for k in h.files}
    f2.f10_calibration(calib, scores, tmp, name="F10_tms-robust", title=": tms-robust")
    _copy(tmp / "F10_tms-robust.png", "p2_F10_tms-robust.png")


def f2_mask(auc: np.ndarray, kw: dict) -> np.ndarray:
    return stats.cluster_mask(auc, **kw)


def phase3() -> None:
    from start_vision.phase3 import figures as f3

    tmp = Path(tempfile.mkdtemp())
    real_out = ROOT / "results" / "phase3" / "figures"
    # make_all writes into results/phase3/figures; run it, then copy the paper's panels
    f3.make_all(ROOT, "primary")
    for src, dst in (("F1_cleaning_gmfp", "p3_F1_cleaning_gmfp"),
                     ("F3_auc_per_recording", "p3_F3_auc_per_recording"),
                     ("F5_controls", "p3_F5_controls"), ("F7_s6_erps", "p3_F7_s6_erps"),
                     ("F9_pipeline_robustness", "p3_F9_pipeline_robustness")):
        _copy(real_out / f"{src}.png", f"{dst}.png")
    del tmp


def q40() -> None:
    d = ROOT / "results" / "questions_to_solve" / "prereg_q40"
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.axvspan(0.15, 0.6, color="#f2f2f2", lw=0, zorder=0)
    ax.text(0.375, 0.538, "Pre-registered window", fontsize=7, color="#52514e", ha="center")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.axvline(0, color="grey", lw=1)
    for p, c, lab in (("primary", "#2a78d6", "Primary pipeline"),
                      ("paperlike", "#eb6834", "Paperlike pipeline")):
        z = np.load(d / f"q40_auc_{p}.npz", allow_pickle=True)
        a, t = z["auc"], z["times"]
        ci = np.array([stats.bootstrap_ci(a[:, i], 10_000, 0) for i in range(a.shape[1])])
        ax.fill_between(t, ci[:, 0], ci[:, 1], color=c, alpha=0.18, lw=0)
        ax.plot(t, a.mean(0), color=c, lw=2, label=lab)
    ax.set_xlim(t[0], t[-1])
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Time after stimulus onset (s)")
    ax.set_ylabel("AUC (seen vs unseen)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "q40_reverse_transfer.png", dpi=300, facecolor="white")
    print("wrote", OUT / "q40_reverse_transfer.png")


def q42_46() -> None:
    """Transfer of each detector recipe: occipital (recipes chosen here) vs held-out parietal."""
    import json
    q = ROOT / "results" / "questions_to_solve"
    j = {k: json.loads((q / f"prereg_{k}" / f"{k}.json").read_text())["primary"]
         for k in ("q42", "q43", "q44", "q46")}
    h1 = pd.read_csv(ROOT / "results" / "phase3" / "per_recording_primary.tsv", sep="\t",
                     dtype={"subject": str}).groupby("subject")["auc"].mean().to_numpy()
    lo, hi = stats.bootstrap_ci(h1, 10_000, 0)

    def ci(d: dict) -> tuple[float, float, float]:
        return d["mean"] + 0.5, d["ci95"][0] + 0.5, d["ci95"][1] + 0.5

    occ = [("Frozen detector", (h1.mean(), lo, hi)),
           ("Window mean", ci(j["q42"]["M1_vs_chance"])),
           ("Label-free alignment", ci(j["q43"]["M0_aligned_vs_chance"])),
           ("Two datasets", ci(j["q44"]["M0_pooled_vs_chance"])),
           ("Two datasets + window mean (v2)", ci(j["q44"]["M1_pooled_vs_chance"])),
           ("Eklund only + window mean", ci(j["q44"]["M1_eklund_only_vs_chance"]))]
    fz = j["q46"]["P_v2_minus_frozen"]
    ips_sub = pd.DataFrame(j["q46"]["subject_auc"])
    fl, fh = stats.bootstrap_ci(ips_sub["frozen"].to_numpy(), 10_000, 0)
    ips = [("Frozen detector", (ips_sub["frozen"].mean(), fl, fh)),
           ("Two datasets + window mean (v2)", ci(j["q46"]["a_v2_vs_chance"])),
           ("Eklund only + window mean", ci(j["q46"]["d_eklund_only_vs_chance"]))]
    del fz
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True,
                             gridspec_kw={"width_ratios": [6, 3]})
    for ax, rows, title, c in ((axes[0], occ, "Occipital phosphenes (recipes chosen here)", "#2a78d6"),
                               (axes[1], ips, "Parietal phosphenes (held out)", "#eb6834")):
        x = np.arange(len(rows))
        m = np.array([r[1][0] for r in rows])
        e = np.array([[r[1][0] - r[1][1], r[1][2] - r[1][0]] for r in rows]).T
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.errorbar(x, m, yerr=e, fmt="o", color=c, ms=5, capsize=3, lw=1.2)
        ax.set_xticks(x, [r[0] for r in rows], rotation=35, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("AUC (phosphene vs none)")
    fig.tight_layout()
    fig.savefig(OUT / "q42_46_transfer.png", dpi=300, facecolor="white")
    print("wrote", OUT / "q42_46_transfer.png")


if __name__ == "__main__":
    jobs = {"p1": phase1, "p2": phase2, "p3": phase3, "q40": q40, "q42_46": q42_46}
    for k in sys.argv[1:] or list(jobs):
        jobs[k]()
