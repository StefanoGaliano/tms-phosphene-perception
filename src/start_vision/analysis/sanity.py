"""Single-subject sanity report (PHASE1 Step 7).

For one subject x task, from the eegunify derivatives in ``runs/<task>/`` and the staged
events: seen vs unseen waves at the Step-2 ROIs (VAN O1/O2 contralateral, LP Pz/CPP3/CPP4),
ICA decisions, bad channels, rejection per condition, contrast balance after rejection.
Writes ``results/phase1/sanity/sub-XX_task-T.png`` and ``.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from start_vision import codes as C  # noqa: N812
from start_vision.matching import balance_report

# PHASE1_REFERENCES §2, §5
VAN_WIN: tuple[int, int] = (180, 280)
VAN_SITES: tuple[str, ...] = ("O1", "O2")
LP_WIN: tuple[int, int] = (350, 550)
LP_SITES: tuple[str, ...] = ("Pz", "CPP3", "CPP4")
PLOT_LOWPASS_HZ = 30.0  # REFERENCES §2: plots low-passed at 30 Hz, measures unfiltered


def _paths(run: Path, sub: str, task: str) -> dict[str, Path]:
    d = run / "derivatives"
    b = f"sub-{sub}_task-{task}"
    return {
        "ave": d / "eegunify-erp" / f"sub-{sub}" / "eeg" / f"{b}_desc-stim_ave.fif",
        "log": d / "eegunify-epochs" / f"sub-{sub}" / "eeg" / f"{b}_desc-stim_log.tsv",
        "ica": d / "eegunify-ica" / f"sub-{sub}" / "eeg" / f"{b}_desc-ica_components.tsv",
        "pre": d / "eegunify-preproc" / f"sub-{sub}" / "eeg" / f"{b}_desc-preproc_qa.json",
    }


def _mean(ev: mne.Evoked, sites: tuple[str, ...], win: tuple[int, int]) -> float:
    t = ev.times * 1000
    sel = (t >= win[0]) & (t <= win[1])
    return float(ev.copy().pick(list(sites)).data[:, sel].mean() * 1e6)


def report(run: Path, staged: Path, sub: str, task: str, out_dir: Path) -> dict[str, Any]:
    p = _paths(run, sub, task)
    evs = {e.comment: e for e in mne.read_evokeds(p["ave"], verbose="error")}
    log = pd.read_csv(p["log"], sep="\t")
    ica = pd.read_csv(p["ica"], sep="\t")
    pre = json.loads(p["pre"].read_text())
    ev_tsv = staged / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{task}_events.tsv"
    st = pd.read_csv(ev_tsv, sep="\t")
    st = st[st["value"] < C.RESPONSE_OFFSET].reset_index(drop=True)

    # rejection per condition (class) and contrast balance among kept matched trials
    log["cls"] = log["code"].astype(int).map(lambda v: C.decode(v).cls)
    rej = (
        (1 - log.groupby("cls")["kept"].mean())
        .rename(index={C.SEEN: "seen", C.UNSEEN: "unseen", C.FA: "fa", C.CR: "cr"})
        .round(3)
        .to_dict()
    )
    # join on the nearest epoch onset (the log is on the 256 Hz grid): within 5 ms
    lo = log["onset_s"].to_numpy()
    j = np.abs(st["onset"].to_numpy()[:, None] - lo[None, :]).argmin(1)
    near = np.abs(st["onset"].to_numpy() - lo[j]) < 0.005
    st["kept"] = near & log["kept"].to_numpy()[j]
    st["cls"] = st["value"].astype(int).map(lambda v: C.decode(v).cls)
    st["matched"] = st["value"] >= 1000
    k = st[st["kept"]]
    bal = balance_report(k, k["matched"])

    contra = {}
    if "seen_contra" in evs:
        for cl in ("seen", "unseen"):
            contra[cl] = evs[f"{cl}_contra"]
    meas = {
        "VAN_bilateral_seen_minus_unseen_m": _mean(evs["seen_minus_unseen_m"], VAN_SITES, VAN_WIN),
        "LP_seen_minus_unseen_m": _mean(evs["seen_minus_unseen_m"], LP_SITES, LP_WIN),
    }
    if contra:
        # eegunify's contra wave: both pair members carry the pair's contralateral value
        meas["VAN_contra_seen_minus_unseen"] = _mean(contra["seen"], ("O1",), VAN_WIN) - _mean(
            contra["unseen"], ("O1",), VAN_WIN
        )

    _figure(evs, contra, sub, task, out_dir / f"sub-{sub}_task-{task}.png")
    excl = ica[ica["excluded"]]
    out = {
        "subject": sub,
        "task": task,
        "bad_channels": pre["steps"].get("interpolated", []),
        "ica_excluded": [
            f"{i}: {r}" for i, r in zip(excl["idx"].astype(int), excl["reason"], strict=True)
        ],
        "ica_blink_removed": bool((excl["iclabel"] == "eye blink").any()),
        "n_kept": int(log["kept"].sum()),
        "n_epochs": len(log),
        "rejection_by_class": rej,
        "balance_after_rejection": {k2: round(v, 4) for k2, v in bal.items()},
        "measures_uv": {k2: round(v, 3) for k2, v in meas.items()},
        "lp_direction_ok": meas["LP_seen_minus_unseen_m"] > 0,
    }
    (out_dir / f"sub-{sub}_task-{task}.json").write_text(json.dumps(out, indent=1))
    return out


def _figure(
    evs: dict[str, mne.Evoked], contra: dict[str, mne.Evoked], sub: str, task: str, path: Path
) -> None:
    def wave(e: mne.Evoked, sites: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        f = e.copy().pick(list(sites)).filter(None, PLOT_LOWPASS_HZ, verbose="error")
        return f.times * 1000, f.data.mean(0) * 1e6

    panels: list[tuple[str, dict[str, mne.Evoked], tuple[str, ...], tuple[int, int]]] = [
        (
            "VAN: O1/O2 bilateral (matched)",
            {"seen": evs["seen_m"], "unseen": evs["unseen_m"]},
            VAN_SITES,
            VAN_WIN,
        ),
        (
            "LP: Pz/CPP3/CPP4 (matched)",
            {"seen": evs["seen_m"], "unseen": evs["unseen_m"]},
            LP_SITES,
            LP_WIN,
        ),
    ]
    if contra:
        panels.insert(1, ("VAN: contralateral O1/O2 (all)", contra, ("O1",), VAN_WIN))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6), sharey=False)
    for ax, (title, pair, sites, win) in zip(np.atleast_1d(axes), panels, strict=True):
        for (lab, e), col in zip(pair.items(), ("#1b6ca8", "#c8553d"), strict=True):
            t, y = wave(e, sites)
            ax.plot(t, y, color=col, label=f"{lab} (n={e.nave})")
        ax.axvspan(*win, color="0.9", zorder=0)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set_xlim(-100, 800)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ms from photodiode onset")
        ax.set_ylabel("µV")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"sub-{sub} {task}: seen vs unseen (plot low-pass {PLOT_LOWPASS_HZ:g} Hz)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
