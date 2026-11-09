"""Q38 phosphene VAN topography and a wider response-hand control (P§9, amendment 2)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from start_vision.decoding.detector import Detector
from start_vision.phase3.analyses import group_one_sided, rec_auc, residualized, subject_values
from start_vision.phase3.score import detector_input
from start_vision.questions import figures as fg
from start_vision.questions import io
from start_vision.questions.params import (
    HAND_BASELINE_S,
    HAND_WINDOW_S,
    LP_SAMAHA_WINDOW_S,
    S6_VAN_WINDOW_S,
    SEED,
    VAN_OZ_WINDOW_S,
)
from start_vision.questions.report import group_row, write_readme
from start_vision.questions.stats import one_sample

DETECTOR = Path("detectors") / "sv-perception-tms-robust-v1.0.0"
ERP_BASELINE_S = (-0.1, 0.0)    # S6 baseline (PHASE3 S6, Eklund)
CONTRA = {"O1": ["O1", "PO7"], "O2": ["O2", "PO8"]}  # S6 ROI


def mirror_pairs(channels: list[str]) -> list[tuple[str, str]]:
    """Odd-numbered channel and its even-numbered mirror (n + 1); midline (z) excluded (P§9)."""
    out = []
    for c in channels:
        m = re.fullmatch(r"([A-Za-z]+?)(\d+)", c)
        if m and int(m.group(2)) % 2 == 1:
            mate = f"{m.group(1)}{int(m.group(2)) + 1}"
            if mate in channels:
                out.append((c, mate))
    return out


def _win(t: np.ndarray, w: tuple[float, float]) -> np.ndarray:
    return (t >= w[0] - 1e-9) & (t <= w[1] + 1e-9)


def hand_index(x: np.ndarray, channels: list[str], times: np.ndarray) -> np.ndarray:
    """Per trial: mean over mirrored pairs of (left - right), 0.15-0.6 s, contract baseline."""
    xb = x - x[..., _win(times, HAND_BASELINE_S)].mean(-1, keepdims=True)
    w = _win(times, HAND_WINDOW_S)
    pairs = mirror_pairs(channels)
    li = [channels.index(a) for a, _ in pairs]
    ri = [channels.index(b) for _, b in pairs]
    return np.asarray((xb[:, li][..., w] - xb[:, ri][..., w]).mean(axis=(1, 2)))


def recording_values(g: pd.DataFrame) -> dict[str, float]:
    gg = g.assign(p_res=residualized(g, "p_seen", "hand"))
    return {"auc_frozen": rec_auc(g, "p_seen"), "auc_res": rec_auc(gg, "p_res"),
            "auc_hand": rec_auc(g, "hand")}


def _erp_diff(ep: Any, chans: list[str], win: tuple[float, float]) -> float:
    e = ep.copy().pick(chans)
    t = e.times
    x = e.get_data() * 1e6
    x = x - x[..., _win(t, ERP_BASELINE_S)].mean(-1, keepdims=True)
    v = x[..., _win(t, win)].mean(axis=(1, 2))
    y = e.metadata["seen"].to_numpy().astype(bool)
    return float(v[y].mean() - v[~y].mean())


def run(root: Path, force: bool = False) -> Path:
    d = io.out_dir(root, "q38", force)
    det = Detector.load(root / DETECTOR)
    ch = list(det.contract["channels"])
    lateral = sorted({c for p in mirror_pairs(ch) for c in p})
    scores = io.gin_scores(root)
    rows = []
    for rid, ep in io.gin_epochs_iter(root, sorted(scores["recording"].unique())):
        e = detector_input(ep, det)
        g = scores[scores["recording"] == rid].reset_index(drop=True)
        if not np.array_equal(g["seen"].to_numpy(), ep.metadata["seen"].to_numpy()):
            raise ValueError(f"{rid}: scores and epochs disagree (pulse order)")
        g = g.assign(hand=hand_index(e.get_data() * 1e6, list(e.ch_names), e.times))
        site = str(g["site"].iloc[0])
        wide = e.copy()
        wide.info["bads"] = lateral
        wide.interpolate_bads(reset_bads=True, verbose="error")
        g["p_wide"] = det.from_epochs(wide)
        r: dict[str, Any] = {"recording": rid, "subject": g["subject"].iloc[0], "site": site,
                             **recording_values(g), "auc_wide": rec_auc(g, "p_wide"),
                             "van_oz_180_220": _erp_diff(ep, ["Oz"], VAN_OZ_WINDOW_S),
                             "van_s6_roi_180_280": _erp_diff(ep, CONTRA[site], S6_VAN_WINDOW_S),
                             "lp_pz_300_800": _erp_diff(ep, ["Pz"], LP_SAMAHA_WINDOW_S)}
        rows.append(r)
    per = pd.DataFrame(rows)
    per.to_csv(d / "per_recording.tsv", sep="\t", index=False, float_format="%.5f")
    prim = group_one_sided(subject_values(per, "auc_res"), seed=SEED)
    wide_r = group_one_sided(subject_values(per, "auc_wide"), seed=SEED)
    hand_r = one_sample(subject_values(per, "auc_hand"), 0.5, tail=2)
    tab = [{"test": "H1 with P(seen) residualized on the hand index > 0.5", "role": "[P]",
            "n": prim["n"], "mean": prim["mean"], "ci_low": prim["ci_low"],
            "ci_high": prim["ci_high"], "d_z": prim["d_z"], "p": prim["p"], "tail": 1.0,
            "n_positive": prim["n_above"]},
           {"test": f"(a) H1 with all {len(lateral)} lateral channels interpolated", "role":
            "secondary", "n": wide_r["n"], "mean": wide_r["mean"], "ci_low": wide_r["ci_low"],
            "ci_high": wide_r["ci_high"], "p": wide_r["p"], "tail": 1.0},
           group_row("(b) hand index alone, AUC vs 0.5", "secondary", hand_r)]
    for site in ("O1", "O2"):
        s = per[per["site"] == site]
        tab.append(group_row(f"(c) {site}: residualized AUC vs 0.5", "secondary",
                             one_sample(s["auc_res"], 0.5, tail=1)))
        for col in ("van_oz_180_220", "van_s6_roi_180_280", "lp_pz_300_800"):
            tab.append(group_row(f"{site}: {col} (µV, phosphene - none)", "descriptive (seen)",
                                 one_sample(s[col], 0.0, tail=2)))
    t = pd.DataFrame(tab)
    t.to_csv(d / "group.tsv", sep="\t", index=False, float_format="%.5f")
    io.write_primary(d, "q38", "H1 residualized on the hand index", "one-sided",
                     {k: float(prim[k]) for k in ("n", "mean", "ci_low", "ci_high", "p")})
    fig, (a, b) = fg.axes(2)
    sf, sr = subject_values(per, "auc_frozen"), subject_values(per, "auc_res")
    a.scatter(sf, sr, color=fg.SLOTS[0], s=20, label=f"subjects (N {len(sf)})")
    a.plot([0.44, 0.66], [0.44, 0.66], color=fg.GRID, lw=1)
    fg.style(a, "subject AUC, frozen", "subject AUC, hand-residualized")
    a.legend(fontsize=7, frameon=False)
    for i, site in enumerate(("O1", "O2")):
        s = per[per["site"] == site]
        b.scatter(np.full(len(s), i) - 0.1, s["van_oz_180_220"], color=fg.SLOTS[0], s=16,
                  label="Oz 180-220 ms" if i == 0 else None)
        b.scatter(np.full(len(s), i) + 0.1, s["van_s6_roi_180_280"], color=fg.SLOTS[1], s=16,
                  label="S6 ROI 180-280 ms" if i == 0 else None)
    b.axhline(0, color=fg.GRID, lw=1)
    b.set_xticks([0, 1], ["O1 site", "O2 site"])
    fg.style(b, "", "phosphene - none (µV)", "VAN, descriptive (data seen)")
    b.legend(fontsize=7, frameon=False)
    fg.save(fig, d)
    write_readme(d, root, "Q38 phosphene VAN topography and a wider response-hand control", "§9",
                 ["derivatives/sv_p3_clean/primary (included recordings)",
                  "results/phase3/scores_primary.tsv"], {"Tests": t},
                 ("H1 survives the hand covariate: the effect is not carried by the lateralized "
                  "hand signal on any channel pair (P§9 rule)." if prim["p"] < 0.05 else
                  "H1 does not survive the hand covariate: the claim narrows to 'not separable "
                  "from the response' (P§9 rule)."),
                 ["Hand fixed to the label by design: bounded, not removed.",
                  "Residualization is linear (as A1): a strongly nonlinear dependence of "
                  "P(seen) on the hand index would not be fully removed.",
                  "A null after (a) is ambiguous: interpolating all lateral channels removes "
                  "posterior signal too.", "VAN rows are descriptive: data seen in the "
                  "exploration of 2026-11-09."])
    return d
