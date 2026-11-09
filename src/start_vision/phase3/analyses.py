"""Phase 3 step 11: pre-registered analyses and controls (docs/PHASE3_PREREG.md, §6, §8, §9).

Input: the per-trial table of ``score.score_all`` (+ the per-recording code check). The subject
is the unit; chance by permutation. Not covered here (separate runs, later): S2 site test needs
both sites; S5 needs the paperlike/gonogo score tables (same functions); S6-S8, R3, R4b.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from start_vision.decoding import stats

GONOGO_USED = {f"sub-{s}_site-O1" for s in ("03", "04", "05", "06", "07", "08", "09", "10", "11")}
MIN_KEPT, MIN_PER_CLASS = 40, 30
THETA = 0.521148143854642      # contract threshold (max_balanced_accuracy)


# ---------- §6 code-direction check (per recording, from the raw .set) ----------

def code_check(set_path: Path, present: str = "S100", absent: str = "S200",
               n_perm: int = 5000, seed: int = 0) -> dict[str, float]:
    """C3-C4, 0.1-10 Hz, pulse-locked, baseline -0.2..0 s, mean 0.3-1.2 s, present - absent;
    trial-label permutation p (two-sided). Negative = the inferred mapping."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="error")
    raw.pick(["C3", "C4"]).resample(500, verbose="error").filter(0.1, 10, verbose="error")
    ev, eid = mne.events_from_annotations(raw, verbose="error")
    ep = mne.Epochs(raw, ev, {present: eid[present], absent: eid[absent]}, tmin=-0.2, tmax=1.5,
                    baseline=(-0.2, 0), preload=True, verbose="error")
    t = ep.times
    m = (t >= 0.3) & (t < 1.2)
    x = {k: (lambda d: (d[:, 0] - d[:, 1])[:, m].mean(1) * 1e6)(ep[k].get_data())
         for k in (present, absent)}
    a = np.r_[x[present], x[absent]]
    n = len(x[present])
    obs = x[present].mean() - x[absent].mean()
    rng = np.random.default_rng(seed)
    null = np.array([(lambda q: q[:n].mean() - q[n:].mean())(rng.permutation(a))
                     for _ in range(n_perm)])
    return {"code_diff_uv": float(obs),
            "code_p": float((1 + (np.abs(null) >= abs(obs)).sum()) / (1 + n_perm))}


# ---------- §6 inclusion ----------

def inclusion(scores: pd.DataFrame, codes: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for rid, g in scores.groupby("recording", sort=True):
        n1, n0 = int(g["seen"].sum()), int((1 - g["seen"]).sum())
        r = {"recording": rid, "subject": g["subject"].iloc[0], "site": g["site"].iloc[0],
             "kept": len(g), "n_seen": n1, "n_unseen": n0, "seen_share": n1 / len(g)}
        reasons = []
        if len(g) < MIN_KEPT:
            reasons.append(f"kept < {MIN_KEPT}")
        if min(n1, n0) < MIN_PER_CLASS:
            reasons.append(f"class < {MIN_PER_CLASS}")
        if codes is not None and rid in set(codes["recording"]):
            c = codes.set_index("recording").loc[rid]
            r |= {"code_diff_uv": c["code_diff_uv"], "code_p": c["code_p"]}
            if c["code_diff_uv"] > 0 and c["code_p"] < 0.05:
                reasons.append("code check opposite (p < 0.05)")
        r["included"] = not reasons
        r["reason"] = "; ".join(reasons) or "ok"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------- helpers ----------

def rec_auc(g: pd.DataFrame, col: str) -> float:
    y = g["seen"].to_numpy()
    return float(roc_auc_score(y, g[col].to_numpy())) if 0 < y.sum() < len(y) else np.nan


def rec_auc_perm_p(g: pd.DataFrame, col: str, n_perm: int = 2000, seed: int = 0) -> float:
    y, s = g["seen"].to_numpy(), g[col].to_numpy()
    obs = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(rng.permutation(y), s) for _ in range(n_perm)])
    return float((1 + (null >= obs).sum()) / (1 + n_perm))


E1_HALF_STEP_UV = 16.0
"""E1 follow-up: half the 32 uV HEOG step rule (PHASE3_PREREG §9)."""


def subject_values(per_rec: pd.DataFrame, col: str) -> pd.Series:
    """Subject value = mean over that subject's included recordings (D6)."""
    return per_rec.groupby("subject")[col].mean()


def group_one_sided(sub_auc: Iterable[float], seed: int = 0) -> dict[str, float]:
    a = np.asarray(list(sub_auc), float)
    one = stats.window_test(a, seed=seed)
    two = stats._signflip_p(a[np.isfinite(a)] - 0.5, 10_000, seed, tail=0)
    return one | {"p_two_sided": two, "n_above": int((a > 0.5).sum())}


def binary_metrics(g: pd.DataFrame, theta: float) -> dict[str, float]:
    y = g["seen"].to_numpy().astype(bool)
    pred = g["p_seen"].to_numpy() >= theta
    hit = pred[y].mean() if y.any() else np.nan
    fa = pred[~y].mean() if (~y).any() else np.nan
    return {"hit": float(hit), "fa": float(fa), "bacc": float((hit + 1 - fa) / 2)}


def residualized(g: pd.DataFrame, col: str, on: str) -> np.ndarray:
    x, s = g[on].to_numpy(), g[col].to_numpy()
    b = np.polyfit(x, s, 1)
    return np.asarray(s - np.polyval(b, x))


# ---------- the run ----------

def per_recording(scores: pd.DataFrame, n_perm: int = 2000) -> pd.DataFrame:
    rows = []
    for rid, g in scores.groupby("recording", sort=True):
        r: dict[str, object] = {"recording": rid, "subject": g["subject"].iloc[0],
                                "site": g["site"].iloc[0]}
        r["auc"] = rec_auc(g, "p_seen")
        r["auc_p"] = rec_auc_perm_p(g, "p_seen", n_perm) if n_perm else np.nan
        r["auc_no_motor"] = rec_auc(g, "p_seen_no_motor") if "p_seen_no_motor" in g else np.nan
        r["auc_early"] = rec_auc(g, "p_seen_early") if "p_seen_early" in g else np.nan
        for c in ("c3c4", "heog", "veog", "alpha_log10"):
            if c in g:
                r[f"auc_{c}"] = rec_auc(g, c)
        if "c3c4" in g:
            rhos = [spearmanr(g.loc[g.seen == k, "p_seen"], g.loc[g.seen == k, "c3c4"])[0]
                    for k in (0, 1)]
            r["rho_c3c4"] = float(np.nanmean(rhos))
        if "alpha_log10" in g:
            gg = g.assign(p_res=residualized(g, "p_seen", "alpha_log10"))
            r["auc_alpha_residualized"] = rec_auc(gg, "p_res")
        if "heog_step_uv" in g:
            # E1 follow-up (PHASE3_PREREG §9): H1 on trials with HEOG step < half the rule
            r["auc_heog_half"] = rec_auc(g[g["heog_step_uv"] < E1_HALF_STEP_UV], "p_seen")
        r |= {f"{k}_theta": v for k, v in binary_metrics(g, THETA).items()}
        r |= {f"{k}_median": v for k, v in
              binary_metrics(g, float(np.median(g["p_seen"]))).items()}
        r["mean_p_seen_class1"] = float(g.loc[g.seen == 1, "p_seen"].mean())
        r["mean_p_seen_class0"] = float(g.loc[g.seen == 0, "p_seen"].mean())
        r["seen_share"] = float(g["seen"].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def tests(per_rec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T2 (primary + secondary) and T3 (controls with flags) on included recordings."""
    h1 = group_one_sided(subject_values(per_rec, "auc"))
    t2 = [{"test": "H1 primary", **h1}]
    sec: list[dict[str, Any]] = []
    untouched = per_rec[~per_rec["recording"].isin(GONOGO_USED)]
    if len(untouched):
        sec.append({"test": "S1 untouched subset",
                    **group_one_sided(subject_values(untouched, "auc"))})
    for rule in ("theta", "median"):
        sec.append({"test": f"S3 balanced accuracy ({rule})",
                    **stats.window_test(subject_values(per_rec, f"bacc_{rule}").to_numpy())})
    band = per_rec[(per_rec["seen_share"] >= 0.3) & (per_rec["seen_share"] <= 0.7)]
    if len(band):
        sec.append({"test": "S9 seen share 30-70 %",
                    **group_one_sided(subject_values(band, "auc"))})
    if sec:
        adj = stats.holm([s["p"] for s in sec])
        for s, a in zip(sec, adj, strict=True):
            s["p_holm"] = float(a)
    t2 += sec
    t3: list[dict[str, Any]] = []
    nm = group_one_sided(subject_values(per_rec, "auc_no_motor"))
    drop = (h1["mean"] - nm["mean"]) > 0.5 * (h1["mean"] - 0.5)
    t3.append({"control": "R1 motor channels removed", **nm,
               "flag": bool(nm["p"] >= 0.05 or drop)})
    if "rho_c3c4" in per_rec:
        rho = per_rec.groupby("subject")["rho_c3c4"].mean().to_numpy()
        p = stats._signflip_p(rho, 10_000, 0, tail=1)
        t3.append({"control": "R2 P(seen) vs C3-C4 within class", "n": len(rho),
                   "mean": float(rho.mean()), "p": p, "flag": bool(rho.mean() > 0 and p < 0.05)})
    t3.append({"control": "R4a early bins only (< 0.30 s)",
               **group_one_sided(subject_values(per_rec, "auc_early")), "flag": None})
    for c, name in (("c3c4", "R5 C3-C4 alone"), ("heog", "E1 HEOG alone"),
                    ("veog", "E1 VEOG alone"), ("alpha_log10", "A1 prestim alpha alone")):
        col = f"auc_{c}"
        if col in per_rec:
            a = subject_values(per_rec, col).to_numpy()
            p2 = stats._signflip_p(a - 0.5, 10_000, 0, tail=0)
            t3.append({"control": name, "n": len(a), "mean": float(a.mean()), "p": p2,
                       "flag": bool(p2 < 0.05) if c in ("heog", "veog") else None})
    if "auc_heog_half" in per_rec:
        flagged = any(r["flag"] for r in t3 if str(r["control"]).startswith("E1 ") and r["flag"])
        t3.append({"control": f"E1 follow-up: H1 on trials with HEOG step < {E1_HALF_STEP_UV:g} uV",
                   **group_one_sided(subject_values(per_rec, "auc_heog_half")),
                   "flag": None, "required": flagged})
    if "auc_alpha_residualized" in per_rec:
        t3.append({"control": "A1 H1 with P(seen) residualized on alpha",
                   **group_one_sided(subject_values(per_rec, "auc_alpha_residualized")),
                   "flag": None})
    return pd.DataFrame(t2), pd.DataFrame(t3)
