"""Phase 3 step 12, the remaining pre-registered analyses (PHASE3_PREREG §8-§9).

S2 site test, S4 calibration shift, S6 ERP signatures, S8 within-recording decoding, R3 pattern
lateralization. All on the included recordings of one pipeline (``inclusion_<pipeline>.tsv``);
subject = unit (mean over included sites); nothing is fitted outside a training fold.
S7 and R4b need the ds006171 epochs (``runs/``) and are not computed here.

Windows and sites: S6 VAN 180-280 ms and LP 350-550 ms (``docs/PHASE1_REFERENCES.md``,
Eklund & Wiens 2018 ``analyze_VAN.m``); contralateral-to-phosphene channels = the stimulated
hemisphere (O1 site -> O1, PO7; O2 site -> O2, PO8) and LP at Pz (PHASE3_PREREG S6).
Baseline -100..0 ms for ERPs (Eklund's baseline, Phase 1). S8 uses the Phase 2 classifier
(StandardScaler + L2 logistic regression, C = 1, balanced) and the detector's baseline window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import wilcoxon

from start_vision.decoding import stats

SEED = 0
VAN_WIN = (0.18, 0.28)
LP_WIN = (0.35, 0.55)
ERP_BASELINE = (-0.1, 0.0)
ERP_CLUSTER_SPAN = (-0.1, 0.8)
CONTRA = {"O1": ["O1", "PO7"], "O2": ["O2", "PO8"]}
LP_SITES = ["Pz"]
S8_SPAN = (-0.5, 0.99)
S8_BASELINE = (-0.2, -0.01)   # contract baseline_window_s
R3_LAT_WIN = (0.3, 0.99)      # prereg 0.3-1.2 s; epochs end at 1.0 s (deviation)
R3_N_NULL = 1000


# ---------- S2 ----------

def s2_site(per_rec: pd.DataFrame, col: str = "auc") -> dict[str, Any]:
    w = per_rec.pivot_table(index="subject", columns="site", values=col).dropna()
    d = (w["O1"] - w["O2"]).to_numpy()
    res = wilcoxon(w["O1"], w["O2"], alternative="two-sided")
    return {"test": "S2 site O1 vs O2 (Wilcoxon)", "n": len(w), "mean_O1": float(w["O1"].mean()),
            "mean_O2": float(w["O2"].mean()), "mean_diff": float(d.mean()),
            "statistic": float(res.statistic), "p_two_sided": float(res.pvalue)}


# ---------- S4 ----------

def s4_calibration(scores: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    """Per P(seen) bin of the training reliability table: GIN n and observed phosphene rate."""
    edges = np.r_[calibration["bin_low"].to_numpy(), calibration["bin_high"].iloc[-1]]
    b = np.clip(np.digitize(scores["p_seen"], edges[1:-1]), 0, len(calibration) - 1)
    rows = []
    for k, r in calibration.reset_index(drop=True).iterrows():
        m = b == k
        rows.append({"bin_low": r["bin_low"], "bin_high": r["bin_high"],
                     "train_mean_p": r["mean_p"], "train_observed": r["observed"],
                     "tms_n": int(m.sum()),
                     "tms_mean_p": float(scores.loc[m, "p_seen"].mean()) if m.any() else np.nan,
                     "tms_observed": float(scores.loc[m, "seen"].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def s4_class_means(scores: pd.DataFrame) -> pd.DataFrame:
    by = scores.pivot_table(index="subject", columns="seen", values="p_seen", aggfunc="mean")
    out = {"mean_p_seen_phosphene": by[1].mean(), "mean_p_seen_none": by[0].mean(),
           "mean_p_seen_all": scores.groupby("subject")["p_seen"].mean().mean(),
           "observed_phosphene_rate": scores.groupby("subject")["seen"].mean().mean()}
    return pd.DataFrame([{"measure": k, "value": float(v)} for k, v in out.items()])


# ---------- per-recording data ----------

def _win(t: np.ndarray, w: tuple[float, float]) -> np.ndarray:
    return (t >= w[0] - 1e-9) & (t <= w[1] + 1e-9)


def erp_roi(ep: mne.BaseEpochs, site: str) -> dict[str, np.ndarray]:
    """Seen and unseen mean waves (µV) at the contralateral ROI and at Pz, baseline -100..0 ms."""
    t = ep.times
    x = ep.get_data(picks="eeg") * 1e6
    x = x - x[..., _win(t, ERP_BASELINE)].mean(-1, keepdims=True)
    ch = ep.ch_names
    y = ep.metadata["seen"].to_numpy().astype(bool)
    out: dict[str, np.ndarray] = {"times": t}
    for roi, names in (("contra", CONTRA[site]), ("lp", LP_SITES)):
        idx = [ch.index(c) for c in names]
        w = x[:, idx].mean(1)
        out[f"{roi}_seen"], out[f"{roi}_unseen"] = w[y].mean(0), w[~y].mean(0)
    return out


def s8_sliding(ep: mne.BaseEpochs, channels: list[str], n_folds: int = 5,
               seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Within-recording time-resolved AUC (stratified 5-fold CV, Phase 2 classifier)."""
    from mne.decoding import SlidingEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    e = ep.copy().pick(channels)
    t = e.times
    m = _win(t, S8_SPAN)
    x = e.get_data() * 1e6
    x = x - x[..., _win(t, S8_BASELINE)].mean(-1, keepdims=True)
    x = x[..., m]
    y = e.metadata["seen"].to_numpy().astype(int)
    aucs = []
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(x, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear"))
        est = SlidingEstimator(clf, scoring="roc_auc", n_jobs=1, verbose="error")
        est.fit(x[tr], y[tr])
        aucs.append(est.score(x[te], y[te]))
    return t[m], np.mean(aucs, axis=0)


def lateralization_map(ep: mne.BaseEpochs, channels: list[str]) -> np.ndarray:
    """R3(b): S100 - S200 mean per channel over 0.3-1.0 s (µV), baseline -100..0 ms."""
    e = ep.copy().pick(channels)
    t = e.times
    x = e.get_data() * 1e6
    x = x - x[..., _win(t, ERP_BASELINE)].mean(-1, keepdims=True)
    y = e.metadata["seen"].to_numpy().astype(bool)
    w = x[..., _win(t, R3_LAT_WIN)].mean(-1)
    return np.asarray(w[y].mean(0) - w[~y].mean(0))


# ---------- group statistics ----------

def subject_mean(values: dict[str, np.ndarray], rec_subject: dict[str, str]) -> np.ndarray:
    subs = sorted(set(rec_subject.values()))
    return np.array([np.mean([v for r, v in values.items() if rec_subject[r] == s], axis=0)
                     for s in subs])


def s6_clusters(t: np.ndarray, diffs: dict[str, np.ndarray], n_perm: int = 10_000,
                seed: int = SEED) -> pd.DataFrame:
    """Cluster permutation (two-sided) over -100..800 ms on subject seen - unseen waves."""
    span = _win(t, ERP_CLUSTER_SPAN)
    ts = t[span]
    clus = []
    for name, d in diffs.items():
        tobs, cl, pv, _ = permutation_cluster_1samp_test(
            d[:, span], n_permutations=n_perm, tail=0, rng=seed, out_type="mask", verbose="error")
        for mask, p in zip(cl, pv, strict=True):
            idx = np.flatnonzero(stats._as_mask(mask, tobs.shape))
            clus.append({"roi": name, "p": float(p), "t_start": float(ts[idx[0]]),
                         "t_end": float(ts[idx[-1]]), "sign": float(np.sign(tobs[idx].mean()))})
    return pd.DataFrame(clus, columns=["roi", "p", "t_start", "t_end", "sign"])


def s6_tests(erps: dict[str, dict[str, np.ndarray]], rec_subject: dict[str, str],
             n_perm: int = 10_000, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Window means (seen - unseen) per subject: sign flip two-sided + expected-direction one-sided;
    cluster permutation over -100..800 ms (two-sided)."""
    t = next(iter(erps.values()))["times"]
    rows = []
    diffs = {}
    for roi, win, name in (("contra", VAN_WIN, "VAN contralateral"), ("lp", LP_WIN, "LP Pz")):
        diff = {r: e[f"{roi}_seen"] - e[f"{roi}_unseen"] for r, e in erps.items()}
        d = subject_mean(diff, rec_subject)
        diffs[name] = d
        v = d[:, _win(t, win)].mean(1)
        sign = -1.0 if roi == "contra" else 1.0
        p2 = stats._signflip_p(v, n_perm, seed, tail=0)
        p1 = stats._signflip_p(sign * v, n_perm, seed, tail=1)
        lo, hi = stats.bootstrap_ci(v, n_perm, seed)
        rows.append({"test": f"S6 {name} {int(win[0]*1e3)}-{int(win[1]*1e3)} ms", "n": len(v),
                     "mean_uv": float(v.mean()), "ci_low": lo, "ci_high": hi,
                     "d_z": float(v.mean() / v.std(ddof=1)), "p_two_sided": p2,
                     "p_expected_direction": p1, "n_expected_direction": int((sign * v > 0).sum())})
    return pd.DataFrame(rows), s6_clusters(t, diffs, n_perm, seed)


def s8_tests(curves: dict[str, np.ndarray], times: np.ndarray, rec_subject: dict[str, str],
             n_perm: int = 10_000, seed: int = SEED) -> tuple[np.ndarray, pd.DataFrame]:
    a = subject_mean(curves, rec_subject)
    cl = stats.cluster_auc(a, times, n_permutations=n_perm, seed=seed)
    return a, cl


def r3_patterns(patterns: np.ndarray, channels: list[str], lat_map: np.ndarray,
                bin_starts: np.ndarray, n_null: int = R3_N_NULL, seed: int = SEED
                ) -> pd.DataFrame:
    """(a) (C3 - C4) / mean |posterior| of each 50 ms pattern bin; (b) spatial r with the TMS
    lateralization map, flagged above the 95th percentile of |r| over channel-permuted maps."""
    rng = np.random.default_rng(seed)
    post = [i for i, c in enumerate(channels) if c.startswith(("O", "P", "I"))]
    c3, c4 = channels.index("C3"), channels.index("C4")
    rows = []
    for b in range(patterns.shape[1]):
        p = patterns[:, b]
        r = float(np.corrcoef(p, lat_map)[0, 1])
        null = np.array([abs(np.corrcoef(p, rng.permutation(lat_map))[0, 1])
                         for _ in range(n_null)])
        thr = float(np.percentile(null, 95))
        rows.append({"bin_start_s": float(bin_starts[b]),
                     "c3_minus_c4_rel": float((p[c3] - p[c4]) / np.mean(np.abs(p[post]))),
                     "spatial_r": r, "null95_abs_r": thr, "flag": bool(abs(r) > thr)})
    return pd.DataFrame(rows)


# ---------- the run ----------

def run(root: Path, pipeline: str, det_dir: Path) -> dict[str, pd.DataFrame]:
    from start_vision.decoding.detector import Detector

    res = root / "results" / "phase3"
    inc = pd.read_csv(res / f"inclusion_{pipeline}.tsv", sep="\t", dtype={"subject": str})
    inc = inc[inc["included"]]
    per_rec = pd.read_csv(res / f"per_recording_{pipeline}.tsv", sep="\t", dtype={"subject": str})
    per_rec = per_rec[per_rec["recording"].isin(inc["recording"])]
    scores = pd.read_csv(res / f"scores_{pipeline}.tsv", sep="\t", dtype={"subject": str})
    scores = scores[scores["recording"].isin(inc["recording"])]
    det = Detector.load(det_dir)
    ch = list(det.contract["channels"])
    calib = pd.read_csv(det_dir / "calibration.tsv", sep="\t")
    rec_subject = dict(zip(inc["recording"], inc["subject"], strict=True))

    erps, curves, maps = {}, {}, {}
    times8 = np.array([])
    for rid in inc["recording"]:
        ep = mne.read_epochs(root / "derivatives" / "sv_p3_clean" / pipeline
                             / f"{rid}_desc-detector_epo.fif", verbose="error")
        site = rid.split("-")[-1]
        erps[rid] = erp_roi(ep, site)
        times8, curves[rid] = s8_sliding(ep, ch)
        maps[rid] = lateralization_map(ep, ch)
        print(f"{rid} done", flush=True)

    s6, s6c = s6_tests(erps, rec_subject)
    a8, s8c = s8_tests(curves, times8, rec_subject)
    lat = subject_mean(maps, rec_subject).mean(0)
    pat = np.load(det_dir / "patterns.npz")["patterns"]
    lo, step = det.contract["feature_window_s"][0], det.contract["feature_bin_s"]
    r3 = r3_patterns(pat, ch, lat, lo + step * np.arange(pat.shape[1]))
    s2 = pd.DataFrame([s2_site(per_rec)])
    out = {"s2": s2, "s4_bins": s4_calibration(scores, calib), "s4_means": s4_class_means(scores),
           "s6": s6, "s6_clusters": s6c, "s8_clusters": s8c, "r3": r3}
    for k, v in out.items():
        v.to_csv(res / f"{k}_{pipeline}.tsv", sep="\t", index=False, float_format="%.5f")
    arrays: dict[str, Any] = {"erp_times": next(iter(erps.values()))["times"], "s8_times": times8,
                              "s8_auc": a8, "lat_map": lat, "channels": np.array(ch)}
    for roi in ("contra", "lp"):
        for k in ("seen", "unseen"):
            arrays[f"{roi}_{k}"] = subject_mean({r: e[f"{roi}_{k}"] for r, e in erps.items()},
                                                rec_subject)
    np.savez_compressed(res / f"secondary_curves_{pipeline}.npz", **arrays)
    return out
