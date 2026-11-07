"""Cross-dataset test of the frozen detector on Eklund & Wiens 2018 (A12 issue #1).

User decision 9 (2026-11-07): primary = the frozen ``sv-perception-tms-robust-v1.0.0`` applied
as is, no retraining; large-Gabor sessions (N = 24) primary, small (N = 15) replication; labels =
the authors' contrast, aware correct vs unaware correct (critical trials); rating-only (2-3 vs 1,
any location) as sensitivity. One AUC per session; chance by sign-flip permutation.

Input path = the detector's training path: eegunify ``stimraw`` epochs (ICA, average reference
over the 64 EEG channels), ``data.prepare`` (40 Hz IIR low-pass, decimate to 128 Hz, crop), then
``Detector.decision_scores`` (prestimulus baseline + per-session z-score over all kept trials,
label-free). The 64 Eklund channels are exactly the contract's channels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from start_vision.decoding import stats
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import prepare
from start_vision.decoding.detector import Detector

DETECTOR = "detectors/sv-perception-tms-robust-v1.0.0"


def epochs_file(runs: Path, sub: str, gabor: str) -> Path:
    return (runs / gabor / "derivatives" / "eegunify-epochs" / f"sub-{sub}" / "eeg"
            / f"sub-{sub}_task-{gabor}_desc-stimraw_epo.fif")


def load_session(runs: Path, staged: Path, sub: str, gabor: str
                 ) -> tuple[mne.BaseEpochs, pd.DataFrame]:
    """Epochs with their staged-event rows (``epochs.selection`` indexes the 580 trials)."""
    ep = mne.read_epochs(epochs_file(runs, sub, gabor), verbose="error")
    ev = pd.read_csv(staged / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{gabor}_events.tsv",
                     sep="\t")
    ev = ev[ev["trial_type"].str.startswith("stimulus/")].reset_index(drop=True)
    meta = ev.iloc[ep.selection].reset_index(drop=True)
    names = {v: k for k, v in ep.event_id.items()}
    got = [names[c] for c in ep.events[:, 2]]
    if got != list(meta["trial_type"]):
        raise ValueError(f"sub-{sub} {gabor}: epochs and staged events disagree")
    return ep, meta


def score_session(det: Detector, ep: mne.BaseEpochs, meta: pd.DataFrame,
                  cfg: DecodingConfig) -> pd.DataFrame:
    """P(seen) and raw decision score per kept trial (excluded trials dropped first)."""
    keep = (meta["exclusion"].fillna("") == "").to_numpy()
    e = prepare(ep[np.flatnonzero(keep)], cfg).pick(det.contract["channels"])
    x = (e.get_data() * 1e6).astype(np.float32)
    det.check(e.ch_names, e.info["sfreq"], e.times)
    s = det.decision_scores(x, e.times)
    out = meta[keep].reset_index(drop=True).copy()
    out["score"] = s
    out["p_seen"] = det.calibrator.predict_proba(s[:, None])[:, 1]
    return out


def session_aucs(t: pd.DataFrame) -> dict[str, Any]:
    crit = t["condition"].between(1, 4)
    prim = t[t["cls"].isin(["aware", "unaware"])]
    sens = t[crit]
    catch = t[t["cls"] == "catch"]
    row: dict[str, Any] = {
        "n_aware": int((prim["cls"] == "aware").sum()),
        "n_unaware": int((prim["cls"] == "unaware").sum()),
        "auc_primary": float(roc_auc_score(prim["cls"] == "aware", prim["score"])),
        "auc_rating_only": float(roc_auc_score(sens["rating"] > 1, sens["score"])),
        "n_catch_fa": int((catch["rating"] > 1).sum()),
    }
    if 0 < row["n_catch_fa"] < len(catch):
        row["auc_catch_fa_vs_cr"] = float(roc_auc_score(catch["rating"] > 1, catch["score"]))
    return row


def group_test(aucs: np.ndarray, cfg: DecodingConfig) -> dict[str, float]:
    """Sign-flip permutation of AUC - 0.5 over sessions; bootstrap CI; d_z."""
    d = aucs - 0.5
    rng = np.random.default_rng(cfg.seed)
    flips = rng.choice([-1.0, 1.0], size=(cfg.n_bootstrap, len(d)))
    null = (flips * d).mean(1)
    p = float((np.sum(null >= d.mean()) + 1) / (len(null) + 1))
    lo, hi = stats.bootstrap_ci(aucs, cfg.n_bootstrap, cfg.seed)
    return {"n": len(aucs), "mean_auc": float(aucs.mean()), "ci_low": lo, "ci_high": hi,
            "d_z": float(d.mean() / d.std(ddof=1)), "p_one_sided": p,
            "n_above_chance": int((aucs > 0.5).sum())}


def run(root: Path, cfg: DecodingConfig, out: Path) -> pd.DataFrame:
    """Score every included session; write per-session AUCs, trial scores and group tests."""
    import json

    det = Detector.load(root / DETECTOR)
    sessions = pd.read_csv(root / "configs" / "eklund2018_sessions.tsv", sep="\t")
    sessions = sessions[sessions["authors_included"]]
    runs, staged = root / "runs" / "eklund2018", root / "sourcedata" / "staged" / "eklund2018"
    out.mkdir(parents=True, exist_ok=True)
    rows, trials, failed = [], [], []
    for rec in sessions.to_dict("records"):
        fp, gabor, sess = int(rec["fp"]), str(rec["gabor"]), int(rec["session"])
        sub = f"{fp:02d}"
        try:
            ep, meta = load_session(runs, staged, sub, gabor)
            t = score_session(det, ep, meta, cfg)
        except Exception as err:  # one bad session does not stop the test; it is reported
            failed.append({"fp": fp, "gabor": gabor, "error": repr(err)})
            continue
        rows.append({"fp": fp, "session": sess, "gabor": gabor} | session_aucs(t))
        trials.append(t.assign(fp=fp))
        print(f"eklund {gabor} sub-{sub}: AUC {rows[-1]['auc_primary']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "session_aucs.tsv", sep="\t", index=False)
    pd.concat(trials).to_csv(out / "trial_scores.tsv.gz", sep="\t", index=False)
    summary: dict[str, Any] = {"detector": DETECTOR, "failed": failed}
    for g in ("large", "small"):
        r = res[res["gabor"] == g]
        summary[g] = {k: group_test(r[k].dropna().to_numpy(), cfg)
                      for k in ("auc_primary", "auc_rating_only")}
        fa = r["auc_catch_fa_vs_cr"].dropna() if "auc_catch_fa_vs_cr" in r else pd.Series([])
        summary[g]["catch_fa_vs_cr"] = {"n_sessions": len(fa),
                                        "mean_auc": float(fa.mean()) if len(fa) else None}
    all_t = pd.concat(trials)
    summary["p_seen_by_class"] = {g: all_t[all_t["gabor"] == g].groupby("cls")["p_seen"]
                                  .mean().round(3).to_dict() for g in ("large", "small")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return res


def contralateral_van(root: Path, out: Path, win_ms: tuple[int, int] = (180, 280)
                      ) -> dict[str, Any]:
    """Eklund's VAN measure on this pipeline: contra(aware) - contra(unaware), O1 for right- and
    O2 for left-field Gabors (analyze_VAN.m; Phase 1 decision 4), 180-280 ms, per session."""
    import json

    from scipy import stats as sps

    from start_vision.analysis import lateral

    sessions = pd.read_csv(root / "configs" / "eklund2018_sessions.tsv", sep="\t")
    rows = []
    for rec in sessions[sessions["authors_included"]].to_dict("records"):
        sub, gabor = f"{int(rec['fp']):02d}", str(rec["gabor"])
        f = (root / "runs" / "eklund2018" / gabor / "derivatives" / "eegunify-erp" / f"sub-{sub}"
             / "eeg" / f"sub-{sub}_task-{gabor}_desc-stim_ave.fif")
        if not f.exists():
            continue
        evs = {e.comment: e for e in mne.read_evokeds(f, verbose="error")}
        d = lateral.contra_difference({"seen_contra": evs["aware_contra"],
                                       "unseen_contra": evs["unaware_contra"]})
        rows.append({"fp": int(rec["fp"]), "gabor": gabor,
                     "van_contra_uv": lateral.window_mean(d, win=win_ms)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "van_contralateral.tsv", sep="\t", index=False)
    res: dict[str, Any] = {}
    for g in ("large", "small"):
        v = df.loc[df["gabor"] == g, "van_contra_uv"].to_numpy()
        t = sps.ttest_1samp(v, 0.0)
        res[g] = {"n": len(v), "mean_uv": float(v.mean()), "sd_uv": float(v.std(ddof=1)),
                  "t": float(t.statistic), "p": float(t.pvalue),
                  "d_z": float(v.mean() / v.std(ddof=1))}
    (out / "van_contralateral.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def retrained_transfer(root: Path, cfg: DecodingConfig, out: Path) -> dict[str, Any]:
    """Secondary test (decision 9): a sliding decoder trained on all included ds006171
    subjects (stimulus-present trials, 64-channel subset) and tested on each Eklund session,
    AUC(t). Group: cluster test over time and window tests (VAN, LP, post), per Gabor size."""
    import json

    from start_vision.decoding import across, channels, stats
    from start_vision.decoding.analyses import SubjectData
    from start_vision.decoding.sources import Eklund2018Adapter
    from start_vision.decoding.stages import Project
    from start_vision.decoding.step0 import included_tasks

    proj = Project.default(root)
    table = pd.read_csv(root / "configs" / "channels" / "ds006171_to_biosemi64.tsv", sep="\t")
    train = []
    for s, tasks in sorted(included_tasks(root).items()):
        train.append(channels.subset(SubjectData(proj.runs, proj.staged, s, tasks, cfg).main,
                                     table))
    ek = Eklund2018Adapter(cfg)
    names = ek.subjects(root)
    test = [ek.load(n, root) for n in names]
    h = channels.harmonize(train + test)
    tr, te = h[:len(train)], h[len(train):]
    df = across.transfer(tr, te, cfg)
    times = tr[0].times
    auc = np.vstack([np.asarray(a, dtype=float) for a in df["auc"]])
    gab = np.array([n.split("_")[1] for n in names])
    np.savez(out / "retrained_transfer_auc.npz", auc=auc, times=times, sessions=np.array(names))
    windows = {"VAN 180-280": (0.18, 0.28), "LP 350-550": (0.35, 0.55),
               "post 200-600": (0.2, 0.6), "prestim -500-0": (-0.5, 0.0)}
    res: dict[str, Any] = {"n_train_subjects": len(train),
                           "n_train_trials": int(sum(len(t) for t in tr)),
                           "channels": len(tr[0].ch_names)}
    for g in ("large", "small"):
        a = auc[gab == g]
        win = {}
        for k, (lo, hi) in windows.items():
            m = (times >= lo - 1e-9) & (times <= hi + 1e-9)
            win[k] = stats.window_test(a[:, m].mean(1), seed=cfg.seed)
        cl = stats.cluster_auc(a, times, seed=cfg.seed)
        res[g] = {"n": len(a), "windows": win,
                  "clusters": cl[cl["p"] < 0.05].to_dict("records"),
                  "peak_auc": float(a.mean(0).max()),
                  "peak_time_s": float(times[int(a.mean(0).argmax())])}
    (out / "retrained_transfer.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res
