"""Questions_to_solve, Q03 follow-up (step 1): where does the 'unseen but correct' signal live?

Run from Start_Vision/: ~/miniconda3/envs/eegunify/bin/python -I scripts/questions_q03_where_eklund.py
Output: results/questions_to_solve/q03_where_eklund/

Per Eklund session (authors-included sessions, critical trials, duration exclusion applied), with the
detector's own input path (eegunify stimraw epochs -> decoding.data.prepare: 40 Hz low-pass, 128 Hz,
crop; per-trial prestimulus baseline), within-session decoding, stratified 5-fold CV x 2 repeats,
L2 logistic regression (C = 1, balanced) per time point, ROC AUC on held-out folds:
  A awareness   : aware correct vs unaware correct
  B q03         : unaware correct vs unaware wrong localization   (the Q03 signal)
  C_unaware     : stimulus hemifield (left vs right) on all unaware trials (stimulus info without awareness)
  C_aware       : stimulus hemifield on all aware trials (reference)
Window decoding (channel x window mean, same CV) for windows early 50-150, VAN 180-280, LP 350-550 ms
and channel groups posterior / central / frontal (64-ch 10-20 names, groups by name prefix).
Group: cluster sign-flip test of AUC > 0.5 over sessions (stats.cluster_auc), window tests
(stats.window_test, one-sided). Large Gabor primary, small replication. All settings fixed here before
any result was seen. Nothing is fitted outside a training fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from mne.decoding import SlidingEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from start_vision.decoding import config as dconfig
from start_vision.decoding import stats
from start_vision.decoding.data import prepare
from start_vision.decoding.eklund_test import load_session
from start_vision.decoding.features import baseline

ROOT = Path.cwd()
OUT = ROOT / "results" / "questions_to_solve" / "q03_where_eklund"
SEED = 20261109
MIN_PER_CLASS = 15
WINDOWS = {"early_50_150": (0.05, 0.15), "VAN_180_280": (0.18, 0.28), "LP_350_550": (0.35, 0.55)}
GROUPS = {
    "posterior": ("O", "PO", "P", "I"),
    "central": ("C", "CP", "FC", "FT", "T", "TP"),
    "frontal": ("Fp", "AF", "F"),
}


def group_of(ch: str) -> str | None:
    best = None
    for g, prefixes in GROUPS.items():
        for p in prefixes:
            if (
                ch.startswith(p)
                and (len(ch) == len(p) or not ch[len(p)].isupper())
                and (best is None or len(p) > best[1])
            ):
                best = (g, len(p))
    return best[0] if best else None


def clf() -> object:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear")
    )


def cv_sliding(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    rk = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    out = []
    for tr, te in rk.split(X, y):
        est = SlidingEstimator(clf(), scoring="roc_auc", n_jobs=4, verbose="error")
        est.fit(X[tr], y[tr])
        out.append(est.score(X[te], y[te]))
    return np.mean(out, axis=0)


def cv_window(F: np.ndarray, y: np.ndarray) -> float:
    rk = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    s = []
    for tr, te in rk.split(F, y):
        m = clf().fit(F[tr], y[tr])  # type: ignore[attr-defined]
        s.append(roc_auc_score(y[te], m.decision_function(F[te])))  # type: ignore[attr-defined]
    return float(np.mean(s))


def contrasts(meta: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    left = meta["quadrant"].isin(["UL", "LL"]).to_numpy()
    cls = meta["cls"].to_numpy()
    aw_c, un_c, un_w = cls == "aware", cls == "unaware", cls == "unaware_incorrect"
    aware_any = np.isin(cls, ["aware", "aware_incorrect"])
    unaware_any = un_c | un_w
    return {
        "A_awareness": (aw_c | un_c, aw_c),
        "B_q03": (unaware_any, un_c),
        "C_unaware_hemifield": (unaware_any, left),
        "C_aware_hemifield": (aware_any, left),
    }


def main() -> None:
    cfg = dconfig.load(ROOT / "configs" / "decoding.yaml")
    sessions = pd.read_csv(ROOT / "configs" / "eklund2018_sessions.tsv", sep="\t")
    sessions = sessions[sessions["authors_included"]]
    runs, staged = ROOT / "runs" / "eklund2018", ROOT / "sourcedata" / "staged" / "eklund2018"
    OUT.mkdir(parents=True, exist_ok=True)
    curves: dict[str, dict[str, list[np.ndarray]]] = {}
    rows, failed, times = [], [], None
    for rec in sessions.to_dict("records"):
        fp, gabor = int(rec["fp"]), str(rec["gabor"])
        try:
            ep, meta = load_session(runs, staged, f"{fp:02d}", gabor)
        except Exception as err:
            failed.append({"fp": fp, "gabor": gabor, "error": repr(err)})
            continue
        keep = (meta["exclusion"].fillna("") == "").to_numpy() & meta["condition"].between(
            1, 4
        ).to_numpy()
        e = prepare(ep[np.flatnonzero(keep)], cfg).pick("eeg")
        m = meta[keep].reset_index(drop=True)
        X = baseline((e.get_data() * 1e6).astype(np.float32), e.times, cfg.baseline_s)
        times = e.times
        groups = np.array([group_of(c) for c in e.ch_names], dtype=object)
        for name, (sel, y) in contrasts(m).items():
            ys = y[sel].astype(int)
            n1, n0 = int(ys.sum()), int((1 - ys).sum())
            row = {"fp": fp, "gabor": gabor, "contrast": name, "n_pos": n1, "n_neg": n0}
            if min(n1, n0) < MIN_PER_CLASS:
                rows.append(row | {"skipped": True})
                continue
            Xs = X[sel]
            curves.setdefault(gabor, {}).setdefault(name, []).append(cv_sliding(Xs, ys))
            for w, (a, b) in WINDOWS.items():
                tm = (times >= a) & (times < b)
                for g in ("posterior", "central", "frontal", "all"):
                    chm = np.ones(len(groups), bool) if g == "all" else groups == g
                    F = Xs[:, chm][:, :, tm].mean(-1)
                    row[f"{w}|{g}"] = cv_window(F, ys)
            rows.append(row)
        print(f"sub-{fp:02d} {gabor} done", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_session_windows.tsv", sep="\t", index=False, float_format="%.4f")
    assert times is not None
    np.savez_compressed(
        OUT / "curves.npz",
        times=times,
        **{f"{g}__{c}": np.array(v) for g, d in curves.items() for c, v in d.items()},
    )
    clusters, wins = [], []
    for g, d in curves.items():
        for c, v in d.items():
            cl = stats.cluster_auc(np.array(v), times, n_permutations=10_000, seed=SEED)
            for r in cl.to_dict("records"):
                clusters.append({"gabor": g, "contrast": c, **r})
            sub = per[
                (per.gabor == g)
                & (per.contrast == c)
                & per.get("skipped", pd.Series(False, index=per.index)).ne(True)
            ]
            for col in [k for k in per.columns if "|" in k]:
                wt = stats.window_test(sub[col].to_numpy(float), seed=SEED)
                w, grp = col.split("|")
                wins.append({"gabor": g, "contrast": c, "window": w, "channels": grp, **wt})
    pd.DataFrame(clusters).to_csv(OUT / "clusters.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(wins).to_csv(OUT / "windows.tsv", sep="\t", index=False, float_format="%.4f")
    (OUT / "failed.json").write_text(json.dumps(failed, indent=2) + "\n")
    print(pd.DataFrame(clusters).query("p < 0.05").to_string(index=False))
    print(pd.DataFrame(wins).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
