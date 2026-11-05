"""Across-subject and across-condition decoding (PHASE2 §7 A7-A9, A12; build step 8).

Every subject's data are made label-free comparable first (``normalize``): per-trial baseline
on ``cfg.baseline_s``, then each channel z-scored with the subject's own prestimulus statistics
(``cfg.norm_window_s``, all trials, no labels), so a new subject can be handled the same way.

- A7 ``loso_sliding`` / ``loso_window``: leave one subject out; held-out AUC per subject, and a
  per-trial prediction table for calibration (A11).
- A8 ``cross_task_within`` (3 x 3 per subject, matched training sizes) and ``loso_cross_task``.
- A9 ``cross_condition_auc``: train on one hemifield/orientation, test on the other (both
  directions averaged). The mirrored-montage variant needs ``channels.mirror`` (step 9).
- A12 ``transfer``: train on a list of TrialSets, test on another (harmonized channel sets).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from start_vision.decoding import cv, models
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from start_vision.decoding.features import baseline, prestim_normalize


def normalize(ts: TrialSet, cfg: DecodingConfig) -> TrialSet:
    x = prestim_normalize(baseline(ts.X, ts.times, cfg.baseline_s), ts.times, cfg.norm_window_s)
    return TrialSet(x.astype(np.float32), ts.times, ts.ch_names, ts.sfreq, ts.meta, ts.info)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


# ---------- A7 ----------
def _sliding_fit_score(cfg: DecodingConfig, xtr: np.ndarray, ytr: np.ndarray,
                       xte: np.ndarray, yte: np.ndarray) -> np.ndarray:
    """Per-time-point fit and AUC, time points in threads (no data copies; liblinear
    releases the GIL). Same estimator as ``models.sliding``, without its process pool, which
    copies the training data into every worker (out of memory on 34 subjects, 2026-11-05)."""
    from joblib import Parallel, delayed

    def one(t: int) -> float:
        est = models.time_point(cfg).fit(xtr[:, :, t], ytr)
        return _auc(yte, est.decision_function(xte[:, :, t]))

    return np.asarray(Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
        delayed(one)(t) for t in range(xtr.shape[2])))


def loso_sliding(sets: Sequence[TrialSet], cfg: DecodingConfig, normalized: bool = False
                 ) -> tuple[np.ndarray, list[str]]:
    """Held-out AUC(t) per subject, shape (n_subjects, n_times). ``sets``: one per subject.
    Folds run one at a time (one training copy in memory)."""
    norm = list(sets) if normalized else [normalize(s, cfg) for s in sets]
    subjects = [str(s.meta["subject"].iloc[0]) for s in norm]
    out = []
    for i, test in enumerate(norm):
        xtr = np.concatenate([s.X for j, s in enumerate(norm) if j != i])
        ytr = np.concatenate([s.labels("seen") for j, s in enumerate(norm) if j != i])
        out.append(_sliding_fit_score(cfg, xtr, ytr, test.X, test.labels("seen")))
        del xtr
    return np.asarray(out), subjects


def loso_window(sets: Sequence[TrialSet], cfg: DecodingConfig, window_s: tuple[float, float],
                C: float | None = None, normalized: bool = False
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Window detector under LOSO: (per-subject AUC table, per-trial predictions table).

    Window features are computed per subject first (``WindowFeatures`` is stateless, nothing
    is fitted), so folds concatenate small feature matrices; folds run in threads.
    ``normalized``: the sets are already ``normalize``-d (saves memory when reused).
    """
    from joblib import Parallel, delayed

    norm = list(sets) if normalized else [normalize(s, cfg) for s in sets]
    feat = models.WindowFeatures(norm[0].times, window_s, cfg.feature_bin_s)
    fs = [feat.transform(s.X) for s in norm]
    ys = [s.labels("seen") for s in norm]

    def fold(i: int) -> tuple[dict[str, Any], pd.DataFrame]:
        ftr = np.concatenate([f for j, f in enumerate(fs) if j != i])
        ytr = np.concatenate([y for j, y in enumerate(ys) if j != i])
        pipe = models.window_detector(cfg, norm[0].times, window_s, C)
        pipe.steps[0] = (pipe.steps[0][0], "passthrough")  # features already computed
        s = pipe.fit(ftr, ytr).decision_function(fs[i])
        test = norm[i]
        sub = str(test.meta["subject"].iloc[0])
        row = {"subject": sub, "n": len(s), "n_seen": int(ys[i].sum()), "auc": _auc(ys[i], s)}
        return row, test.meta[["subject", "task", "trial_number", "seen"]].assign(score=s)

    res = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(delayed(fold)(i)
                                                        for i in range(len(norm)))
    return (pd.DataFrame([r for r, _ in res]),
            pd.concat([p for _, p in res], ignore_index=True))


# ---------- A8 ----------
def cross_task_within(ts: TrialSet, cfg: DecodingConfig, tasks: Sequence[str] | None = None
                      ) -> tuple[np.ndarray, list[str], int]:
    """AUC(t) for train task i -> test task j, shape (n_tasks, n_tasks, n_times).

    Diagonal by stratified CV inside the task; every training set is subsampled to the same
    size ``n_train`` (the smallest CV training fold over tasks), so cells are comparable.
    """
    tasks = list(tasks or [t for t in cfg.tasks if (ts.meta["task"] == t).any()])
    x = baseline(ts.X, ts.times, cfg.baseline_s)
    y = ts.labels("seen")
    task = ts.meta["task"].to_numpy()
    k = cfg.n_folds
    n_train = min(int((task == t).sum() * (k - 1) // k) for t in tasks)
    out = np.full((len(tasks), len(tasks), len(ts.times)), np.nan)
    for i, ti in enumerate(tasks):
        idx_i = np.flatnonzero(task == ti)
        reps = cv.within_subject(ts.meta.iloc[idx_i].reset_index(drop=True), k, 1, cfg.seed)
        scores = []
        for f, (tr, te) in enumerate(reps[0]):
            tr_s = cv.subsample_to(idx_i[tr], y, n_train, cfg.seed + f)
            scores.append(models.sliding(cfg).fit(x[tr_s], y[tr_s]).score(x[idx_i[te]],
                                                                          y[idx_i[te]]))
        out[i, i] = np.mean(scores, axis=0)
        tr_s = cv.subsample_to(idx_i, y, n_train, cfg.seed)
        est = models.sliding(cfg).fit(x[tr_s], y[tr_s])
        for j, tj in enumerate(tasks):
            if j != i:
                te = np.flatnonzero(task == tj)
                out[i, j] = est.score(x[te], y[te])
    return out, tasks, n_train


def loso_cross_task(sets: Sequence[TrialSet], cfg: DecodingConfig,
                    window_s: tuple[float, float], tasks: Sequence[str] | None = None,
                    normalized: bool = False) -> tuple[np.ndarray, list[str]]:
    """Window AUC, train on the other subjects' task i, test the held-out subject's task j.
    Shape (n_subjects, n_tasks, n_tasks); NaN where a subject lacks a task."""
    tasks = list(tasks or cfg.tasks)
    norm = list(sets) if normalized else [normalize(s, cfg) for s in sets]
    out = np.full((len(norm), len(tasks), len(tasks)), np.nan)
    for s_i, test in enumerate(norm):
        train = TrialSet.concat([s for j, s in enumerate(norm) if j != s_i])
        for i, ti in enumerate(tasks):
            tr = train.meta["task"].to_numpy() == ti
            if not tr.any():
                continue
            pipe = models.window_detector(cfg, train.times, window_s).fit(
                train.X[tr], train.labels("seen")[tr])
            for j, tj in enumerate(tasks):
                te = test.meta["task"].to_numpy() == tj
                if te.any():
                    out[s_i, i, j] = _auc(test.labels("seen")[te],
                                          pipe.decision_function(test.X[te]))
    return out, tasks


# ---------- A9 ----------
def cross_condition_auc(ts: TrialSet, cfg: DecodingConfig, col: str, a: object, b: object
                        ) -> dict[str, np.ndarray]:
    """AUC(t) train ``col == a`` -> test ``b``, the reverse, and their mean."""
    x = baseline(ts.X, ts.times, cfg.baseline_s)
    y = ts.labels("seen")
    out = {}
    for name, (p, q) in (("a_to_b", (a, b)), ("b_to_a", (b, a))):
        tr, te = cv.cross_condition(ts.meta, col, p, q)
        out[name] = models.sliding(cfg).fit(x[tr], y[tr]).score(x[te], y[te])
    out["mean"] = (out["a_to_b"] + out["b_to_a"]) / 2
    return out


# ---------- A12 ----------
def transfer(train: Sequence[TrialSet], test: Sequence[TrialSet], cfg: DecodingConfig
             ) -> pd.DataFrame:
    """Train a sliding decoder on ``train`` (pooled), AUC(t) on each test TrialSet.

    All sets must share channels, sfreq and times (``channels.harmonize`` first)."""
    tr = TrialSet.concat([normalize(s, cfg) for s in train])
    est = models.sliding(cfg).fit(tr.X, tr.labels("seen"))
    rows = []
    for s in test:
        n = normalize(s, cfg)
        if n.ch_names != tr.ch_names or n.sfreq != tr.sfreq:
            raise ValueError("test set not harmonized with the training sets")
        rows.append({"dataset": str(n.meta["dataset"].iloc[0]),
                     "subject": str(n.meta["subject"].iloc[0]),
                     "auc": est.score(n.X, n.labels("seen"))})
    return pd.DataFrame(rows)
