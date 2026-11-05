"""Within-subject decoding (PHASE2 §7 A1-A3; build step 7).

Inputs are one subject's TrialSet (main set: stimulus-present trials). Data are baselined per
trial on ``cfg.baseline_s`` (label-free, so no leakage). Every estimator is fitted inside a
training fold; scores are ROC AUC on the held-out fold, averaged over folds and repeats.

- ``sliding_auc`` (A1): AUC(t), shape (n_times,).
- ``generalization_auc`` (A2): AUC(train t, test t), shape (n_times, n_times).
- ``patterns`` (A3): Haufe patterns fitted on all trials (no CV), shape (n_channels, n_times).
- ``cv_window_auc``: AUC of any 2D-feature pipeline with the same splits (used by confounds).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from mne.decoding import get_coef
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from start_vision.decoding import cv, models
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from start_vision.decoding.features import baseline
from start_vision.decoding.labels import contrast_bins


def prepared(ts: TrialSet, cfg: DecodingConfig) -> np.ndarray:
    return baseline(ts.X, ts.times, cfg.baseline_s)


def splits_for(ts: TrialSet, cfg: DecodingConfig, n_folds: int | None = None,
               n_repeats: int | None = None) -> list[list[cv.Split]]:
    bins = contrast_bins(ts.meta, cfg.n_contrast_bins, cfg.min_trials_per_bin)
    return cv.within_subject(ts.meta, n_folds or cfg.n_folds, n_repeats or cfg.n_repeats,
                             cfg.seed, bins)


def _score_sets(est_factory: Callable[[], object], X: np.ndarray, y: np.ndarray,
                reps: list[list[cv.Split]]) -> np.ndarray:
    out = []
    for folds in reps:
        fold_scores = []
        for tr, te in folds:
            est = est_factory()
            est.fit(X[tr], y[tr])  # type: ignore[attr-defined]
            fold_scores.append(est.score(X[te], y[te]))  # type: ignore[attr-defined]
        out.append(np.mean(fold_scores, axis=0))
    return np.asarray(np.mean(out, axis=0))


def sliding_auc(ts: TrialSet, cfg: DecodingConfig, target: str = "seen",
                reps: list[list[cv.Split]] | None = None) -> np.ndarray:
    """A1: cross-validated AUC per time point."""
    reps = reps if reps is not None else splits_for(ts, cfg)
    return _score_sets(lambda: models.sliding(cfg), prepared(ts, cfg), ts.labels(target), reps)


def generalization_auc(ts: TrialSet, cfg: DecodingConfig, target: str = "seen",
                       reps: list[list[cv.Split]] | None = None) -> np.ndarray:
    """A2: cross-validated temporal generalization matrix (train time x test time)."""
    reps = reps if reps is not None else splits_for(ts, cfg)
    return _score_sets(lambda: models.generalizing(cfg), prepared(ts, cfg), ts.labels(target),
                       reps)


def patterns(ts: TrialSet, cfg: DecodingConfig, target: str = "seen") -> np.ndarray:
    """A3: Haufe patterns (n_channels, n_times), fitted on all trials (interpretation only)."""
    est = models.sliding(cfg).fit(prepared(ts, cfg), ts.labels(target))
    return np.asarray(get_coef(est, "patterns_", inverse_transform=True))


def cv_window_auc(pipe: Pipeline, F: np.ndarray, y: np.ndarray,
                  reps: list[list[cv.Split]],
                  fold_transform: Callable[[np.ndarray, np.ndarray, np.ndarray],
                                           tuple[np.ndarray, np.ndarray]] | None = None) -> float:
    """Mean over repeats of the AUC of out-of-fold scores pooled within a repeat.

    ``fold_transform(train_idx, F_train, F_test)`` lets a confound transform be fitted on the
    training fold only (A5b, C4b).
    """
    aucs = []
    for folds in reps:
        score = np.full(len(y), np.nan)
        for tr, te in folds:
            ftr, fte = F[tr], F[te]
            if fold_transform is not None:
                ftr, fte = fold_transform(tr, ftr, fte)
            est = clone(pipe).fit(ftr, y[tr])
            score[te] = est.decision_function(fte)
        aucs.append(roc_auc_score(y, score))
    return float(np.mean(aucs))
