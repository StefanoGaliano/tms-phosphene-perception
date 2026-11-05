"""Pipeline factories (PHASE2 D5, §9; build step 5).

Per time point: StandardScaler + LogisticRegression(liblinear, C, class_weight="balanced"),
wrapped in ``mne.decoding.LinearModel`` so Haufe patterns are available (A3). Metric: ROC AUC.
Everything is an unfitted sklearn estimator; fitting happens only inside a training fold.
"""

from __future__ import annotations

import numpy as np
from mne.decoding import GeneralizingEstimator, LinearModel, SlidingEstimator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.features import window_means


def logistic(cfg: DecodingConfig, C: float | None = None) -> LogisticRegression:
    return LogisticRegression(solver="liblinear", C=cfg.C if C is None else C,
                              class_weight="balanced", random_state=cfg.seed)


def time_point(cfg: DecodingConfig, C: float | None = None) -> Pipeline:
    """2D (n_trials, n_features) classifier with Haufe-pattern access."""
    return make_pipeline(StandardScaler(), LinearModel(logistic(cfg, C)))


def sliding(cfg: DecodingConfig) -> SlidingEstimator:
    return SlidingEstimator(time_point(cfg), scoring="roc_auc", n_jobs=cfg.n_jobs,
                            verbose=False)


def generalizing(cfg: DecodingConfig) -> GeneralizingEstimator:
    return GeneralizingEstimator(time_point(cfg), scoring="roc_auc", n_jobs=cfg.n_jobs,
                                 verbose=False)


class WindowFeatures(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """3D epochs -> flattened channel x 50 ms-bin means in ``window`` (§9 step 3). Stateless."""

    def __init__(self, times: np.ndarray, window: tuple[float, float],
                 bin_s: float = 0.05) -> None:
        self.times = times
        self.window = window
        self.bin_s = bin_s

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> WindowFeatures:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return window_means(np.asarray(X), self.times, self.window, self.bin_s)


def window_detector(cfg: DecodingConfig, times: np.ndarray, window: tuple[float, float],
                    C: float | None = None) -> Pipeline:
    """One score per trial: window features + scaler + balanced LR (D8)."""
    return make_pipeline(WindowFeatures(times, window, cfg.feature_bin_s), StandardScaler(),
                         LinearModel(logistic(cfg, C)))
