"""Confound controls, within subject (PHASE2 §7 A4-A6, C3-C5; build step 7).

- A4 ``within_bin_auc``: A1 trained and tested inside each contrast bin, averaged by trial count.
- A5 ``contrast_decoding`` (a): decode high vs low contrast within seen and within unseen;
  ``residualized_auc`` (b): contrast regressed out of X per channel x time, fitted on the
  training fold only, then A1.
- A6 ``response_key``: (a) decode the pressed key; (b) cross-key generalization of seen/unseen.
  ds006171 has no key information (decision 3): returns ``None`` with a reason.
- C3 ``eog_only_auc`` (a), ``heog_step_mask`` (c). (b) = A1 on the ``desc-preproc`` copy.
- C4 ``alpha_models``: window AUC of alpha-only, post-only, alpha+post and post residualized on
  alpha (fold-fitted); ``alpha_split_scores``: detector scores by alpha median split.
- C5 ``undersampled_auc``: window AUC with equalized classes (random undersampling, 20 draws);
  ``catch_scores``: window detector from stimulus-present trials applied to catch trials.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from start_vision.decoding import cv, models, within
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from start_vision.decoding.features import alpha_power, individual_alpha_band, window_means
from start_vision.decoding.labels import contrast_bins


# ---------- A4 ----------
def within_bin_auc(ts: TrialSet, cfg: DecodingConfig, n_repeats: int | None = None
                   ) -> tuple[np.ndarray, dict[int, int]]:
    """AUC(t) averaged over contrast bins weighted by trial count; and trials per bin used."""
    bins = contrast_bins(ts.meta, cfg.n_contrast_bins, cfg.min_trials_per_bin).to_numpy()
    acc, weights = [], {}
    for b in np.unique(bins):
        sub = ts.select(bins == b)
        y = sub.labels("seen")
        n_min = int(min(y.sum(), (1 - y).sum()))
        if n_min < 2:
            continue
        reps = cv.within_subject(sub.meta, min(cfg.n_folds, n_min), n_repeats or cfg.n_repeats,
                                 cfg.seed, bins=sub.meta["task"].map(lambda _: 0))
        acc.append(within.sliding_auc(sub, cfg, reps=reps) * len(sub))
        weights[int(b)] = len(sub)
    if not acc:
        raise ValueError("no contrast bin has both classes")
    return np.asarray(np.sum(acc, axis=0) / sum(weights.values())), weights


# ---------- A5 ----------
def contrast_decoding(ts: TrialSet, cfg: DecodingConfig, n_repeats: int | None = None
                      ) -> dict[str, np.ndarray]:
    """(a) AUC(t) for high vs low contrast (median split) within seen and within unseen."""
    out = {}
    for name, cls in (("within_seen", True), ("within_unseen", False)):
        sub = ts.select(ts.meta["seen"].astype(bool).to_numpy() == cls)
        c = sub.meta["gabor_contrast"].to_numpy(float)
        high = (c > np.median(c)).astype(int)
        meta = sub.meta.assign(seen=high)  # stratify on the contrast class
        reps = cv.within_subject(meta, cfg.n_folds, n_repeats or cfg.n_repeats, cfg.seed,
                                 bins=meta["task"].map(lambda _: 0))
        x = within.prepared(sub, cfg)
        out[name] = within._score_sets(lambda: models.sliding(cfg), x, high, reps)
    return out


def residualize(c_train: np.ndarray, x_train: np.ndarray, x_test: np.ndarray,
                c_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regress ``c`` out of every feature with slopes fitted on the training trials only."""
    cm = c_train.mean()
    cc = c_train - cm
    shape = x_train.shape[1:]
    a, b = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
    mu = a.mean(0)
    beta = (cc @ (a - mu)) / (cc @ cc)
    ra = a - np.outer(cc, beta)
    rb = b - np.outer(c_test - cm, beta)
    return ra.reshape((-1, *shape)), rb.reshape((-1, *shape))


def residualized_auc(ts: TrialSet, cfg: DecodingConfig,
                     reps: list[list[cv.Split]] | None = None) -> np.ndarray:
    """(b) A1 after regressing contrast out of X (fold-fitted)."""
    reps = reps if reps is not None else within.splits_for(ts, cfg)
    x, y = within.prepared(ts, cfg), ts.labels("seen")
    c = ts.meta["gabor_contrast"].to_numpy(float)
    out = []
    for folds in reps:
        scores = []
        for tr, te in folds:
            xtr, xte = residualize(c[tr], x[tr], x[te], c[te])
            est = models.sliding(cfg).fit(xtr, y[tr])
            scores.append(est.score(xte, y[te]))
        out.append(np.mean(scores, axis=0))
    return np.asarray(np.mean(out, axis=0))


# ---------- A6 ----------
def response_key(ts: TrialSet, cfg: DecodingConfig, n_repeats: int | None = None
                 ) -> dict[str, Any] | None:
    """(a) pressed-key AUC(t); (b) seen/unseen trained on yes=left, tested on yes=right, and
    reverse (mean of both directions). ``None`` if the key columns are missing."""
    m = ts.meta
    if m["resp_key_pressed"].isna().any() or m["resp_key_seen"].isna().any():
        return None
    pressed = (m["resp_key_pressed"] == "left").astype(int).to_numpy()
    meta = m.assign(seen=pressed)
    reps = cv.within_subject(meta, cfg.n_folds, n_repeats or cfg.n_repeats, cfg.seed,
                             bins=meta["task"].map(lambda _: 0))
    x, y = within.prepared(ts, cfg), ts.labels("seen")
    key_auc = within._score_sets(lambda: models.sliding(cfg), x, pressed, reps)
    cross = []
    for a, b in (("left", "right"), ("right", "left")):
        tr, te = cv.cross_condition(m, "resp_key_seen", a, b)
        cross.append(models.sliding(cfg).fit(x[tr], y[tr]).score(x[te], y[te]))
    return {"key_auc": key_auc, "cross_key_auc": np.mean(cross, axis=0)}


# ---------- C3 ----------
def eog_only_auc(ts_eog: TrialSet, cfg: DecodingConfig,
                 reps: list[list[cv.Split]] | None = None) -> np.ndarray:
    """(a) A1 on the EOG channels only (``ts_eog`` holds just them)."""
    return within.sliding_auc(ts_eog, cfg, reps=reps)


def heog_step_mask(heog: np.ndarray, times: np.ndarray, threshold_uv: float,
                   window_s: tuple[float, float], win_s: float = 0.05) -> np.ndarray:
    """(c) True for trials whose HEOG step (mean of the later minus the earlier half of a
    sliding ``win_s`` window) exceeds ``threshold_uv`` inside ``window_s`` (step-like test)."""
    sf = 1 / (times[1] - times[0])
    half = max(1, round(win_s * sf / 2))
    sel = np.flatnonzero((times >= window_s[0]) & (times <= window_s[1]))
    worst = np.zeros(len(heog))
    for i in sel:
        if i - half < 0 or i + half > heog.shape[1]:
            continue
        step = np.abs(heog[:, i:i + half].mean(1) - heog[:, i - half:i].mean(1))
        worst = np.maximum(worst, step)
    return np.asarray(worst > threshold_uv)


# ---------- C4 ----------
def _lr(cfg: DecodingConfig) -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(
        solver="liblinear", C=cfg.C, class_weight="balanced", random_state=cfg.seed))


def alpha_features(ts: TrialSet, cfg: DecodingConfig) -> tuple[np.ndarray, tuple[float, float]]:
    band = individual_alpha_band(ts, cfg.alpha_band_hz, cfg.alpha_peak_halfwidth_hz,
                                 cfg.alpha_psd_window_s)
    return alpha_power(ts, band, cfg.alpha_window_s), band


def alpha_models(ts: TrialSet, cfg: DecodingConfig, post_window_s: tuple[float, float],
                 reps: list[list[cv.Split]] | None = None, alpha_ts: TrialSet | None = None
                 ) -> dict[str, float]:
    """Window AUC: alpha-only, post-only, alpha+post, post residualized on alpha (fold-fitted).

    ``alpha_ts``: the same trials before decimation (alpha from the non-decimated data, C4);
    defaults to ``ts``.
    """
    reps = reps if reps is not None else within.splits_for(ts, cfg)
    y = ts.labels("seen")
    a, _ = alpha_features(alpha_ts if alpha_ts is not None else ts, cfg)
    post = window_means(within.prepared(ts, cfg), ts.times, post_window_s, cfg.feature_bin_s)
    both = np.hstack([a, post])


    return {
        "alpha_only": within.cv_window_auc(_lr(cfg), a, y, reps),
        "post_only": within.cv_window_auc(_lr(cfg), post, y, reps),
        "alpha_plus_post": within.cv_window_auc(_lr(cfg), both, y, reps),
        "post_resid_alpha": _cv_resid(cfg, post, a.mean(1), y, reps),
    }


def _cv_resid(cfg: DecodingConfig, post: np.ndarray, am: np.ndarray, y: np.ndarray,
              reps: list[list[cv.Split]]) -> float:
    aucs = []
    for folds in reps:
        score = np.full(len(y), np.nan)
        for tr, te in folds:
            ftr, fte = residualize(am[tr], post[tr], post[te], am[te])
            score[te] = _lr(cfg).fit(ftr, y[tr]).decision_function(fte)
        aucs.append(roc_auc_score(y, score))
    return float(np.mean(aucs))


def alpha_split_scores(ts: TrialSet, cfg: DecodingConfig, post_window_s: tuple[float, float],
                       reps: list[list[cv.Split]] | None = None,
                       alpha_ts: TrialSet | None = None) -> dict[str, float]:
    """Out-of-fold window-detector AUC within low- and high-alpha halves (median split)."""
    reps = reps if reps is not None else within.splits_for(ts, cfg, n_repeats=1)
    y = ts.labels("seen")
    am = alpha_features(alpha_ts if alpha_ts is not None else ts, cfg)[0].mean(1)
    pipe = models.window_detector(cfg, ts.times, post_window_s)
    x = within.prepared(ts, cfg)
    score = np.full(len(y), np.nan)
    for tr, te in reps[0]:
        score[te] = pipe.fit(x[tr], y[tr]).decision_function(x[te])
    low = am <= np.median(am)
    return {name: float(roc_auc_score(y[m], score[m]))
            for name, m in (("low_alpha", low), ("high_alpha", ~low))
            if len(np.unique(y[m])) == 2}


# ---------- C5 ----------
def undersampled_auc(ts: TrialSet, cfg: DecodingConfig, post_window_s: tuple[float, float],
                     n_draws: int = 20) -> np.ndarray:
    """Window AUC with classes equalized by random undersampling, one value per draw."""
    rng = np.random.default_rng(cfg.seed)
    y = ts.labels("seen")
    n = int(min(y.sum(), (1 - y).sum()))
    out = []
    for _ in range(n_draws):
        keep = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=False)
                                       for c in (0, 1)]))
        sub = ts.select(np.isin(np.arange(len(y)), keep))
        reps = within.splits_for(sub, cfg, n_repeats=1)
        f = window_means(within.prepared(sub, cfg), sub.times, post_window_s, cfg.feature_bin_s)
        out.append(within.cv_window_auc(_lr(cfg), f, sub.labels("seen"), reps))
    return np.asarray(out)


def catch_scores(main: TrialSet, catch: TrialSet, cfg: DecodingConfig,
                 post_window_s: tuple[float, float]) -> dict[str, Any]:
    """Train on all stimulus-present trials, score catch trials; AUC of FA vs CR scores."""
    pipe = models.window_detector(cfg, main.times, post_window_s)
    pipe.fit(within.prepared(main, cfg), main.labels("seen"))
    s = pipe.decision_function(within.prepared(catch, cfg))
    fa = catch.meta["seen"].astype(bool).to_numpy()
    auc = float(roc_auc_score(fa, s)) if 0 < fa.sum() < len(fa) else float("nan")
    return {"n_fa": int(fa.sum()), "n_cr": int((~fa).sum()), "auc_fa_vs_cr": auc,
            "scores": s}
