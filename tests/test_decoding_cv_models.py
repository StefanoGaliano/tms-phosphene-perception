"""Phase 2 build step 5: splitters honour strata, LOSO never shares a subject; models fit."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from start_vision.decoding import cv, models
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from start_vision.decoding.labels import contrast_bins
from tests.synthetic.decoding import make_subject, make_synthetic_trialset


def test_within_subject_folds_are_stratified_and_partition() -> None:
    m = make_subject("01", "null", n_trials=240).meta
    bins = contrast_bins(m)
    reps = cv.within_subject(m, 5, 3, seed=1, bins=bins)
    assert len(reps) == 3
    key = cv.safe_key(m, 5, bins)
    for folds in reps:
        test_all = np.sort(np.concatenate([te for _, te in folds]))
        assert np.array_equal(test_all, np.arange(len(m)))  # each trial tested once
        for tr, te in folds:
            assert not set(tr) & set(te)
            for k in np.unique(key):  # every stratum spread over folds (±1)
                share = (key[te] == k).sum() / (key == k).sum()
                assert abs(share - 0.2) <= 1 / (key == k).sum() + 1e-9
    assert not all(np.array_equal(a[0][1], b[0][1]) for a, b in itertools.pairwise(reps))


def test_safe_key_coarsens_rare_strata() -> None:
    m = pd.DataFrame({"seen": [1] * 12 + [0] * 12, "task": ["a"] * 24,
                      "subject": "01", "stim_present": True,
                      "gabor_contrast": np.linspace(0, 1, 24)})
    bins = pd.Series([0] * 11 + [1] + [0] * 12)  # one seen trial alone in bin 1
    assert len(np.unique(cv.safe_key(m, 5, bins))) == 2
    with pytest.raises(ValueError):
        cv.safe_key(m.iloc[:15], 5, bins.iloc[:15])  # only 3 unseen


def test_loso_never_shares_a_subject() -> None:
    meta = TrialSet.concat(make_synthetic_trialset(4, 30)).meta
    splits = list(cv.loso(meta))
    assert len(splits) == 4
    for tr, te in splits:
        assert len(set(meta.subject.iloc[te])) == 1
        assert not set(meta.subject.iloc[tr]) & set(meta.subject.iloc[te])


def test_cross_condition_and_subsample() -> None:
    m = make_subject("01", "null", n_trials=100).meta
    tr, te = cv.cross_condition(m, "hemifield", "left", "right")
    assert set(m.hemifield.iloc[tr]) == {"left"} and set(m.hemifield.iloc[te]) == {"right"}
    y = m.seen.to_numpy(int)
    sub = cv.subsample_to(tr, y, 20, seed=0)
    assert abs(len(sub) - 20) <= 1 and set(sub) <= set(tr)
    with pytest.raises(ValueError):
        cv.cross_condition(m, "hemifield", "left", "up")


def test_models_decode_planted_signal_and_not_null() -> None:
    cfg = DecodingConfig(n_jobs=1)
    for gen, expect in (("seen_late", True), ("null", False)):
        ts = make_subject("01", gen, n_trials=200, d=1.0, seed=4)
        y = ts.labels()
        tr, te = np.arange(150), np.arange(150, 200)
        est = models.window_detector(cfg, ts.times, (0.3, 0.5)).fit(ts.X[tr], y[tr])
        auc = roc_auc_score(y[te], est.decision_function(ts.X[te]))
        assert (auc > 0.75) if expect else (0.25 < auc < 0.75), (gen, auc)
    sl = models.sliding(cfg).fit(ts.X[tr], y[tr])
    assert sl.score(ts.X[te], y[te]).shape == (len(ts.times),)
