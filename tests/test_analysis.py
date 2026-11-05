"""analysis: pooling weights, contrasts from pooled conditions, stats equal to eegunify's."""

from __future__ import annotations

import mne
import numpy as np
from scipy import stats as sps

from eegunify.erp.group import effect_size
from start_vision.analysis import pool, stats


def _ev(value: float, nave: int, name: str) -> mne.Evoked:
    info = mne.create_info(["O1", "O2", "Pz"], 256.0, "eeg")
    e = mne.EvokedArray(np.full((3, 50), value * 1e-6), info, tmin=-0.1, nave=nave)
    e.comment = name
    return e


def test_pool_weights_by_trials_and_recomputes_contrast() -> None:
    a = {"seen": _ev(2.0, 30, "seen"), "unseen": _ev(0.0, 10, "unseen")}
    b = {"seen": _ev(-2.0, 10, "seen"), "unseen": _ev(4.0, 30, "unseen")}
    p = pool.pool_subject({"nocue": a, "informative": b})
    assert np.allclose(p["seen"].data * 1e6, 1.0) and p["seen"].nave == 40
    assert np.allclose(p["unseen"].data * 1e6, 3.0)
    assert np.allclose(p["seen_minus_unseen"].data * 1e6, -2.0)  # not the mean of contrasts


def test_pool_skips_task_without_condition() -> None:
    a = {"seen_contra": _ev(1.0, 20, "seen_contra")}
    p = pool.pool_subject({"nocue": a, "noninformative": {}})
    assert p["seen_contra"].nave == 20


def test_paired_equals_eegunify() -> None:
    x = np.random.default_rng(3).normal(-1.0, 1.5, 20)
    r = stats.paired(x)
    dz, lo, hi = effect_size(x, 10_000, 97)
    assert abs(r["d_z"] - dz) < 1e-12 and abs(r["ci_low"] - lo) < 1e-12
    assert abs(r["ci_high"] - hi) < 1e-12
    assert abs(r["t"] - sps.ttest_1samp(x, 0).statistic) < 1e-12


def test_paired_no_ci_below_8() -> None:
    assert np.isnan(stats.paired([1.0, 2.0, 0.5])["ci_low"])
