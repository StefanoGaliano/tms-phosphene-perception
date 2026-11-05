"""Phase 2 build step 7: synthetic acceptance criteria of PHASE2 §7 (A1-A6, C3-C5)."""

from __future__ import annotations

import numpy as np
import pytest

from start_vision.decoding import confounds, stats, within
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from tests.synthetic.decoding import make_subject

CFG = DecodingConfig(n_jobs=1, n_repeats=1)


def _win(times: np.ndarray, a: float, b: float) -> np.ndarray:
    return (times >= a) & (times < b)


def test_a1_finds_late_signal_in_its_window() -> None:
    aucs = np.array([within.sliding_auc(make_subject(f"{i}", "seen_late", 160, d=0.8, seed=i),
                                        CFG) for i in range(8)])
    times = make_subject("0", "null", 10).times
    assert aucs[:, _win(times, 0.32, 0.48)].mean() > 0.7
    assert abs(aucs[:, times < 0].mean() - 0.5) < 0.05
    mask = stats.cluster_mask(aucs, n_permutations=500)
    assert mask[_win(times, 0.32, 0.48)].all() and not mask[times < 0].any()


@pytest.mark.slow
def test_a1_null_no_prestimulus_cluster_in_95_percent() -> None:
    """100 null group simulations (10 subjects each, drawn from a pool of 40 decoded nulls)."""
    pool = np.array([within.sliding_auc(make_subject(f"{i}", "null", 160, seed=100 + i), CFG)
                     for i in range(40)])
    times = make_subject("0", "null", 10).times
    rng = np.random.default_rng(0)
    bad = 0
    for k in range(100):
        g = pool[rng.choice(40, 10, replace=False)]
        bad += bool(stats.cluster_mask(g, n_permutations=200, seed=k)[times < 0].any())
    assert bad <= 5, bad


def test_a2_square_vs_diagonal() -> None:
    times = make_subject("0", "null", 10).times
    lo, hi = _win(times, 0.25, 0.35), _win(times, 0.65, 0.75)
    sus = within.generalization_auc(make_subject("1", "sustained", 200, d=1.0, seed=1), CFG)
    chain = within.generalization_auc(make_subject("1", "chain", 200, d=1.0, seed=1), CFG)
    # sustained: training early generalizes late; chain: it does not
    assert sus[np.ix_(lo, hi)].mean() > 0.7
    assert abs(chain[np.ix_(lo, hi)].mean() - 0.5) < 0.08
    assert chain[np.ix_(lo, lo)].mean() > 0.7 and chain[np.ix_(hi, hi)].mean() > 0.7


def test_a3_patterns_match_ground_truth() -> None:
    ts = make_subject("1", "seen_late", 300, d=1.0, seed=2)
    p = within.patterns(ts, CFG)
    eeg = slice(0, len(ts.ch_names) - 1)
    pat = p[eeg][:, _win(ts.times, 0.35, 0.45)].mean(1)
    truth = np.random.default_rng(12345).standard_normal(len(ts.ch_names) - 1)
    assert abs(np.corrcoef(pat, truth)[0, 1]) > 0.9


def test_a4_contrast_only_decodes_in_a1_not_within_bins() -> None:
    ts = make_subject("1", "contrast_only", 400, d=2.0, seed=3)
    times = ts.times
    w = _win(times, 0.15, 0.45)
    a1 = within.sliding_auc(ts, CFG)
    a4, used = confounds.within_bin_auc(ts, CFG)
    assert a1[w].mean() > 0.65
    assert a4[w].mean() < a1[w].mean() - 0.08 and sum(used.values()) <= len(ts)


def test_a5_residualization_removes_contrast_signal() -> None:
    ts = make_subject("1", "contrast_only", 400, d=2.0, seed=3)
    w = _win(ts.times, 0.15, 0.45)
    assert confounds.residualized_auc(ts, CFG)[w].mean() < within.sliding_auc(ts, CFG)[w].mean()
    cd = confounds.contrast_decoding(ts, CFG)
    assert cd["within_seen"][w].mean() > 0.55 or cd["within_unseen"][w].mean() > 0.55


def test_residualize_is_fitted_on_train_only() -> None:
    rng = np.random.default_rng(0)
    c = rng.normal(size=50)
    x = 3 * c[:, None] + rng.normal(size=(50, 4))
    ra, _ = confounds.residualize(c[:40], x[:40], x[40:], c[40:])
    assert abs(np.corrcoef(c[:40], ra[:, 0])[0, 1]) < 1e-6
    # test rows use the training slope: changing test x does not change the training fit
    ra2, _ = confounds.residualize(c[:40], x[:40], x[40:] + 100, c[40:])
    assert np.allclose(ra, ra2)


def test_a6_key_lateralized_decodes_key_but_not_across_mappings() -> None:
    ts = make_subject("1", "key_lateralized", 300, d=1.0, seed=4)
    w = _win(ts.times, 0.35, 0.55)
    r = confounds.response_key(ts, CFG)
    assert r is not None
    assert r["key_auc"][w].mean() > 0.75
    assert r["cross_key_auc"][w].mean() < 0.4  # sign flips across mappings: not a percept


def test_ds006171_a6_without_key_columns_returns_none() -> None:
    ts = make_subject("1", "null", 60)
    meta = ts.meta.assign(resp_key_seen=np.nan, resp_key_pressed=np.nan)
    ts2 = TrialSet(ts.X, ts.times, ts.ch_names, ts.sfreq, meta, ts.info)
    assert confounds.response_key(ts2, CFG) is None


def test_c3_eog_only_and_step_mask() -> None:
    ts = make_subject("1", "eog_only", 200, d=1.0, seed=5)
    eog = ts.pick(["HEOG"])
    assert confounds.eog_only_auc(eog, CFG)[_win(ts.times, 0.25, 0.45)].mean() > 0.7
    heog = np.zeros((3, len(ts.times)))
    heog[1, ts.times >= 0.1] = 50.0  # one step of 50 µV at 100 ms
    m = confounds.heog_step_mask(heog, ts.times, 32.0, (-0.05, 0.35))
    assert list(m) == [False, True, False]


def test_c4_alpha_models() -> None:
    ts = make_subject("1", "alpha_pre", 300, d=1.0, seed=6)
    r = confounds.alpha_models(ts, CFG, (0.2, 0.6))
    assert r["alpha_only"] > 0.7 and abs(r["post_only"] - 0.5) < 0.1
    late = make_subject("1", "seen_late", 300, d=1.0, seed=6)
    r2 = confounds.alpha_models(late, CFG, (0.2, 0.6))
    assert r2["post_only"] > 0.75 and r2["post_resid_alpha"] > 0.75
    s = confounds.alpha_split_scores(late, CFG, (0.2, 0.6))
    assert set(s) == {"low_alpha", "high_alpha"}


def test_c5_undersampling_and_catch() -> None:
    ts = make_subject("1", "seen_late", 200, d=1.0, seed=7)
    u = confounds.undersampled_auc(ts, CFG, (0.3, 0.5), n_draws=3)
    assert u.shape == (3,) and (u > 0.75).all()
    catch = make_subject("2", "seen_late", 60, d=1.0, seed=8)  # 'seen' here = FA with signal
    r = confounds.catch_scores(ts, catch, CFG, (0.3, 0.5))
    assert r["n_fa"] + r["n_cr"] == 60 and r["auc_fa_vs_cr"] > 0.7
