"""Phase 2 build step 8: LOSO, cross-task, cross-condition, transfer (PHASE2 §7 criteria)."""

from __future__ import annotations

import numpy as np
import pytest

from start_vision.decoding import across, stats
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from tests.synthetic.decoding import make_subject, make_synthetic_trialset

CFG = DecodingConfig(n_jobs=1, n_repeats=1)


def _win(times: np.ndarray, a: float, b: float) -> np.ndarray:
    return (times >= a) & (times < b)


def test_normalize_is_label_free_and_per_subject() -> None:
    ts = make_subject("1", "seen_late", 80, seed=1)
    flipped = TrialSet(ts.X, ts.times, ts.ch_names, ts.sfreq,
                       ts.meta.assign(seen=~ts.meta.seen.astype(bool)), ts.info)
    assert np.array_equal(across.normalize(ts, CFG).X, across.normalize(flipped, CFG).X)


def test_a7_loso_decodes_jittered_subjects_and_not_noise() -> None:
    sig = make_synthetic_trialset(6, 150, generator="seen_late", d=1.0, seed=2)
    auc, subs = across.loso_sliding(sig, CFG)
    times = sig[0].times
    w = _win(times, 0.32, 0.48)
    assert subs == [f"{i:02d}" for i in range(1, 7)]
    assert (auc[:, w].mean(1) > 0.65).all()
    table, preds = across.loso_window(sig, CFG, (0.3, 0.5))
    assert (table.auc > 0.7).all() and len(preds) == 6 * 150
    assert set(preds.columns) >= {"subject", "trial_number", "seen", "score"}
    # 16 subjects: the percentile bootstrap CI undercovers at N = 8 (10/12 null runs)
    null = make_synthetic_trialset(16, 150, generator="null", seed=3)
    t_null, _ = across.loso_window(null, CFG, (0.3, 0.5))
    r = stats.window_test(t_null.auc.to_numpy())
    assert r["ci_low"] <= 0.5 <= r["ci_high"]


def _two_task_subject(gen_b: str, seed: int) -> TrialSet:
    a = make_subject("1", "seen_late", 200, d=1.0, seed=seed)
    b = make_subject("1", gen_b, 160, d=1.0, seed=seed + 1)
    b = TrialSet(b.X, b.times, b.ch_names, b.sfreq, b.meta.assign(task="informative"), b.info)
    return TrialSet.concat([a, b])


def test_a8_cross_task_matrix_invariant_and_specific() -> None:
    times = make_subject("0", "null", 10).times
    w = _win(times, 0.32, 0.48)
    inv, tasks, n_train = across.cross_task_within(_two_task_subject("seen_late", 4), CFG)
    assert tasks == ["nocue", "informative"] and n_train == 128
    assert (inv[:, :, w].mean(-1) > 0.65).all()  # signal shared: every cell decodes
    spec, _, _ = across.cross_task_within(_two_task_subject("null", 4), CFG)
    m = spec[:, :, w].mean(-1)
    assert m[0, 0] > 0.65 and abs(m[0, 1] - 0.5) < 0.1 and abs(m[1, 1] - 0.5) < 0.1


def test_a8_loso_cross_task_shape_and_missing_task() -> None:
    sets = [_two_task_subject("seen_late", 10 + i) for i in range(3)]
    sets = [TrialSet(s.X, s.times, s.ch_names, s.sfreq, s.meta.assign(subject=f"{i}"), s.info)
            for i, s in enumerate(sets)]
    out, _ = across.loso_cross_task(sets, CFG, (0.3, 0.5))
    assert out.shape == (3, 3, 3) and np.isnan(out[:, 1, :]).all()  # no noninformative
    assert np.nanmin(out) > 0.6


def test_a9_cross_hemifield() -> None:
    ts = make_subject("1", "seen_late", 300, d=1.0, seed=5)
    r = across.cross_condition_auc(ts, CFG, "hemifield", "left", "right")
    assert r["mean"][_win(ts.times, 0.32, 0.48)].mean() > 0.7
    with pytest.raises(ValueError):
        across.cross_condition_auc(ts, CFG, "hemifield", "left", "unknown")


def test_a12_transfer_and_harmonization_check() -> None:
    train = make_synthetic_trialset(3, 120, generator="seen_late", seed=6)
    test = make_synthetic_trialset(2, 120, generator="seen_late", seed=7)
    df = across.transfer(train, test, CFG)
    assert len(df) == 2 and all(a[_win(train[0].times, 0.32, 0.48)].mean() > 0.65
                                for a in df.auc)
    other = make_subject("9", "seen_late", 60, n_channels=16, seed=8)
    with pytest.raises(ValueError):
        across.transfer(train, [other], CFG)


def test_ds006171_loso_low_memory_matches_reference() -> None:
    """The memory-saving LOSO (34 subjects ran out of memory, 2026-11-05) must give the same
    held-out scores as the plain SlidingEstimator / full window pipeline."""
    from start_vision.decoding import models

    sets = [across.normalize(s, CFG) for s in
            make_synthetic_trialset(4, 80, generator="seen_late", d=1.0, seed=7)]
    auc, _ = across.loso_sliding(sets, CFG, normalized=True)
    table, preds = across.loso_window(sets, CFG, (0.3, 0.5), normalized=True)
    for i, test in enumerate(sets):
        train = TrialSet.concat([s for j, s in enumerate(sets) if j != i])
        ref = models.sliding(CFG).fit(train.X, train.labels("seen")).score(
            test.X, test.labels("seen"))
        np.testing.assert_allclose(auc[i], ref, atol=1e-10)
        pipe = models.window_detector(CFG, test.times, (0.3, 0.5)).fit(
            train.X, train.labels("seen"))
        sub = str(test.meta["subject"].iloc[0])
        np.testing.assert_allclose(preds.loc[preds.subject == sub, "score"].to_numpy(),
                                   pipe.decision_function(test.X), atol=1e-8)
