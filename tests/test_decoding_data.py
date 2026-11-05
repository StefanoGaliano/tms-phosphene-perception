"""Phase 2 build steps 2-4: config, synthetic TrialSets, labels, features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from start_vision.decoding import features as F  # noqa: N812
from start_vision.decoding import labels as L  # noqa: N812
from start_vision.decoding.config import DecodingConfig, load
from start_vision.decoding.data import META_COLUMNS, TrialSet, meta_from_staged
from tests.synthetic.decoding import GENERATORS, make_subject, make_synthetic_trialset

ROOT = Path(__file__).parents[1]


def test_config_defaults_and_yaml() -> None:
    c = DecodingConfig()
    assert c.sfreq == 128 and c.window_s == (-0.5, 1.0) and c.C == 1.0
    assert load(ROOT / "configs" / "decoding.yaml").n_jobs >= 1
    with pytest.raises(ValueError):
        DecodingConfig(unknown_key=1)  # type: ignore[call-arg]


@pytest.mark.parametrize("gen", GENERATORS)
def test_synthetic_shapes_and_meta(gen: str) -> None:
    ts = make_subject("01", gen, n_trials=60)
    assert ts.X.shape == (60, 32, 192) and ts.X.dtype == np.float32
    assert list(ts.meta.columns) == list(META_COLUMNS)
    assert set(ts.labels("seen")) <= {0, 1}


def test_synthetic_is_deterministic() -> None:
    a, b = make_synthetic_trialset(2, 50, seed=3), make_synthetic_trialset(2, 50, seed=3)
    assert all(np.array_equal(x.X, y.X) for x, y in zip(a, b, strict=True))


def test_trialset_select_concat_and_shape_check() -> None:
    a, b = make_synthetic_trialset(2, 40)
    both = TrialSet.concat([a, b])
    assert len(both) == 80 and set(both.meta.subject) == {"01", "02"}
    assert len(both.select(both.meta.subject == "02")) == 40
    with pytest.raises(ValueError, match="shapes"):
        TrialSet(a.X[:5], a.times, a.ch_names, a.sfreq, a.meta, a.info)


def test_ds006171_meta_from_staged_tags() -> None:
    rows = pd.DataFrame({
        "trial_type": ["stimulus/seen/right/vertical/m1", "stimulus/unseen/left/horizontal/m0",
                       "stimulus/fa/left", "stimulus/cr/sideunknown"],
        "trial_number": [1, 2, 3, 4], "raw_code": [21, 12, 31, 98],
        "gabor_contrast": [0.03, 0.02, np.nan, np.nan], "rt_ms": [600, 700, 800, 900],
        "cue_valid": [np.nan] * 4, "objective_correct": [np.nan] * 4,
    })
    m = meta_from_staged(rows, "01", "nocue")
    assert list(m.seen) == [True, False, True, False]
    assert list(m.stim_present) == [True, True, False, False]
    assert list(m.hemifield) == ["right", "left", "left", "unknown"]
    assert list(m.matched) == [True, False, False, False]
    assert m.resp_key_seen.isna().all()


def test_contrast_bins_merge_when_sparse() -> None:
    ts = make_subject("01", "null", n_trials=300)
    b = L.contrast_bins(ts.meta, 3, 10)
    assert b.nunique() == 3
    sparse = make_subject("01", "contrast_only", n_trials=60).meta
    assert L.contrast_bins(sparse, 3, 10).nunique() < 3


def test_stratify_key_and_inclusion() -> None:
    ts = make_subject("01", "null", n_trials=200)
    k = L.stratify_key(ts.meta)
    assert len(np.unique(k)) == 6  # seen x 1 task x 3 bins
    assert L.enough_trials(ts, 30)[0] and not L.enough_trials(ts, 150)[0]


def test_baseline_and_prestim_normalize() -> None:
    ts = make_subject("01", "null", n_trials=50)
    xb = F.baseline(ts.X, ts.times, (-0.2, 0.0))
    pre = (ts.times >= -0.2) & (ts.times < 0)
    assert np.allclose(xb[..., pre].mean(-1), 0, atol=1e-5)
    z = F.prestim_normalize(ts.X, ts.times, (-0.5, -0.1))
    m = (ts.times >= -0.5) & (ts.times < -0.1)
    assert np.allclose(z[..., m].mean(axis=(0, 2)), 0, atol=1e-5)
    assert np.allclose(z[..., m].std(axis=(0, 2)), 1, atol=1e-4)


def test_window_means_shape_and_values() -> None:
    times = np.arange(-64, 128) / 128
    x = np.broadcast_to(times, (4, 3, len(times))).copy()
    f = F.window_means(x, times, (0.2, 0.6), 0.05)
    assert f.shape == (4, 3 * 8)
    assert f[0, 0] == pytest.approx(times[(times >= 0.2) & (times < 0.25)].mean())


def test_alpha_power_recovers_planted_difference() -> None:
    ts = make_subject("01", "alpha_pre", n_trials=120, d=1.0)
    p = F.alpha_power(ts, (8, 13), (-0.5, -0.1)).mean(1)
    seen = ts.meta.seen.to_numpy(bool)
    assert p[~seen].mean() > p[seen].mean()
