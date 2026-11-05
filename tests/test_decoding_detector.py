"""Phase 2 build step 10: detector fit/calibrate/threshold, round trip, contract, TMS artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pytest

from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from start_vision.decoding.detector import (
    ContractError,
    Detector,
    best_threshold,
    threshold_for_fa,
)
from tests.synthetic.decoding import make_subject, make_synthetic_trialset

CFG = DecodingConfig(n_jobs=1)
WIN = (0.2, 0.6)  # starts after t_min_robust = 150 ms


@pytest.fixture(scope="module")
def det() -> Detector:
    sets = make_synthetic_trialset(5, 150, generator="seen_late", d=1.0, seed=1)
    return Detector.fit(sets, CFG, WIN, variant="test", trained_on={"dataset": "synthetic"})


@pytest.fixture(scope="module")
def new_subject() -> TrialSet:
    return make_subject("99", "seen_late", 120, d=1.0, seed=999)


def test_fit_performance_and_calibration(det: Detector, new_subject: TrialSet) -> None:
    perf = det.performance
    assert perf["loso_auc_mean"] > 0.75 and perf["n_subjects"] == 5
    assert 0 < det.contract["threshold"] < 1 and perf["balanced_accuracy_at_threshold"] > 0.65
    p = det.predict_proba(new_subject)
    y = new_subject.labels("seen")
    assert p[y == 1].mean() > p[y == 0].mean() + 0.2
    cal = det.calibration.dropna().query("n >= 20")  # middle bins nearly empty here
    assert (abs(cal.mean_p - cal.observed) < 0.25).all()  # roughly calibrated


def test_round_trip_identical_and_manifest(det: Detector, new_subject: TrialSet,
                                           tmp_path: Path) -> None:
    folder = det.save(tmp_path / "sv-perception-test-v1.0.0")
    again = Detector.load(folder)
    assert np.array_equal(det.predict_proba(new_subject), again.predict_proba(new_subject))
    assert json.loads((folder / "contract.json").read_text())["variant"] == "test"
    with pytest.raises(FileExistsError):
        det.save(folder)  # immutable
    (folder / "contract.json").write_text("{}")
    with pytest.raises(ContractError, match="MANIFEST"):
        Detector.load(folder)


def test_contract_rejects_wrong_channels_sfreq_window(det: Detector,
                                                      new_subject: TrialSet) -> None:
    s = new_subject
    with pytest.raises(ContractError, match="channels"):
        det.predict_proba(s.pick(list(s.ch_names[::-1])))
    with pytest.raises(ContractError, match="sfreq"):
        det.predict_proba(TrialSet(s.X, s.times, s.ch_names, 256.0, s.meta, s.info))
    short = s.times < 0.5
    with pytest.raises(ContractError, match="window"):
        det.predict_proba(TrialSet(s.X[..., short], s.times[short], s.ch_names, s.sfreq,
                                   s.meta, s.info))
    with pytest.raises(ContractError, match="trials"):
        det.predict_proba(s.select(np.arange(len(s)) < 10))


def test_tms_like_artifacts_leave_scores_unchanged(det: Detector,
                                                   new_subject: TrialSet) -> None:
    s = new_subject
    t = s.times
    art = np.zeros_like(s.X)
    decay = 500 * np.exp(-np.clip(t, 0, None) / 0.01) * ((t >= 0) & (t < 0.05))
    spike = 5000 * ((t >= 0) & (t < 0.015))
    rng = np.random.default_rng(0)
    art += (decay + spike)[None, None, :] * rng.uniform(0.5, 1.5, (len(s), len(s.ch_names), 1))
    hit = TrialSet((s.X + art).astype(np.float32), t, s.ch_names, s.sfreq, s.meta, s.info)
    assert np.allclose(det.predict_proba(s), det.predict_proba(hit), atol=1e-6)


def test_from_epochs_reorders_interpolates_and_resamples(det: Detector) -> None:
    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
             "Oz", "T7", "T8"]
    sets = [make_subject(f"{i}", "seen_late", 150, n_channels=len(names) + 1, d=1.0, seed=i)
            for i in range(4)]
    sets = [TrialSet(x.X[:, :-1], x.times, tuple(names), x.sfreq, x.meta,
                     mne.create_info(names, x.sfreq, "eeg")) for x in sets]
    d = Detector.fit(sets, CFG, WIN, variant="t16")
    test = make_subject("9", "seen_late", 120, n_channels=len(names) + 1, d=1.0, seed=50)
    info = mne.create_info(names, 128.0, "eeg")
    ep = mne.EpochsArray(test.X[:, :-1] * 1e-6, info, tmin=test.times[0], verbose="error")
    ep.set_montage("standard_1005", verbose="error")
    p_ref = d.from_epochs(ep)
    shuffled = ep.copy().reorder_channels(names[::-1])
    assert np.allclose(d.from_epochs(shuffled), p_ref)
    dropped = ep.copy().drop_channels(["Oz"])
    assert np.corrcoef(d.from_epochs(dropped), p_ref)[0, 1] > 0.9  # interpolated
    with pytest.raises(ContractError):
        d.from_epochs(ep.copy().drop_channels(names[:6]))  # 6 > max_missing_channels
    up = ep.copy().resample(256, verbose="error")
    with pytest.raises(ContractError, match="allow_resample"):
        d.from_epochs(up)
    assert np.corrcoef(d.from_epochs(up, allow_resample=True), p_ref)[0, 1] > 0.95


def test_threshold_rules() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.45, 0.55, 0.7, 0.9])
    assert 0.45 < best_threshold(y, p) <= 0.55
    assert threshold_for_fa(np.linspace(0, 1, 101), 0.1) == pytest.approx(0.9, abs=0.011)
