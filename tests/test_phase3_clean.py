"""Step 8: detector-matched chain on synthetic TMS-EEG (blueprint §12)."""
import json
from pathlib import Path

import numpy as np
import pytest

from start_vision.decoding.detector import Detector
from start_vision.phase3 import clean, config
from tests.synthetic.tms import simulate

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.slow


def _rec(seed: int = 5, shuffle: bool = False) -> clean.Recording:
    s = simulate(n_pulses=80, sfreq=2000.0, seed=seed)
    raw = s.raw.copy()
    raw.set_annotations(None)
    codes = ["S100" if lab else "S200" for lab in s.labels]
    if shuffle:
        codes = list(np.random.default_rng(0).permutation(codes))
    return clean.Recording(raw, s.events[:, 0], codes)


@pytest.fixture(scope="module")
def cfg() -> config.Phase3Config:
    return config.load(ROOT / "configs" / "phase3.yaml")


@pytest.fixture(scope="module")
def cleaned(cfg: config.Phase3Config):  # type: ignore[no-untyped-def]
    return clean.clean(_rec(), cfg, ROOT)


def test_phase3_detector_ready_contract(cleaned, cfg) -> None:  # type: ignore[no-untyped-def]
    ep, qa = cleaned
    det = Detector.load(ROOT / cfg.detector)
    c = det.contract
    assert ep.info["sfreq"] == c["sfreq"]
    eeg = ep.copy().pick("eeg")
    assert eeg.ch_names == c["channels"]
    win = eeg.copy().crop(c["epoch_window_s"][0], c["epoch_window_s"][1])
    det.check(win.ch_names, win.info["sfreq"], win.times)
    p = det.from_epochs(win)
    assert len(p) == len(ep) and np.isfinite(p).all()
    assert set(ep.metadata.columns) >= {"pulse", "code", "seen"}
    json.dumps(qa, default=str)


def test_tms_chain_label_blind(cleaned, cfg) -> None:  # type: ignore[no-untyped-def]
    ep, _ = cleaned
    ep2, _ = clean.clean(_rec(shuffle=True), cfg, ROOT)
    assert np.array_equal(ep.get_data(), ep2.get_data())
    assert list(ep.metadata["pulse"]) == list(ep2.metadata["pulse"])
