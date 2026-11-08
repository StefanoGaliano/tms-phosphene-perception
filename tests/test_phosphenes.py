import json
from pathlib import Path

import mne
import numpy as np

from start_vision.decoding.channels import positions
from start_vision.phosphenes import complete_contract_channels

CONTRACT = Path(__file__).parents[1] / "detectors/sv-perception-tms-robust-v1.0.0/contract.json"
ABSENT = ["P9", "P10", "Iz", "Fpz", "AFz"]


def _cap(contract: list[str]) -> list[str]:
    # the GIN cap: contract minus 5 channels, Fp1/Fp2 upper-case, plus EOG and mastoid
    eeg = [c.upper() if c in ("Fp1", "Fp2") else c for c in contract if c not in ABSENT]
    return [*eeg, "HEOGL", "VEOGUP", "LM"]


def test_phosphenes_gin_cap_interpolates_five_missing() -> None:
    chans = json.loads(CONTRACT.read_text())["channels"]
    names = _cap(chans)
    pos = positions([c for c in chans if c not in ABSENT])
    # smooth field (a linear function of position) -> interpolation should recover it
    def field(p: np.ndarray) -> float:
        return float(1e-4 * (p @ np.array([1.0, 2.0, 3.0])))
    rows = []
    for n in names:
        key = {"FP1": "Fp1", "FP2": "Fp2"}.get(n, n)
        rows.append(np.full(50, field(pos[key])) if key in pos else np.zeros(50))
    types = ["eeg"] * (len(names) - 3) + ["eog", "eog", "misc"]
    raw = mne.io.RawArray(np.array(rows), mne.create_info(names, 500.0, types), verbose="error")
    out, missing = complete_contract_channels(raw, chans)
    assert sorted(missing) == sorted(ABSENT)
    assert out.ch_names == chans
    full = positions(chans)
    span = np.ptp([field(full[c]) for c in chans])
    for c in ABSENT:
        got = out.get_data(picks=[c])[0, 0]
        assert abs(got - field(full[c])) < 0.25 * span


def test_phosphenes_epochs_shape() -> None:
    chans = json.loads(CONTRACT.read_text())["channels"]
    names = _cap(chans)[:-3]
    rng = np.random.default_rng(0)
    ep = mne.EpochsArray(rng.normal(size=(4, len(names), 20)) * 1e-6,
                         mne.create_info(names, 128.0, "eeg"), verbose="error")
    out, missing = complete_contract_channels(ep, chans)
    assert out.get_data().shape == (4, 64, 20) and len(missing) == 5
