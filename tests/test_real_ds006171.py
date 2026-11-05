"""Real-data checks on ds006171 (pytest -m real; need sourcedata/ds006171)."""

from __future__ import annotations

from pathlib import Path

import pytest

from start_vision import codes as C  # noqa: N812
from start_vision.stage import stage_trials

ROOT = Path(__file__).parents[1] / "sourcedata" / "ds006171"
BDF = ROOT / "sub-01" / "eeg" / "sub-01_task-nocue_eeg.bdf"

pytestmark = [
    pytest.mark.real,
    pytest.mark.skipif(not BDF.exists(), reason="ds006171 sub-01 not downloaded"),
]


def test_sub01_nocue_step4_acceptance() -> None:
    tr, _, dropped, sfreq = stage_trials(BDF, "nocue")
    assert sfreq == 1024 and dropped == []
    assert len(tr) == 400 and (tr["cls"] == C.EXCLUDED).sum() == 0
    n = tr["cls"].value_counts()
    assert n.get(C.SEEN, 0) + n.get(C.FA, 0) == 188  # 124 count
    assert n.get(C.UNSEEN, 0) + n.get(C.CR, 0) == 212  # 125 count
    d = tr["photodiode_delay_ms"]
    assert d.min() > 37.5 and d.max() < 56.5


def test_ds006171_sub23_status_glitch() -> None:
    bdf = ROOT / "sub-23" / "eeg" / "sub-23_task-nocue_eeg.bdf"
    if not bdf.exists():
        pytest.skip("sub-23 not downloaded")
    tr, _, dropped, _ = stage_trials(bdf, "nocue")
    assert dropped == [6] and len(tr) == 400
