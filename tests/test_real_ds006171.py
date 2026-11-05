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


def test_ds006171_epochs_metadata_join_real() -> None:
    import mne

    from start_vision.trials import attach_metadata, staged_events

    f = (Path(__file__).parents[1] / "runs/nocue/derivatives/eegunify-epochs/sub-03/eeg"
         / "sub-03_task-nocue_desc-stimraw_epo.fif")
    if not f.exists():
        pytest.skip("runs/nocue sub-03 not built")
    ep = mne.read_epochs(f, preload=False, verbose="error")
    m = attach_metadata(ep, staged_events(ROOT.parent / "staged" / "ds006171", "03", "nocue"))
    assert len(m.metadata) == len(ep) == 308


@pytest.mark.parametrize(("sub", "task"), [("03", "nocue"), ("01", "noninformative")])
def test_ds006171_phase2_trialset_contract(sub: str, task: str) -> None:
    from start_vision.decoding.config import DecodingConfig
    from start_vision.decoding.data import epochs_path, load_trialset

    runs = Path(__file__).parents[1] / "runs"
    if not epochs_path(runs, sub, task).exists():
        pytest.skip(f"runs/{task} sub-{sub} not built")
    ts = load_trialset(runs, ROOT.parent / "staged" / "ds006171", sub, task, DecodingConfig())
    assert ts.X.shape[1:] == (128, 192) and ts.sfreq == 128
    assert ts.meta.loc[ts.meta.stim_present, "gabor_contrast"].notna().all()
    if task == "noninformative":
        assert (ts.meta.hemifield == "unknown").all()
