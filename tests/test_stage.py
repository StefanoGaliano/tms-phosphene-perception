"""stage: photodiode onsets exact to the sample, exclusions counted, alignment checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start_vision import codes as C  # noqa: N812
from start_vision.stage import StageError, align, build_trials, drop_unknown, staged_events
from tests.synthetic.vision import make_event_stream


@pytest.mark.parametrize("task", C.TASKS)
def test_photodiode_onsets_exact(task: str) -> None:
    ev, truth = make_event_stream(task=task, missing_photodiode=(5, 17), missing_report=(9,))
    tr = build_trials(ev, task)
    assert len(tr) == len(truth)
    for r, t in zip(tr.itertuples(), truth, strict=True):
        assert r.raw_code == t.raw_code
        if t.photodiode_s is None:
            assert np.isnan(r.onset_s)
        else:
            assert round(r.onset_s * 1024) == round(t.photodiode_s * 1024)
            assert 30 <= r.photodiode_delay_ms <= 60 + 1e-6
    excl = tr[tr["cls"] == C.EXCLUDED]
    assert sorted(excl["trial_number"]) == [5, 9, 17]
    assert set(excl["exclusion"]) == {"no_photodiode", "no_report"}


def test_classes_and_orientation() -> None:
    ev, truth = make_event_stream(n_trials=200)
    tr = build_trials(ev, "nocue")
    for r, t in zip(tr.itertuples(), truth, strict=True):
        rs = C.RAW_STIMULI["nocue"][t.raw_code]
        seen = t.report == C.REPORT_SEEN
        want = (C.FA if seen else C.CR) if rs.catch else (C.SEEN if seen else C.UNSEEN)
        assert r.cls == want
        if t.orient_response is not None:
            assert r.objective_correct == (t.orient_response == rs.orient)


def test_report_contradicting_subjective_r_is_error() -> None:
    ev, _ = make_event_stream()
    i = ev.index[ev["code"].isin([11, 12, 21, 22]) & ev["subjective_r"].notna()][0]
    ev.loc[i, "subjective_r"] = 1 - ev.loc[i, "subjective_r"]
    with pytest.raises(StageError, match="subjective_r"):
        build_trials(ev, "nocue")


def _pair(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    status = ev[["onset_s", "code"]].copy()
    status.attrs["sfreq"] = 1024.0
    bids = ev.rename(columns={"onset_s": "onset"}).copy()
    return status, bids


def test_ds006171_events_tsv_onset_precision() -> None:
    # events.tsv rounds onsets to 6 significant digits: 10 ms steps after 1000 s
    ev, _ = make_event_stream()
    status, bids = _pair(ev)
    bids["onset"] = (bids["onset"] + 1000).map(lambda x: float(f"{x:.6g}")) - 1000
    assert align(status, bids)["onset_s"].equals(status["onset_s"])
    bids.loc[3, "onset"] += 0.006
    with pytest.raises(StageError, match="onset"):
        align(status, bids)


def test_ds006171_status_glitch_code_dropped() -> None:
    # sub-23 nocue: Status has one code 6, 1 ms before a 22; events.tsv does not
    ev, _ = make_event_stream()
    status, bids = _pair(ev)
    glitch = pd.DataFrame({"onset_s": [status["onset_s"].iloc[10] - 0.001], "code": [6]})
    status = pd.concat([status.iloc[:10], glitch, status.iloc[10:]], ignore_index=True)
    with pytest.raises(StageError, match="code sequences differ"):
        align(status, bids)
    clean, dropped = drop_unknown(status, "nocue")
    assert dropped == [6]
    align(clean, bids)


def test_staged_events_map_completely() -> None:
    ev, _ = make_event_stream(task="noninformative", missing_photodiode=(4,))
    tr = build_trials(ev, "noninformative")
    tr["matched"] = False
    out = staged_events(tr, ev, "noninformative")
    m = C.eegunify_mapping()
    assert all(str(v) in m for v in out["value"])
    assert C.PHOTODIODE not in set(out["value"])
    assert out["onset"].is_monotonic_increasing
