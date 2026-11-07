"""Eklund & Wiens 2018 staging (A12 issue #1): Status decoding, labels, duration rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start_vision import eklund2018 as ek

SF = 512.0
INV = {v: k for k, v in ek.TTL_TO_CONDITION.items()}


def _status(conditions: list[int], delay: int = 4, dur: int = 28) -> np.ndarray:
    """A Status channel as BioSemi writes it: a short TTL pulse (ends before the photodiode
    flanks, as in the recordings), then Light1 while the stimulus is on, then Light2."""
    st = np.zeros(len(conditions) * 400 + 2000, dtype=np.int64)
    t = 500
    for c in conditions:
        ttl = INV[c] << 10
        st[t:t + 2] |= ttl  # 2-sample TTL pulse
        if c != 10:
            st[t + delay:t + delay + dur] |= 1  # Light1 while the stimulus is on
            st[t + delay + dur:t + delay + dur + 5] |= 2  # Light2 at offset
        t += 400
    return st


def test_status_decoding_follows_trialfun() -> None:
    conds = [10, 1, 9, 5, 2, 10, 3, 4, 8, 10, 6, 7, 10, 9, 10]
    t = ek.find_trials(ek.status_events(_status(conds), SF), SF)
    assert list(t["condition"]) == [c for c in conds if c != 10]
    assert np.allclose(t["delay_ms"], 4 / SF * 1000)
    assert np.allclose(t["duration_ms"], 28 / SF * 1000)
    assert np.allclose(t["sample"] - t["code_sample"], 4)


def test_ds006171_eklund_wrong_block_count_raises() -> None:
    with pytest.raises(ek.EklundError, match="block codes"):
        ek.find_trials(ek.status_events(_status([10, 1, 2]), SF), SF)


def test_labels_match_analyze_van() -> None:
    t = pd.DataFrame({
        "condition": [1, 2, 3, 4, 5, 9, 1],
        "obj_corr": [1, 1, 0, 0, 1, 3, 1],
        "Subj_resp": [3, 1, 1, 2, 3, 1, 2],
        "duration_ms": [54.7, 54.7, 54.7, 54.7, 117.2, 54.7, 60.0],
    })
    out = ek.label(t)
    assert list(out["cls"]) == ["aware", "unaware", "unaware_incorrect", "aware_incorrect",
                                "control", "catch", "aware"]
    assert list(out["quadrant"]) == ["UL", "UR", "LL", "LR", "UL", "none", "UL"]
    # condition 1 mode is 54.7 ms; the 60 ms trial is >= 2 ms off -> excluded
    assert list(out["exclusion"]) == ["", "", "", "", "", "", "duration"]


def test_event_codes_round_trip() -> None:
    m = ek.eegunify_mapping()
    assert m["11"] == "stimulus/aware/left/UL" and m["24"] == "stimulus/unaware/right/LR"
    assert m["60"] == "stimulus/catch/none/none"
    assert len(set(m.values())) == len(m)
