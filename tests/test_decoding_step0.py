"""Phase 2 step 13: contract checks of Step 0."""

from __future__ import annotations

import numpy as np

from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from start_vision.decoding.step0 import contract_problems
from tests.synthetic.decoding import make_subject


def test_contract_problems_flags_each_violation() -> None:
    cfg = DecodingConfig()
    ts = make_subject("1", "null", 20, n_channels=128)
    assert contract_problems(ts, cfg, ts.ch_names) == []
    bad = TrialSet(ts.X.copy(), ts.times, ts.ch_names, ts.sfreq,
                   ts.meta.assign(rt_ms=np.nan), ts.info)
    bad.X[0, 0, 0] = np.nan
    p = contract_problems(bad, cfg, tuple(reversed(ts.ch_names)))
    assert "channel names/order differ" in p and "meta rt_ms has NaN" in p
    assert "non-finite data" in p
    small = make_subject("1", "null", 20, n_channels=32)
    assert "32 channels" in contract_problems(small, cfg, None)
