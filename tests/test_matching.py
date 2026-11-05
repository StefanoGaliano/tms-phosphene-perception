"""matching: equal counts per bin, small contrast difference, deterministic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from start_vision import codes as C  # noqa: N812
from start_vision.matching import balance_report, match


def _trials(seed: int = 1, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.01, 0.04, n)
    seen = rng.random(n) < (c - 0.01) / 0.03
    return pd.DataFrame({"cls": np.where(seen, C.SEEN, C.UNSEEN),
                         "side": rng.integers(1, 3, n), "gabor_contrast": c})


def test_matched_balanced() -> None:
    t = _trials()
    m = match(t)
    for _, g in t[m].groupby("side"):
        assert (g["cls"] == C.SEEN).sum() == (g["cls"] == C.UNSEEN).sum()
    b = balance_report(t, m)
    assert abs(b["smd_all"]) > 0.5 and abs(b["smd_matched"]) < 0.1


def test_deterministic_with_seed() -> None:
    t = _trials()
    assert match(t, seed=97).equals(match(t, seed=97))
    assert not match(t, seed=97).equals(match(t, seed=3))


def test_catch_never_matched() -> None:
    t = _trials()
    t.loc[:20, "cls"] = C.FA
    t.loc[:20, "gabor_contrast"] = np.nan
    assert not match(t).loc[:20].any()
