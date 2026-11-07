"""Cross-dataset test on Eklund & Wiens 2018: session AUCs and the group test (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from start_vision.decoding import eklund_test as et
from start_vision.decoding.config import DecodingConfig

CFG = DecodingConfig(n_bootstrap=2000)


def _session(shift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cls = np.array(["aware"] * 120 + ["unaware"] * 80 + ["unaware_incorrect"] * 60
                   + ["catch"] * 50 + ["control"] * 40)
    cond = np.r_[np.full(260, 1), np.full(50, 9), np.full(40, 5)]
    rating = np.r_[np.full(120, 3), np.ones(140), np.r_[np.full(5, 2), np.ones(45)],
                   np.full(40, 3)]
    score = rng.normal(size=len(cls)) + shift * (cls == "aware")
    return pd.DataFrame({"cls": cls, "condition": cond, "rating": rating, "score": score})


def test_session_aucs_use_the_authors_contrast() -> None:
    r = et.session_aucs(_session(1.0, 0))
    assert r["n_aware"] == 120 and r["n_unaware"] == 80 and r["n_catch_fa"] == 5
    assert 0.7 < r["auc_primary"] < 0.85  # d = 1 -> AUC ~ 0.76
    assert "auc_catch_fa_vs_cr" in r


def test_group_test_null_and_effect() -> None:
    rng = np.random.default_rng(1)
    null = et.group_test(0.5 + rng.normal(0, 0.03, 24), CFG)
    eff = et.group_test(0.6 + rng.normal(0, 0.03, 24), CFG)
    assert null["p_one_sided"] > 0.01 and eff["p_one_sided"] < 0.001
    assert eff["ci_low"] > 0.5 and eff["n_above_chance"] >= 22
