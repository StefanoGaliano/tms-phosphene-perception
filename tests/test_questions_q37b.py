"""Step 8b: Q37b (P§7) - independence model, block AUC, k* on O1 then O2."""
import json
from pathlib import Path

import numpy as np

from start_vision.questions import q37b
from tests.synthetic.questions import gin_scores_table


def test_questions_q37b_predicted_identity_and_independence() -> None:
    assert np.isclose(q37b.predicted_auc(0.55, 1), 0.55)
    rng = np.random.default_rng(0)
    p1, p0 = rng.normal(0.18, 1, 3000), rng.normal(0, 1, 3000)   # single-pulse AUC ~0.55
    from sklearn.metrics import roc_auc_score
    a1 = roc_auc_score(np.r_[np.ones(3000), np.zeros(3000)], np.r_[p1, p0])  # all pulses, as
    # recording_table does (a resampled k = 1 estimate adds noise)
    for k in (5, 10, 20):
        assert abs(q37b.block_auc(p1, p0, k, 2000, rng) - q37b.predicted_auc(a1, k)) < 0.03


def test_questions_q37b_blocks_ignore_pulse_order() -> None:
    """Blocks are random draws within class (P§7, as the first look), so pulse order and any
    dependence between neighbouring pulses cannot enter: shuffling pulses leaves the result."""
    t = gin_scores_table(n_subjects=2, n_pulses=400, seed=3)
    a = q37b.recording_table(t)
    perm = t.copy()
    for _, g in t.groupby("recording"):
        for v in (0, 1):
            idx = g.index[g["seen"] == v]
            perm.loc[idx, "p_seen"] = np.random.default_rng(5).permutation(t.loc[idx, "p_seen"])
    b = q37b.recording_table(perm)
    assert np.allclose(a["auc1"], b["auc1"])
    assert (a["obs_20"] - b["obs_20"]).abs().max() < 0.03


def test_questions_q37b_kstar_written_before_o2(tmp_path: Path) -> None:
    per = q37b.recording_table(gin_scores_table(auc_shift=0.6, seed=4))
    k1 = q37b.choose_k_star(per[per["site"] == "O1"])
    res = q37b.o2_test(per, k1, tmp_path)
    assert json.loads((tmp_path / "k_star.json").read_text())["k_star"] == k1
    assert res["k_used"] == (k1 if k1 is not None else 20)
    o2 = per["site"] == "O2"
    shifted = per.copy()
    shifted.loc[o2, [c for c in per.columns if c.startswith("obs_")]] = 0.5
    assert q37b.choose_k_star(shifted[shifted["site"] == "O1"]) == k1   # O2 never moves k*
    none = per.copy()
    none.loc[:, [c for c in per.columns if c.startswith("obs_")]] = 0.55
    assert q37b.choose_k_star(none[none["site"] == "O1"]) is None
