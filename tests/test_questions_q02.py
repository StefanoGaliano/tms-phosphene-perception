"""Step 5: Q02 (P§3) - correctness slope (Eklund) and contrast-residualized AUC (ds006171)."""
import numpy as np

from start_vision.questions import q02
from tests.synthetic.questions import ds006171_table, eklund_table


def test_questions_q02_slope_recovered_and_null() -> None:
    per = q02.eklund_slopes(eklund_table(correct_effect=0.1, seed=4))
    assert abs(per["slope_cov"].mean() - 0.1) < 0.03
    assert q02.eklund_group(per)["p"] < 0.05
    null = q02.eklund_slopes(eklund_table(correct_effect=0.0, seed=5))
    assert q02.eklund_group(null)["p"] > 0.05


def test_questions_q02_wrong_cell_minimum() -> None:
    t = eklund_table(n_sessions=2, seed=6)
    drop = t.index[(t.fp == 1) & (t.rating >= 2) & (t.obj_corr == 0)][2:]
    per = q02.eklund_slopes(t.drop(drop)).set_index("fp")
    assert np.isnan(per.loc[1, "slope_cov"]) and np.isfinite(per.loc[2, "slope_cov"])


def test_questions_q02_ds006171_contrast_only_stays_null() -> None:
    t = ds006171_table(contrast_auc_only=True, seed=7)
    t["correct"] = np.where(t["correct"].notna(), (t["gabor_contrast"] > 0.03).astype(float),
                            np.nan)   # correctness follows contrast only
    per = q02.ds006171_auc(t)
    assert per["auc_raw"].mean() > 0.6
    assert abs(per["auc_resid"].mean() - 0.5) < 0.05
