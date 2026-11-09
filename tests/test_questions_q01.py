"""Step 4: Q01 (P§2) - the rating slope survives / vanishes with a duration confound."""
import numpy as np

from start_vision.questions import q01
from tests.synthetic.questions import eklund_table


def test_questions_q01_rating_effect_recovered() -> None:
    per = q01.session_slopes(eklund_table(rating_effect=0.08, seed=1))
    r = q01.group(per)
    assert r["primary"]["p"] < 0.05 and r["primary"]["mean"] > 0.05
    assert q01.interpretation(r).startswith("graded")


def test_questions_q01_duration_confound_removed() -> None:
    per = q01.session_slopes(eklund_table(rating_effect=0.0, duration_effect=0.08,
                                          duration_follows_rating=True, seed=2))
    r = q01.group(per)
    assert r["no_covariate"]["mean"] > 0.05 and r["no_covariate"]["p"] < 0.05
    assert abs(r["primary"]["mean"]) < 0.02 and r["primary"]["p"] > 0.05
    assert q01.interpretation(r).startswith("gradation attributed to duration")


def test_questions_q01_eligibility_and_steps() -> None:
    t = eklund_table(n_sessions=3, seed=3)
    t = t[~((t.fp == 1) & (t.rating >= 2))]      # session 1: one rating level only
    t = t[~((t.fp == 2) & (t.rating == 3))]      # session 2: no rating 3
    per = q01.session_slopes(t).set_index("fp")
    assert not per.loc[1, "eligible"] and np.isnan(per.loc[1, "slope_cov"])
    assert per.loc[2, "eligible"] and np.isnan(per.loc[2, "step_3_minus_2"])
