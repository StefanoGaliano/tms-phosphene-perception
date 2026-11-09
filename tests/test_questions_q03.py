"""Step 6: Q03 guessing mixture (P§4)."""
import numpy as np
import pandas as pd

from start_vision.questions import q03


def test_questions_q03_mixture_formula_edges() -> None:
    f, dp = q03.mixture(a=0.25 + 0.75 * 0.2, c=0.25, m_i=0.65, m_g=0.45)
    assert np.isclose(f, 0.2) and np.isclose(dp, 0.2 / (0.2 + 0.8 * 0.25) * 0.2)
    assert q03.mixture(0.2, 0.25, 0.6, 0.4) == (0.0, 0.0)
    assert np.isclose(q03.mixture(1.0, 0.25, 0.6, 0.4)[1], 0.2)


def _sessions(n: int, informed_mean: float, seed: int) -> pd.DataFrame:
    """Rating-1 trials: informed fraction 0.2 always correct (score informed_mean), guesses uniform
    over 4 quadrants (score 0.45); rating-2 correct trials score 0.65."""
    rng = np.random.default_rng(seed)
    rows = []
    for fp in range(n):
        m = 600
        informed = rng.random(m) < 0.2
        corr = np.where(informed, 1, (rng.random(m) < 0.25).astype(int))
        p = np.where(informed, informed_mean, 0.45) + rng.normal(0, 0.05, m)
        rows.append(pd.DataFrame({"gabor": "large", "fp": fp, "rating": 1, "obj_corr": corr,
                                  "p_seen": p}))
        rows.append(pd.DataFrame({"gabor": "large", "fp": fp, "rating": 2, "obj_corr": 1,
                                  "p_seen": 0.65 + rng.normal(0, 0.05, 100)}))
    return pd.concat(rows, ignore_index=True)


def test_questions_q03_criterion_account_recovered() -> None:
    per = q03.session_values(_sessions(14, 0.65, 1))
    assert abs(per["f"].mean() - 0.2) < 0.05
    assert q03.group(per)["primary"]["p"] > 0.05


def test_questions_q03_less_than_predicted_detected() -> None:
    per = q03.session_values(_sessions(14, 0.50, 2))
    r = q03.group(per)["primary"]
    assert r["mean"] < 0 and r["p"] < 0.05
    assert q03.interpretation(q03.group(per)).startswith("below")
