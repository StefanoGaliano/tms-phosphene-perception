"""Step 2: the generic one-sample test (P§1)."""
import numpy as np

from start_vision.questions.stats import one_sample


def test_questions_one_sample_effect_and_null_rate() -> None:
    rng = np.random.default_rng(1)
    assert one_sample(rng.normal(0.8, 1, 20), 0.0, tail=1)["p"] < 0.05
    hits = sum(one_sample(rng.normal(0, 1, 20), 0.0, tail=1)["p"] < 0.05 for _ in range(200))
    assert abs(hits / 200 - 0.05) <= 0.035


def test_questions_one_sample_centre_dz_seed() -> None:
    x = np.array([0.55, 0.6, 0.52, 0.58, 0.61, 0.57, 0.54, 0.59, 0.56, 0.62, 0.53, 0.6, 0.58, 0.55])
    r = one_sample(x, 0.5, tail=1)
    d = x - 0.5
    assert np.isclose(r["d_z"], d.mean() / d.std(ddof=1)) and r["p"] < 0.001
    assert r == one_sample(x, 0.5, tail=1)            # reproducible with the default seed
    assert one_sample(x, 0.58, tail=2)["p"] > 0.05    # centre shift
    assert np.isnan(one_sample([np.nan, 1.0])["p"])   # n < 2
