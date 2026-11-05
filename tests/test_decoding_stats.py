"""Phase 2 build step 6: cluster FPR <= 5 % (±2 %) on 200 null simulations; detects effects."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps
from scipy.ndimage import gaussian_filter1d

from start_vision.decoding import stats


def _auc_null(rng: np.random.Generator, n_sub: int = 20, n_t: int = 64) -> np.ndarray:
    # smooth subject AUC courses symmetric around 0.5 (as AUC under H0)
    return 0.5 + 0.03 * gaussian_filter1d(rng.standard_normal((n_sub, n_t)), 2, axis=1) * 3


def test_cluster_fpr_on_200_null_simulations() -> None:
    rng = np.random.default_rng(0)
    hits = sum(
        (stats.cluster_auc(_auc_null(rng), n_permutations=200, seed=i)["p"] < 0.05).any()
        for i in range(200)
    )
    assert hits / 200 <= 0.07, hits


def test_cluster_finds_planted_effect_and_its_span() -> None:
    rng = np.random.default_rng(1)
    times = np.arange(64) / 64
    auc = _auc_null(rng)
    auc[:, 30:40] += 0.06
    cl = stats.cluster_auc(auc, times, n_permutations=500)
    top = cl.iloc[0]
    assert top.p < 0.05 and top.t_start <= times[31] and top.t_end >= times[38]
    mask = stats.cluster_mask(auc, n_permutations=500)
    assert mask[30:40].mean() > 0.8 and not mask[:20].any()


def test_cluster_2d_and_tfce() -> None:
    rng = np.random.default_rng(2)
    auc = 0.5 + 0.03 * rng.standard_normal((15, 20, 20))
    auc[:, 8:14, 8:14] += 0.08
    cl = stats.cluster_auc(auc, n_permutations=300)
    assert cl.iloc[0].p < 0.05 and cl.iloc[0].extent >= 20
    p = stats.tfce_auc(auc[:, 10], n_permutations=300)
    assert p[8:14].max() < 0.05 and p.shape == (20,)


def test_window_test_exact_and_ci() -> None:
    a = np.array([0.55, 0.6, 0.52, 0.58, 0.61, 0.57])
    r = stats.window_test(a, n_permutations=10_000)
    assert r["p"] == pytest.approx(1 / 64)  # exact: all 2^6 sign patterns, all positive
    assert r["ci_low"] < r["mean"] < r["ci_high"] and r["mean"] == pytest.approx(a.mean())
    null = stats.window_test(np.random.default_rng(3).normal(0.5, 0.05, 30))
    assert null["p"] > 0.01


def test_paired_tost_holm_rankbiserial() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(0.6, 0.02, 30)
    assert stats.paired_test(a, a - 0.05)["p"] < 0.001
    eq = stats.tost(rng.normal(0, 0.01, 30), 0.03)
    assert eq["p"] < 0.05
    assert stats.tost(rng.normal(0.05, 0.01, 30), 0.03)["p"] > 0.5
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    d = np.array([1.0, 2, 3, -0.5])
    w = sps.wilcoxon(d)
    assert stats.rank_biserial(d) == pytest.approx((10 - 2 * w.statistic) / 10)


def test_label_permutation_chance() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 200)
    x = y + rng.normal(0, 1, 200)

    def score(lab: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(lab, x))

    r = stats.label_permutation(score, y, strata=np.zeros(200), n_permutations=200)
    assert r["p"] < 0.01 and abs(r["null_mean"] - 0.5) < 0.02
