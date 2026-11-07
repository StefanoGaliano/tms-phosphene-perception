"""GPU selection path (build step 15): same fits as sklearn liblinear, same choices."""

from __future__ import annotations

import numpy as np
import pytest

from start_vision.decoding import export as ex
from start_vision.decoding.config import DecodingConfig
from tests.synthetic.decoding import make_synthetic_trialset

torch = pytest.importorskip("torch")
CFG = DecodingConfig(n_jobs=1, C_grid=(0.01, 0.1, 1.0))


def test_gpu_logistic_matches_liblinear_coefficients() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from start_vision.decoding import gpu_select

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 40)) * rng.uniform(0.5, 3, 40) + 2.0
    y = (x[:, :3].sum(1) + rng.normal(size=600) > 6.5).astype(float)  # imbalanced
    for c in CFG.C_grid:
        ref = LogisticRegression(solver="liblinear", C=c, class_weight="balanced",
                                 tol=1e-10, max_iter=10_000).fit(
            StandardScaler().fit_transform(x), y)
        _, _, th = gpu_select.fit_logistic(torch.as_tensor(x, dtype=torch.float64),
                                           torch.as_tensor(y, dtype=torch.float64), c,
                                           tol=1e-10)
        got = th.cpu().numpy()
        np.testing.assert_allclose(got[:-1], ref.coef_[0], atol=1e-6)
        np.testing.assert_allclose(got[-1], ref.intercept_[0], atol=1e-6)


def test_gpu_nested_selection_matches_sklearn() -> None:
    sets = make_synthetic_trialset(5, 90, generator="seen_late", d=0.6, seed=11)
    catch = [None] * 5
    wins = [(0.2, 0.6), (0.3, 0.8)]
    ref = ex.nested_selection(sets, catch, CFG, wins)
    gpu = ex.nested_selection(sets, catch, CFG, wins, backend="torch")
    assert gpu["window_s"] == ref["window_s"] and gpu["C"] == ref["C"]
    assert gpu["nested_choices"] == ref["nested_choices"]
    np.testing.assert_allclose(gpu["nested_auc"], ref["nested_auc"], atol=2e-3)
    for a, b in zip(gpu["grid"], ref["grid"], strict=True):
        assert abs(a["loso_auc_mean"] - b["loso_auc_mean"]) < 2e-3


def test_ds006171_gpu_logistic_converges_in_float32() -> None:
    """Bug 2026-11-07: an absolute 1e-6 gradient rule is below fp32 precision, so every fit
    ran all 50 Newton steps (full variant, 1536 features: ~1.9 s per fit, ~4 h per variant)."""
    from start_vision.decoding import gpu_select

    rng = np.random.default_rng(1)
    dev = gpu_select.device()
    x = torch.tensor(rng.normal(size=(6000, 1536)), dtype=torch.float32, device=dev)
    y = torch.tensor(rng.random(6000) < 0.5, dtype=torch.float32, device=dev)
    x[:, :5] += y[:, None] * 0.3
    for c in (0.01, 1.0):
        _, _, th32 = gpu_select.fit_logistic(x, y, c)
        assert gpu_select.fit_logistic.last_iterations < 15  # type: ignore[attr-defined]
        _, _, th64 = gpu_select.fit_logistic(x.double(), y.double(), c, tol=1e-10)
        np.testing.assert_allclose(th32.double().cpu().numpy(), th64.cpu().numpy(),
                                   atol=2e-3)
