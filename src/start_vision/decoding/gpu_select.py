"""GPU fits for the A11 window x C selection (build step 15, 2026-11-06).

Same model as ``models.window_detector`` on precomputed window features: StandardScaler fitted
on the training fold, then L2 logistic regression with balanced class weights. The objective is
liblinear's primal, intercept included as a penalized constant column (``intercept_scaling=1``):

    min_theta  0.5 * ||theta||^2 + C * sum_i s_i * log(1 + exp(-y_i * theta . [x_i, 1]))

It is strictly convex, so the exact minimizer is unique; here it is found by damped Newton steps
(full Hessian, Cholesky) on the GPU. liblinear stops at its tolerance (1e-4), so scores differ from
it in the last digits; ``tests/test_decoding_gpu_select.py`` bounds the difference.
Only the selection uses this path. The exported detector is still fitted by sklearn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _torch() -> Any:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # full fp32 matmuls
    return torch


def device() -> str:
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit_logistic(X: Any, y: Any, C: float, tol: float = 1e-4, max_iter: int = 50) -> Any:
    """Standardize X (training stats), fit balanced L2 LR; return (mean, sd, theta).

    Stops when ||grad|| <= tol * ||grad at 0|| (liblinear's rule, eps 1e-4), or when a Newton
    step no longer lowers the objective (the float32 precision floor; fixed 2026-11-07: an
    absolute 1e-6 rule was below fp32 precision, so every fit ran all ``max_iter`` steps).
    """
    torch = _torch()
    mu = X.mean(0)
    sd = X.std(0, unbiased=False)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))  # sklearn: zero variance -> scale 1
    xa = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], 1)
    n1 = y.sum()
    n0 = len(y) - n1
    s = torch.where(y > 0, len(y) / (2 * n1), len(y) / (2 * n0)) * C  # balanced weights x C
    ys = 2 * y - 1
    theta = torch.zeros(xa.shape[1], device=X.device, dtype=X.dtype)
    eye = torch.eye(xa.shape[1], device=X.device, dtype=X.dtype)

    def objective(th: Any) -> Any:
        return 0.5 * th @ th + (s * torch.nn.functional.softplus(-ys * (xa @ th))).sum()

    f = objective(theta)
    g0 = None
    for it in range(max_iter):
        m = ys * (xa @ theta)
        sig = torch.sigmoid(-m)  # d/dm log(1+exp(-m)) = -sig
        g = theta - xa.T @ (s * sig * ys)
        gn = float(g.norm())
        g0 = gn if g0 is None else g0
        if gn <= tol * g0:
            break
        d = s * sig * (1 - sig)
        h = eye + xa.T @ (xa * d[:, None])
        step = torch.cholesky_solve(g[:, None], torch.linalg.cholesky(h))[:, 0]
        t = 1.0
        while True:  # backtracking (Armijo) keeps every step a decrease
            new = theta - t * step
            fn = objective(new)
            if fn <= f - 1e-4 * t * (g @ step):
                break
            t /= 2
            if t < 1e-3:  # no decrease left: at the precision floor
                fit_logistic.last_iterations = it  # type: ignore[attr-defined]
                return mu, sd, theta
        theta, f = new, fn
    fit_logistic.last_iterations = it  # type: ignore[attr-defined]
    return mu, sd, theta


def fit_score(C: float, fs: Sequence[np.ndarray], ys: Sequence[np.ndarray],
              train: Sequence[int], test: Sequence[int], cache: dict[str, Any]
              ) -> dict[int, float]:
    """GPU twin of ``export._fit_score``: train on subjects ``train``, AUC per ``test``."""
    torch = _torch()
    if "X" not in cache:  # one upload per feature set
        dev = device()
        cache["X"] = [torch.tensor(f, dtype=torch.float32, device=dev) for f in fs]
        cache["y"] = [torch.tensor(y, dtype=torch.float32, device=dev) for y in ys]
    x = torch.cat([cache["X"][k] for k in train])
    y = torch.cat([cache["y"][k] for k in train])
    mu, sd, theta = fit_logistic(x, y, C)
    out = {}
    for k in test:
        score = ((cache["X"][k] - mu) / sd) @ theta[:-1] + theta[-1]
        out[k] = float(roc_auc_score(ys[k], score.double().cpu().numpy()))
    return out
