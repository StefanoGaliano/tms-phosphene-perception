"""Group statistics for decoding (PHASE2 D6, D7, §8; build step 6).

The subject is the unit: inputs are one AUC (or AUC time course / matrix) per subject.
Under H0 the subject AUC is symmetric around 0.5, so the group null is a sign flip of AUC - 0.5.

- ``cluster_auc``: sign-flip cluster permutation (1D time courses or 2D generalization matrices),
  cluster-forming threshold = t at p (one-sided, df = n - 1), tail +1 (AUC > 0.5).
- ``tfce_auc``: the same with TFCE (sensitivity analysis).
- ``window_test``: one AUC per subject vs 0.5: sign-flip p, t, d_z, bootstrap CI of the mean.
- ``paired_test``: A vs B paired (two-sided sign flip on the difference).
- ``tost``: equivalence of a paired difference within ±bound (two one-sided t tests).
- ``holm``: Holm step-down adjusted p-values.
- ``label_permutation``: empirical chance of a per-subject score by shuffling labels within
  strata (D7).
- ``rank_biserial``: effect size for Wilcoxon signed-rank (catch FA vs CR).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

CHANCE = 0.5


def _as_mask(cluster: Any, shape: tuple[int, ...]) -> np.ndarray:
    """MNE clusters are boolean masks (nD) or tuples of slices (1D): both index an array."""
    m = np.zeros(shape, dtype=bool)
    m[cluster] = True
    return m


def _t_threshold(n: int, p: float) -> float:
    return float(stats.t.ppf(1 - p, n - 1))


def cluster_auc(auc: np.ndarray, times: np.ndarray | None = None, n_permutations: int = 10_000,
                p: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Clusters of AUC > 0.5. ``auc``: (n_subjects, n_times) or (n_subjects, n_t, n_t).

    One row per cluster: p, mass (sum of t), extent in samples, and for 1D the first/last time.
    """
    x = np.asarray(auc, dtype=float) - CHANCE
    n = x.shape[0]
    t_obs, clusters, p_vals, _ = permutation_cluster_1samp_test(
        x, threshold=_t_threshold(n, p), n_permutations=n_permutations, tail=1,
        rng=seed, out_type="mask", verbose="error",
    )
    rows: list[dict[str, Any]] = []
    for mask, pv in zip(clusters, p_vals, strict=True):
        m = _as_mask(mask, t_obs.shape)
        row: dict[str, Any] = {"p": float(pv), "mass": float(t_obs[m].sum()),
                               "extent": int(m.sum()), "mean_auc": float(auc[:, m].mean())}
        if x.ndim == 2 and times is not None:
            idx = np.flatnonzero(m)
            row["t_start"], row["t_end"] = float(times[idx[0]]), float(times[idx[-1]])
        rows.append(row)
    cols = ["p", "mass", "extent", "mean_auc", "t_start", "t_end"]
    df = pd.DataFrame(rows, columns=cols if x.ndim == 2 else cols[:4])
    return df.sort_values("p", ignore_index=True)


def cluster_mask(auc: np.ndarray, alpha: float = 0.05, **kw: Any) -> np.ndarray:
    """Boolean map of samples inside clusters with p < alpha."""
    x = np.asarray(auc, dtype=float) - CHANCE
    n = x.shape[0]
    _, clusters, p_vals, _ = permutation_cluster_1samp_test(
        x, threshold=_t_threshold(n, kw.get("p", 0.05)),
        n_permutations=kw.get("n_permutations", 10_000), tail=1, rng=kw.get("seed", 0),
        out_type="mask", verbose="error",
    )
    out = np.zeros(x.shape[1:], dtype=bool)
    for mask, pv in zip(clusters, p_vals, strict=True):
        if pv < alpha:
            out |= _as_mask(mask, out.shape)
    return out


def tfce_auc(auc: np.ndarray, n_permutations: int = 10_000, seed: int = 0,
             step: float = 0.2) -> np.ndarray:
    """Per-sample p-values with TFCE (sensitivity analysis, D6)."""
    x = np.asarray(auc, dtype=float) - CHANCE
    _, _, p_vals, _ = permutation_cluster_1samp_test(
        x, threshold={"start": 0.0, "step": step}, n_permutations=n_permutations, tail=1,
        rng=seed, verbose="error",
    )
    return np.asarray(p_vals).reshape(x.shape[1:])


def _signflip_p(x: np.ndarray, n_permutations: int, seed: int, tail: int) -> float:
    """Sign-flip permutation p of the mean of ``x`` (exact when 2^n <= n_permutations)."""
    n = len(x)
    obs = x.mean()
    if 2**n <= n_permutations:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(n_permutations, n))
        signs[0] = 1  # include the observed labelling
    null = (signs * x).mean(1)
    if tail == 1:
        return float((null >= obs - 1e-12).mean())
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())


def bootstrap_ci(x: np.ndarray, n_bootstrap: int = 10_000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_bootstrap, len(x)), replace=True).mean(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def _summary(x: np.ndarray, p: float, n_bootstrap: int, seed: int) -> dict[str, float]:
    n = len(x)
    sd = x.std(ddof=1) if n > 1 else np.nan
    t = x.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    lo, hi = bootstrap_ci(x, n_bootstrap, seed) if n > 1 else (np.nan, np.nan)
    return {"n": n, "mean": float(x.mean()), "t": float(t), "p": p,
            "d_z": float(x.mean() / sd) if n > 1 and sd > 0 else np.nan,
            "ci_low": lo, "ci_high": hi}


def window_test(auc: Sequence[float] | np.ndarray, n_permutations: int = 10_000,
                n_bootstrap: int = 10_000, seed: int = 0) -> dict[str, float]:
    """One AUC per subject vs 0.5 (one-sided sign flip). ``mean``/CI are of AUC; d_z of AUC-0.5."""
    a = np.asarray(auc, dtype=float)
    a = a[np.isfinite(a)]
    x = a - CHANCE
    out = _summary(x, _signflip_p(x, n_permutations, seed, tail=1), n_bootstrap, seed)
    out["mean"] += CHANCE
    out["ci_low"] += CHANCE
    out["ci_high"] += CHANCE
    return out


def paired_test(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray,
                n_permutations: int = 10_000, n_bootstrap: int = 10_000,
                seed: int = 0) -> dict[str, float]:
    """A - B per subject (two-sided sign flip); mean/CI are of the difference."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    return _summary(d, _signflip_p(d, n_permutations, seed, tail=0), n_bootstrap, seed)


def tost(diff: Sequence[float] | np.ndarray, bound: float) -> dict[str, float]:
    """Equivalence of the mean paired difference within ±bound: p = max of the two one-sided t."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    p_low = 1 - stats.t.cdf((d.mean() + bound) / se, n - 1)  # H0: mean <= -bound
    p_high = stats.t.cdf((d.mean() - bound) / se, n - 1)  # H0: mean >= +bound
    return {"n": n, "mean": float(d.mean()), "bound": bound, "p_low": float(p_low),
            "p_high": float(p_high), "p": float(max(p_low, p_high))}


def holm(p: Sequence[float] | np.ndarray) -> np.ndarray:
    pv = np.asarray(p, dtype=float)
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    return adj


def label_permutation(score: Callable[[np.ndarray], float], y: np.ndarray,
                      strata: np.ndarray, n_permutations: int = 200,
                      seed: int = 0) -> dict[str, Any]:
    """Empirical chance (D7): ``score(y)`` vs ``score(y shuffled within strata)``."""
    rng = np.random.default_rng(seed)
    obs = score(y)
    null = np.empty(n_permutations)
    for k in range(n_permutations):
        yp = y.copy()
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            yp[idx] = rng.permutation(y[idx])
        null[k] = score(yp)
    return {"observed": float(obs), "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "p": float((1 + (null >= obs).sum()) / (1 + n_permutations)), "null": null}


def rank_biserial(diff: Sequence[float] | np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation of a Wilcoxon signed-rank test."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if not len(d):
        return float("nan")
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())
