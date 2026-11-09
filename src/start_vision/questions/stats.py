"""Generic one-sample test (P§1): sign flip, bootstrap CI, d_z; any centre, one or two tails."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from start_vision.decoding import stats as dstats
from start_vision.questions.params import N_BOOT, N_PERM, SEED


def one_sample(x: Any, centre: float = 0.0, tail: Literal[1, 2] = 2,
               seed: int = SEED) -> dict[str, float]:
    """``tail=1`` tests mean > centre; ``tail=2`` two-sided. Non-finite values are dropped."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    d = a - centre
    n = len(d)
    if n < 2:
        return {"n": float(n), "mean": float(a.mean()) if n else np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "d_z": np.nan, "p": np.nan, "tail": float(tail),
                "n_positive": float((d > 0).sum())}
    p = dstats._signflip_p(d, N_PERM, seed, tail=1 if tail == 1 else 0)
    lo, hi = dstats.bootstrap_ci(a, N_BOOT, seed)
    sd = d.std(ddof=1)
    return {"n": float(n), "mean": float(a.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "d_z": float(d.mean() / sd) if sd > 0 else np.nan, "p": float(p),
            "tail": float(tail), "n_positive": float((d > 0).sum())}
