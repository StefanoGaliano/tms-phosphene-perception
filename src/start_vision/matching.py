"""Contrast matching of seen and unseen Gabor trials (PHASE1_BLUEPRINT D7).

Within one recording (subject x task) and hemifield, Gabor trials are binned by
``gabor_contrast`` (deciles of the pooled seen + unseen contrasts). In each bin the larger class
is randomly subsampled to the size of the smaller one. The result is a boolean per trial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from start_vision.codes import SEEN, UNSEEN


def match(trials: pd.DataFrame, n_bins: int = 10, seed: int = 97) -> pd.Series:
    """``True`` for trials in the matched subset (columns ``cls``, ``side``, ``gabor_contrast``).

    Only Gabor trials (seen/unseen) with a finite contrast can be matched; the rest are False.
    """
    rng = np.random.default_rng(seed)
    out = pd.Series(False, index=trials.index)
    gabor = trials[trials["cls"].isin([SEEN, UNSEEN]) & np.isfinite(trials["gabor_contrast"])]
    for _, g in gabor.groupby("side", sort=True):
        bins = pd.qcut(g["gabor_contrast"], q=n_bins, labels=False, duplicates="drop")
        for b in sorted(bins.unique()):
            in_bin = g[bins == b]
            s = in_bin.index[in_bin["cls"] == SEEN]
            u = in_bin.index[in_bin["cls"] == UNSEEN]
            n = min(len(s), len(u))
            if n == 0:
                continue
            keep = [*rng.choice(s, n, replace=False), *rng.choice(u, n, replace=False)]
            out.loc[keep] = True
    return out


def _smd(a: pd.Series, b: pd.Series) -> float:
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def balance_report(trials: pd.DataFrame, matched: pd.Series) -> dict[str, float]:
    """Mean contrast per class and the standardized difference (seen - unseen), before/after."""
    out: dict[str, float] = {}
    for tag, sel in (("all", trials["cls"].isin([SEEN, UNSEEN])), ("matched", matched)):
        t = trials[sel]
        s = t.loc[t["cls"] == SEEN, "gabor_contrast"]
        u = t.loc[t["cls"] == UNSEEN, "gabor_contrast"]
        out[f"n_seen_{tag}"] = float(len(s))
        out[f"n_unseen_{tag}"] = float(len(u))
        out[f"contrast_seen_{tag}"] = float(s.mean()) if len(s) else float("nan")
        out[f"contrast_unseen_{tag}"] = float(u.mean()) if len(u) else float("nan")
        out[f"smd_{tag}"] = _smd(s, u) if len(s) > 1 and len(u) > 1 else float("nan")
    return out
