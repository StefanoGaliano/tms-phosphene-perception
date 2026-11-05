"""Group statistics, the same recipe as eegunify's group stage (``eegunify.erp.group``).

``paired``: t (``ttest_1samp`` on differences), Wilcoxon, ``d_z = mean / SD(ddof=1)``, percentile
bootstrap 95 % CI (10 000 resamples, seed 97; none below 8 subjects). ``cluster``: MNE
``spatio_temporal_cluster_1samp_test`` with EEG adjacency, as eegunify. Step 9 checks that they
reproduce eegunify's per-task numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy import stats

N_BOOT, SEED, MIN_BOOTSTRAP_N = 10_000, 97, 8  # eegunify group defaults


def _dz(x: np.ndarray, axis: int = -1) -> np.ndarray:
    sd = np.std(x, ddof=1, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.mean(x, axis=axis) / sd)


def paired(diff: Sequence[float] | np.ndarray | pd.Series) -> dict[str, float]:
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    out: dict[str, float] = {"n": n, "mean": float(x.mean()) if n else math.nan,
                             "t": math.nan, "p": math.nan, "W": math.nan, "p_w": math.nan,
                             "d_z": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    if n >= 2:
        r = stats.ttest_1samp(x, 0.0)
        out["t"], out["p"] = float(r.statistic), float(r.pvalue)
    if n >= 1 and np.any(x != 0):
        w = stats.wilcoxon(x)
        out["W"], out["p_w"] = float(w.statistic), float(w.pvalue)
    if n >= 2 and np.std(x, ddof=1) > 0:
        out["d_z"] = float(_dz(x))
        if n >= MIN_BOOTSTRAP_N:
            res = stats.bootstrap((x,), _dz, n_resamples=N_BOOT, method="percentile",
                                  random_state=np.random.default_rng(SEED), vectorized=True)
            out["ci_low"] = float(res.confidence_interval.low)
            out["ci_high"] = float(res.confidence_interval.high)
    return out


def cluster(contrast: Sequence[mne.Evoked], tmin_ms: float = 0, tmax_ms: float = 800,
            tail: int = 0, n_permutations: int = 5000) -> pd.DataFrame:
    evs = [e.copy().pick("eeg", exclude="bads").crop(tmin_ms / 1000, tmax_ms / 1000)
           for e in contrast]
    chans = evs[0].ch_names
    evs = [e.pick(chans) for e in evs]
    x = np.stack([e.data.T for e in evs])
    adjacency, _ = mne.channels.find_ch_adjacency(evs[0].info, ch_type="eeg")
    n = len(evs)
    t_obs, clusters, pvals, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        x, n_permutations=min(n_permutations, 2**n), tail=tail, adjacency=adjacency,
        seed=SEED, out_type="indices", verbose=False)
    times = evs[0].times * 1000
    rows: list[dict[str, Any]] = []
    for k, (cl, p) in enumerate(zip(clusters, pvals, strict=True)):
        ti, ci = cl
        rows.append({"cluster": k + 1, "tmin_ms": float(times[np.min(ti)]),
                     "tmax_ms": float(times[np.max(ti)]),
                     "channels": "+".join(sorted({chans[i] for i in ci})),
                     "t_sum": float(t_obs[ti, ci].sum()), "p": float(p), "n_subjects": n})
    cols = ["cluster", "tmin_ms", "tmax_ms", "channels", "t_sum", "p", "n_subjects"]
    return pd.DataFrame(rows, columns=cols).sort_values("p", kind="stable").reset_index(drop=True)
