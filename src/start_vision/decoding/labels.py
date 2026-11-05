"""Trial sets, contrast bins and stratification keys (PHASE2 §6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from start_vision.decoding.data import TrialSet


def main_set(ts: TrialSet) -> TrialSet:
    """Stimulus-present trials; target ``seen``."""
    return ts.select(ts.meta["stim_present"].to_numpy(dtype=bool))


def catch_set(ts: TrialSet) -> TrialSet:
    """Stimulus-absent trials; never used for training."""
    return ts.select(~ts.meta["stim_present"].to_numpy(dtype=bool))


def contrast_bins(meta: pd.DataFrame, n_bins: int = 3, min_per_bin: int = 10) -> pd.Series:
    """Quantile bins of ``gabor_contrast`` per subject x task (stimulus-present rows).

    Starts with ``n_bins`` and merges down (to 2, then 1) while any bin has fewer than
    ``min_per_bin`` trials of either class. Catch rows get -1.
    """
    out = pd.Series(-1, index=meta.index, dtype=int)
    present = meta["stim_present"].astype(bool)
    for _, g in meta[present].groupby(["subject", "task"], sort=False):
        for k in range(n_bins, 0, -1):
            b = pd.qcut(g["gabor_contrast"], q=k, labels=False, duplicates="drop")
            counts = pd.crosstab(b, g["seen"].astype(bool))
            if k == 1 or (counts.shape[1] == 2 and int(counts.to_numpy().min()) >= min_per_bin):
                out.loc[g.index] = b.astype(int)
                break
    return out


def stratify_key(meta: pd.DataFrame, bins: pd.Series | None = None) -> np.ndarray:
    """``seen x task x contrast bin`` as integer codes (A1 stratification)."""
    b = bins if bins is not None else contrast_bins(meta)
    key = meta["seen"].astype(int).astype(str) + "|" + meta["task"].astype(str) + "|" + b.astype(
        str
    )
    return np.asarray(pd.factorize(key)[0])


def enough_trials(ts: TrialSet, min_per_class: int = 30) -> tuple[bool, str]:
    """§6 inclusion: >= ``min_per_class`` seen and unseen stimulus-present trials."""
    m = main_set(ts).meta
    n_s, n_u = int(m["seen"].sum()), int((~m["seen"].astype(bool)).sum())
    ok = n_s >= min_per_class and n_u >= min_per_class
    return ok, "" if ok else f"seen {n_s}, unseen {n_u} < {min_per_class}"
