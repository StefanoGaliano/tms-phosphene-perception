"""Splitters (PHASE2 §7, build step 5). Every splitter returns explicit index arrays.

- ``within_subject``: StratifiedKFold(n_folds, shuffle) on the A1 key (seen x task x contrast
  bin), repeated with seeds seed, seed+1, ... Strata too small for n_folds are coarsened (drop
  the bin, then the task) so the class split is always honoured.
- ``loso``: LeaveOneGroupOut over subjects (A7): a test subject never appears in training.
- ``cross_condition``: train on ``col == a``, test on ``col == b`` (A8, A9, A6b).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from start_vision.decoding.labels import contrast_bins

Split = tuple[np.ndarray, np.ndarray]


def safe_key(meta: pd.DataFrame, n_folds: int, bins: pd.Series | None = None) -> np.ndarray:
    """Finest stratification key whose every stratum has >= ``n_folds`` trials."""
    seen = meta["seen"].astype(int).astype(str)
    b = (bins if bins is not None else contrast_bins(meta)).astype(str)
    task = meta["task"].astype(str)
    for key in (seen + "|" + task + "|" + b, seen + "|" + task, seen):
        codes, _ = pd.factorize(key)
        if np.bincount(codes).min() >= n_folds:
            return np.asarray(codes)
    raise ValueError(f"a class has fewer than {n_folds} trials; cannot stratify")


def within_subject(meta: pd.DataFrame, n_folds: int, n_repeats: int, seed: int,
                   bins: pd.Series | None = None) -> list[list[Split]]:
    """One list of folds per repeat."""
    if (meta["subject"] != meta["subject"].iloc[0]).any():
        raise ValueError("within_subject expects one subject")
    key = safe_key(meta, n_folds, bins)
    out = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed + r)
        out.append([(tr, te) for tr, te in skf.split(np.zeros(len(key)), key)])
    return out


def loso(meta: pd.DataFrame) -> Iterator[Split]:
    groups = meta["subject"].to_numpy()
    yield from LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups)


def cross_condition(meta: pd.DataFrame, col: str, a: object, b: object) -> Split:
    v = meta[col].to_numpy()
    tr, te = np.flatnonzero(v == a), np.flatnonzero(v == b)
    if not len(tr) or not len(te):
        raise ValueError(f"no trials with {col} == {a!r} or == {b!r}")
    return tr, te


def subsample_to(idx: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Class-stratified random subsample of ``idx`` to ``n`` trials (A8 matched sizes)."""
    if n >= len(idx):
        return idx
    frac = n / len(idx)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c]
        keep.append(rng.choice(members, max(1, round(frac * len(members))), replace=False))
    return np.sort(np.concatenate(keep))
