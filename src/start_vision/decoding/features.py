"""Feature transforms (PHASE2 §7 C4, §9). All unsupervised or fold-fitted; no labels seen here.

- ``baseline``: subtract the per-trial mean of a prestimulus window (§4: -200..0 ms).
- ``prestim_normalize``: z-score each channel with the subject's own prestimulus mean/SD over
  all trials (A7, §9 step 2; label-free, so a new subject can do it).
- ``window_means``: mean amplitude per channel in consecutive bins (§9 step 3), flattened.
- ``alpha_power``: log power per channel in the alpha band of a prestimulus window (C4).
"""

from __future__ import annotations

import numpy as np
from mne.time_frequency import psd_array_multitaper

from start_vision.decoding.data import TrialSet


def _mask(times: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    m = (times >= win[0] - 1e-9) & (times < win[1] - 1e-9)
    if not m.any():
        raise ValueError(f"window {win} has no samples in {times[0]:.3f}..{times[-1]:.3f} s")
    return m


def baseline(X: np.ndarray, times: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    return X - X[..., _mask(times, win)].mean(-1, keepdims=True)


def prestim_normalize(X: np.ndarray, times: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    """Per-channel z-score from the prestimulus samples of all trials (one subject)."""
    pre = X[..., _mask(times, win)]
    mu = pre.mean(axis=(0, 2), keepdims=True)
    sd = pre.std(axis=(0, 2), keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def window_means(X: np.ndarray, times: np.ndarray, win: tuple[float, float],
                 bin_s: float) -> np.ndarray:
    """(n_trials, n_channels * n_bins): mean per channel in bins of ``bin_s`` from win[0]."""
    edges = np.arange(win[0], win[1] - 1e-9, bin_s)
    if len(edges) == 0:
        raise ValueError(f"window {win} shorter than one bin of {bin_s} s")
    cols = [X[..., _mask(times, (a, min(a + bin_s, win[1])))].mean(-1) for a in edges]
    return np.stack(cols, axis=-1).reshape(len(X), -1)


def alpha_power(ts: TrialSet, band: tuple[float, float], win: tuple[float, float]) -> np.ndarray:
    """(n_trials, n_channels) log10 alpha power in ``win`` (multitaper)."""
    seg = ts.X[..., _mask(ts.times, win)].astype(np.float64)
    psd, _ = psd_array_multitaper(seg, ts.sfreq, fmin=band[0], fmax=band[1],
                                      verbose="error")
    return np.asarray(np.log10(psd.mean(-1)))


def individual_alpha_band(ts: TrialSet, default: tuple[float, float], halfwidth: float,
                          win: tuple[float, float]) -> tuple[float, float]:
    """Peak of the mean prestimulus PSD (posterior-agnostic, all channels) in the default band,
    ± ``halfwidth``; the default band if the peak sits on an edge (no peak)."""
    seg = ts.X[..., _mask(ts.times, win)].astype(np.float64)
    psd, freqs = psd_array_multitaper(seg, ts.sfreq, fmin=default[0] - 2, fmax=default[1] + 2,
                                      verbose="error")
    spec = np.log10(psd).mean(axis=(0, 1))
    inb = (freqs >= default[0]) & (freqs <= default[1])
    k = int(np.argmax(np.where(inb, spec, -np.inf)))
    if k in (int(np.flatnonzero(inb)[0]), int(np.flatnonzero(inb)[-1])):
        return default
    return (float(freqs[k] - halfwidth), float(freqs[k] + halfwidth))
