"""TMS pulse handling for Phase 3 (plans/PHASE3_BLUEPRINT.md §7, steps C1-C5).

Label-blind: no function here takes S100/S200. Values come from configs/phase3.yaml, whose sources
are in docs/PHASE3_REFERENCES.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mne
import numpy as np

FIT_MS = 2.0
"""Samples on each side used for the cubic fit across the cut (project choice, see refs)."""


def artifact_end_ms(raw: mne.io.BaseRaw, pulses: np.ndarray, k: float, hold_ms: float,
                    search_ms: float = 100.0) -> float:
    """C1: end of the pulse artifact (ms after the pulse).

    Pulse-locked median over pulses of |x - prestimulus mean| per EEG channel; the 95th
    percentile over channels must stay below ``k`` x the prestimulus SD of that same trace for
    ``hold_ms``. Prestimulus = -500..-10 ms.
    """
    sf = raw.info["sfreq"]
    x = raw.get_data(picks="eeg")
    pre = np.arange(int(-0.5 * sf), int(-0.01 * sf))
    post = np.arange(0, int(search_ms / 1000 * sf))
    ok = [p for p in pulses if p + pre[0] >= 0 and p + post[-1] < x.shape[1]]
    seg_pre = np.stack([x[:, p + pre] for p in ok])          # (n, ch, t)
    seg_post = np.stack([x[:, p + post] for p in ok])
    mu = seg_pre.mean(axis=2, keepdims=True)
    env = np.percentile(np.median(np.abs(seg_post - mu), axis=0), 95, axis=0)
    base = np.percentile(np.median(np.abs(seg_pre - mu), axis=0), 95, axis=0)
    thr = k * base.std() + base.mean()
    hold = max(1, int(hold_ms / 1000 * sf))
    below = env < thr
    for i in range(len(below) - hold):
        if below[i:i + hold].all():
            return float(i / sf * 1000)
    return float(search_ms)


def cut_window_ms(cut_ms: tuple[float, float], measured_end_ms: float, extend: bool,
                  max_cut_ms: float) -> tuple[float, float]:
    """C2 window: the paper's cut, extended to the measured end if longer, capped."""
    end = cut_ms[1]
    if extend and measured_end_ms > end:
        end = min(measured_end_ms, max_cut_ms)
    return cut_ms[0], end


def interpolate_pulses(raw: mne.io.BaseRaw, pulses: np.ndarray, window_ms: tuple[float, float],
                       picks: Any = "data", fit_ms: float = FIT_MS) -> None:
    """C2 (in place): replace samples in ``window_ms`` around each pulse by a cubic polynomial
    fitted to ``fit_ms`` of data on both sides."""
    sf = raw.info["sfreq"]
    a = int(np.floor(window_ms[0] / 1000 * sf))
    b = int(np.ceil(window_ms[1] / 1000 * sf))
    nfit = max(2, round(fit_ms / 1000 * sf))
    idx = mne.pick_types(raw.info, eeg=True, eog=True, misc=True, exclude=[]) \
        if picks == "data" else mne.pick_channels(raw.ch_names, picks)
    d = raw._data
    for p in pulses:
        i0, i1 = p + a, p + b                      # replaced: i0..i1 inclusive
        left = np.arange(i0 - nfit, i0)
        right = np.arange(i1 + 1, i1 + 1 + nfit)
        if left[0] < 0 or right[-1] >= d.shape[1]:
            continue
        xs = np.r_[left, right] - p
        xi = np.arange(i0, i1 + 1) - p
        ys = d[np.ix_(idx, np.r_[left, right])]   # (ch, 2*nfit)
        # cubic fit per channel, vectorized: solve least squares with a shared design matrix
        scale = max(abs(xs).max(), 1)
        vand = np.vander(xs / scale, 4)
        coef, *_ = np.linalg.lstsq(vand, ys.T, rcond=None)
        d[np.ix_(idx, np.arange(i0, i1 + 1))] = (np.vander(xi / scale, 4) @ coef).T


def resample_with_pulses(raw: mne.io.BaseRaw, pulses: np.ndarray,
                         sfreq: float) -> tuple[mne.io.BaseRaw, np.ndarray]:
    """C3: resample; pulse samples rescaled with the data."""
    ev = np.c_[pulses, np.zeros_like(pulses), np.ones_like(pulses)]
    out, ev2 = raw.resample(sfreq, events=ev, verbose="error")
    return out, ev2[:, 0]


@dataclass
class DecayIcaResult:
    excluded: list[int]
    ratios: list[float]
    n_components: int
    info: dict[str, Any] = field(default_factory=dict)


def decay_ica(raw: mne.io.BaseRaw, pulses: np.ndarray, cut_ms: tuple[float, float],
              epoch_s: tuple[float, float], fit_highpass_hz: float,
              window_ms: tuple[float, float], ratio_threshold: float, method: str = "picard",
              seed: int = 97) -> DecayIcaResult:
    """C4-C5 (in place): TMS-evoked artifact ICA round (TESA tmsMuscle rule).

    Fit on pulse-locked epochs of a high-passed, average-referenced copy with the cut window
    set to a constant (the paper); component ratio = mean |trial-averaged component| in
    ``window_ms`` / mean over the epoch (TESA ``tesa_compselect``). Components with ratio >=
    ``ratio_threshold`` are removed from ``raw`` (continuous); the cut window is then
    re-interpolated.
    """
    sf = raw.info["sfreq"]
    fit = raw.copy().pick("eeg", exclude="bads")
    fit.filter(fit_highpass_hz, None, verbose="error")
    fit.set_eeg_reference("average", projection=False, verbose="error")
    ev = np.c_[pulses, np.zeros_like(pulses), np.ones_like(pulses)]
    ep = mne.Epochs(fit, ev, tmin=epoch_s[0], tmax=epoch_s[1], baseline=None, preload=True,
                    reject_by_annotation=False, verbose="error")
    t = ep.times
    m = (t >= cut_ms[0] / 1000) & (t <= cut_ms[1] / 1000)
    dat = ep.get_data()
    i_after = np.searchsorted(t, cut_ms[1] / 1000) + 1
    dat[:, :, m] = dat[:, :, [i_after]]                # constant over the cut (paper)
    ep._data = dat
    params: dict[str, Any] = {"ortho": False, "extended": True} if method == "picard" else {}
    ica = mne.preprocessing.ICA(n_components=0.99, method=method, fit_params=params, rng=seed,
                                max_iter="auto")
    ica.fit(ep, verbose="error")
    src = ica.get_sources(ep).get_data()               # (n, comp, t)
    score = np.abs(src.mean(axis=0))                   # |trial-averaged component|
    w = (t >= window_ms[0] / 1000) & (t <= window_ms[1] / 1000)
    ratios = score[:, w].mean(axis=1) / score.mean(axis=1)
    excl = [int(i) for i in np.where(ratios >= ratio_threshold)[0]]
    if excl:
        ica.exclude = excl
        eeg = raw.copy().pick("eeg", exclude="bads")
        eeg.set_eeg_reference("average", projection=False, verbose="error")
        ica.apply(eeg, verbose="error")
        picks = mne.pick_channels(raw.ch_names, eeg.ch_names, ordered=True)
        raw._data[picks] = eeg.get_data()
    interpolate_pulses(raw, pulses, cut_ms)
    return DecayIcaResult(excl, [float(r) for r in ratios], int(ica.n_components_),
                          {"sfreq": sf, "reference_after": "average" if excl else "unchanged"})
