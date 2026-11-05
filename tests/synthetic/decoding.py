"""Synthetic TrialSets with known decodable signals (PHASE2 §11).

Background: pink noise per source, mixed to channels by a random spatial matrix. Generators set
what (if anything) separates seen from unseen:

- ``seen_late``: pattern A at 300-500 ms on seen trials (amplitude ``d``)
- ``sustained``: one pattern, 200-800 ms (square generalization matrix)
- ``chain``: three patterns in sequence, 200-400 / 400-600 / 600-800 ms (diagonal matrix)
- ``contrast_only``: signal scales with contrast; seen depends on contrast only
- ``key_lateralized``: pattern on the pressed key's side, 300-600 ms; yes-key side random
- ``eog_only``: signal on the ``HEOG`` channel only
- ``alpha_pre``: lower prestimulus 10 Hz power before seen trials
- ``null``: nothing

Per-subject jitter: gain, topography rotation (mixing with a random pattern), latency shift.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from start_vision.decoding.data import META_COLUMNS, TrialSet

GENERATORS = ("seen_late", "sustained", "chain", "contrast_only", "key_lateralized",
              "eog_only", "alpha_pre", "null")


def _pink(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    n = shape[-1]
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = (rng.standard_normal((*shape[:-1], len(f)))
            + 1j * rng.standard_normal((*shape[:-1], len(f)))) / np.sqrt(f)
    x = np.fft.irfft(spec, n=n)
    return np.asarray(x / x.std())


def _bump(times: np.ndarray, a: float, b: float) -> np.ndarray:
    w = ((times >= a) & (times < b)).astype(float)
    return np.asarray(np.convolve(w, np.hanning(9) / np.hanning(9).sum(), mode="same"))


def make_subject(subject: str, generator: str, n_trials: int = 200, n_channels: int = 32,
                 sfreq: float = 128.0, d: float = 1.0, task: str = "nocue", seed: int = 0,
                 window_s: tuple[float, float] = (-0.5, 1.0)) -> TrialSet:
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}")
    rng = np.random.default_rng(seed)
    times = np.arange(round(window_s[0] * sfreq), round(window_s[1] * sfreq)) / sfreq
    n_t = len(times)
    names = [f"E{i:02d}" for i in range(n_channels - 1)] + ["HEOG"]
    mixing = rng.standard_normal((n_channels, n_channels)) / np.sqrt(n_channels)
    x = np.einsum("ck,nkt->nct", mixing, _pink(rng, (n_trials, n_channels, n_t)))
    x += 0.5 * rng.standard_normal(x.shape)

    contrast = rng.uniform(0.01, 0.05, n_trials)
    if generator == "contrast_only":
        p_seen = 1 / (1 + np.exp(-(contrast - 0.03) / 0.004))
        seen = rng.uniform(size=n_trials) < p_seen
    else:
        seen = rng.uniform(size=n_trials) < 0.5
    key_seen = rng.choice(["left", "right"], n_trials)
    pressed = np.where(seen, key_seen, np.where(key_seen == "left", "right", "left"))

    # per-subject jitter
    gain = rng.uniform(0.7, 1.3)
    shift = rng.uniform(-0.03, 0.03)
    base_pattern = np.random.default_rng(12345).standard_normal(n_channels - 1)  # shared truth
    pattern = base_pattern + 0.3 * rng.standard_normal(n_channels - 1)
    pattern /= np.linalg.norm(pattern) / np.sqrt(n_channels - 1)
    eeg = slice(0, n_channels - 1)

    def add(mask: np.ndarray, pat: np.ndarray, course: np.ndarray, amp: float | np.ndarray,
            chans: slice | int = eeg) -> None:
        a = np.broadcast_to(np.asarray(amp, dtype=float), (n_trials,))[mask]
        x[mask, chans] += (a[:, None, None] * np.atleast_1d(pat)[None, :, None]
                           * course[None, None, :]).reshape(x[mask, chans].shape)

    t = times - shift
    if generator == "seen_late":
        add(seen, pattern, _bump(t, 0.3, 0.5), d * gain)
    elif generator == "sustained":
        add(seen, pattern, _bump(t, 0.2, 0.8), d * gain)
    elif generator == "chain":
        pats = np.random.default_rng(54321).standard_normal((3, n_channels - 1))
        for p, (a, b) in zip(pats, ((0.2, 0.4), (0.4, 0.6), (0.6, 0.8)), strict=True):
            add(seen, p, _bump(t, a, b), d * gain)
    elif generator == "contrast_only":
        add(np.ones(n_trials, bool), pattern, _bump(t, 0.1, 0.5),
            d * gain * (contrast - 0.03) / 0.01)
    elif generator == "key_lateralized":
        lat = np.where(np.arange(n_channels - 1) < (n_channels - 1) // 2, 1.0, -1.0)
        add(pressed == "left", lat, _bump(t, 0.3, 0.6), d * gain)
        add(pressed == "right", -lat, _bump(t, 0.3, 0.6), d * gain)
    elif generator == "eog_only":
        add(seen, np.ones(1), _bump(t, 0.2, 0.5), 5 * d * gain, chans=n_channels - 1)
    elif generator == "alpha_pre":
        osc = np.sin(2 * np.pi * 10 * times) * (times < 0)
        amp = np.where(seen, 1.0, 1.0 + d)
        x[:, eeg] += amp[:, None, None] * pattern[None, :, None] * osc[None, None, :]

    meta = pd.DataFrame({
        "subject": subject, "dataset": "synthetic", "task": task,
        "trial_number": np.arange(1, n_trials + 1), "event_code": 11, "stim_present": True,
        "hemifield": rng.choice(["left", "right"], n_trials),
        "orientation": rng.choice(["vertical", "horizontal"], n_trials),
        "gabor_contrast": contrast, "seen": seen, "resp_key_seen": key_seen,
        "resp_key_pressed": pressed, "rt_ms": rng.uniform(400, 1200, n_trials),
        "cue_valid": np.nan, "objective_correct": np.nan, "matched": True,
    })[list(META_COLUMNS)]
    info = mne.create_info(names, sfreq, ["eeg"] * (n_channels - 1) + ["eog"])
    return TrialSet(x.astype(np.float32), times, tuple(names), sfreq, meta, info)


def make_synthetic_trialset(n_subjects: int = 3, n_trials: int = 200, n_channels: int = 32,
                            sfreq: float = 128.0, generator: str = "seen_late", seed: int = 0,
                            d: float = 1.0) -> list[TrialSet]:
    return [make_subject(f"{i + 1:02d}", generator, n_trials, n_channels, sfreq, d,
                         seed=seed * 1000 + i) for i in range(n_subjects)]
