"""Synthetic TMS-EEG recordings for Phase 3 tests (plans/PHASE3_BLUEPRINT.md §12).

Continuous data with the GIN cap (59 contract channels recorded, 5 absent) + 4 EOG + LM, pulses
every ~3 s, a "seen" posterior component on labeled trials, an optional response-hand component,
and a pulse artifact (spike, ringing, exponential decay, muscle burst). Values are test settings,
not science values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

from start_vision.decoding.channels import positions

CONTRACT = Path(__file__).parents[2] / "detectors/sv-perception-tms-robust-v1.0.0/contract.json"
ABSENT = ("P9", "P10", "Iz", "Fpz", "AFz")
EOG = ("HEOGL", "HEOGR", "VEOGUP", "VEOGDOWN")


@dataclass
class TmsSim:
    raw: mne.io.RawArray
    events: np.ndarray            # (n, 3) samples, 0, code (1 = S100 present, 2 = S200 absent)
    labels: np.ndarray            # 1 present, 0 absent
    erp: np.ndarray               # (n_eeg, n_times) ground-truth seen ERP (V), -1..1 s
    erp_times: np.ndarray
    meta: dict[str, float] = field(default_factory=dict)


def contract_channels() -> list[str]:
    return list(json.loads(CONTRACT.read_text())["channels"])


def recorded_channels() -> list[str]:
    return [c.upper() if c in ("Fp1", "Fp2") else c for c in contract_channels()
            if c not in ABSENT]


def _topo(pos: np.ndarray, center: np.ndarray, width: float) -> np.ndarray:
    d = np.linalg.norm(pos - center, axis=1)
    return np.exp(-(d / width) ** 2)


def simulate(n_pulses: int = 120, sfreq: float = 2000.0, seen_rate: float = 0.5,
             effect_uv: float = 3.0, hand_uv: float = 0.0, artifact: bool = True,
             spike_uv: float = 5000.0, decay_uv: float = 300.0, decay_tau_ms: float = 15.0,
             muscle_uv: float = 200.0, blinks: bool = True, noise_uv: float = 5.0,
             seed: int = 0) -> TmsSim:
    rng = np.random.default_rng(seed)
    rec = recorded_channels()
    cc = contract_channels()
    pos_c = positions(cc)
    names_c = [c for c in cc if c not in ABSENT]
    pos = np.array([pos_c[c] for c in names_c])
    n_eeg = len(rec)
    isi = rng.uniform(2.6, 3.3, n_pulses)
    onsets_s = 2.0 + np.cumsum(isi)
    dur = onsets_s[-1] + 3.0
    n = int(dur * sfreq)
    t = np.arange(n) / sfreq
    # background: pink-ish noise (cumulated white, high-passed) + posterior alpha
    white = rng.normal(size=(n_eeg, n))
    bg = np.cumsum(white, axis=1)
    bg -= mne.filter.filter_data(bg, sfreq, None, 0.5, verbose="error")
    bg = bg / bg.std(axis=1, keepdims=True) * noise_uv
    post = _topo(pos, pos_c["Oz"], 0.05)
    alpha = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi)) * (1 + 0.5 * np.sin(
        2 * np.pi * 0.07 * t))
    x = bg + 4.0 * post[:, None] * alpha[None, :]
    labels = (rng.random(n_pulses) < seen_rate).astype(int)
    # seen ERP: posterior positivity peaking 0.35 s (gaussian in time), parietal topography
    et = np.arange(int(-1 * sfreq), int(1 * sfreq)) / sfreq
    shape = np.exp(-((et - 0.35) / 0.1) ** 2)
    topo_seen = _topo(pos, pos_c["Pz"], 0.06)
    erp = effect_uv * 1e-6 * topo_seen[:, None] * shape[None, :]
    c3 = _topo(pos, pos_c["C3"], 0.04) - _topo(pos, pos_c["C4"], 0.04)
    hand_shape = np.exp(-((et - 0.45) / 0.15) ** 2)
    art_topo = _topo(pos, pos_c["O1"], 0.07) + 0.2
    mus_topo = _topo(pos, pos_c["PO7"], 0.03)
    spike_n = int(0.005 * sfreq)
    tt = np.arange(int(0.2 * sfreq)) / sfreq
    decay = np.exp(-tt / (decay_tau_ms / 1000))
    muscle = np.where((tt >= 0.005) & (tt <= 0.04), np.sin(2 * np.pi * 120 * tt), 0.0)
    xv = x * 1e-6
    for k, on in enumerate(onsets_s):
        i0 = int(on * sfreq)
        a, b = i0 + int(-1 * sfreq), i0 + int(1 * sfreq)
        if labels[k]:
            xv[:, a:b] += erp
            xv[:, a:b] += hand_uv * 1e-6 * (-c3)[:, None] * hand_shape[None, :]
        else:
            xv[:, a:b] += hand_uv * 1e-6 * (c3)[:, None] * hand_shape[None, :]
        if artifact:
            xv[:, i0 - int(0.001 * sfreq):i0 + spike_n] += (
                spike_uv * 1e-6 * art_topo[:, None] * rng.choice([-1, 1]))
            xv[:, i0:i0 + len(tt)] += decay_uv * 1e-6 * art_topo[:, None] * decay[None, :]
            xv[:, i0:i0 + len(tt)] += muscle_uv * 1e-6 * mus_topo[:, None] * muscle[None, :]
    # EOG + blinks
    eog = rng.normal(size=(4, n)) * 5e-6
    if blinks:
        front = _topo(pos, pos_c["Fp1"], 0.06) + _topo(pos, pos_c["Fp2"], 0.06)
        bshape = np.hanning(int(0.3 * sfreq))
        for bt in rng.uniform(1, dur - 1, int(dur / 8)):
            j = int(bt * sfreq)
            xv[:, j:j + len(bshape)] += 100e-6 * front[:, None] * bshape[None, :]
            eog[2, j:j + len(bshape)] += 150e-6 * bshape
            eog[3, j:j + len(bshape)] -= 50e-6 * bshape
    lm = rng.normal(size=(1, n)) * 5e-6
    data = np.vstack([xv, eog, lm])
    info = mne.create_info(rec + list(EOG) + ["LM"], sfreq,
                           ["eeg"] * n_eeg + ["eog"] * 4 + ["misc"])
    raw = mne.io.RawArray(data, info, verbose="error")
    samples = (onsets_s * sfreq).astype(int)
    events = np.c_[samples, np.zeros(n_pulses, int), np.where(labels == 1, 1, 2)]
    raw.set_annotations(mne.Annotations(onsets_s, 0.0, np.where(labels == 1, "S100", "S200")))
    return TmsSim(raw, events, labels, erp, et,
                  {"spike_end_ms": 5.0, "decay_tau_ms": decay_tau_ms, "sfreq": sfreq})
