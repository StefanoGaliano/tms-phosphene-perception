"""Synthetic ds006171-like event streams with known per-trial truth (PHASE1 Step 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from start_vision import codes as C  # noqa: N812


@dataclass
class Truth:
    raw_code: int
    photodiode_s: float | None
    report: int | None
    orient_response: int | None


def make_event_stream(
    n_trials: int = 80,
    task: str = "nocue",
    seed: int = 0,
    sfreq: float = 1024.0,
    missing_photodiode: tuple[int, ...] = (),
    missing_report: tuple[int, ...] = (),
    stray_128: int = 3,
) -> tuple[pd.DataFrame, list[Truth]]:
    """Return an aligned events frame (``onset_s, code, trial_number, gabor_contrast,
    subjective_r, objective_r``) as ``stage.align`` produces, and the truth per trial.

    Photodiode delays are 30-60 ms (on the sample grid); seen probability rises with contrast.
    """
    rng = np.random.default_rng(seed)
    stim = C.RAW_STIMULI[task]
    gabors = [k for k, v in stim.items() if not v.catch]
    catches = [k for k, v in stim.items() if v.catch]
    rows: list[dict[str, float]] = []
    truth: list[Truth] = []
    q = lambda x: round(x * sfreq) / sfreq  # noqa: E731

    def add(t: float, code: int, n: int, **kw: float) -> None:
        rows.append(
            {
                "onset_s": q(t),
                "code": code,
                "trial_number": n,
                "gabor_contrast": np.nan,
                "subjective_r": np.nan,
                "objective_r": np.nan,
                **kw,
            }
        )

    t = 2.0
    for _ in range(stray_128):
        add(t, C.PHOTODIODE, 0)
        t += 0.5
    for n in range(1, n_trials + 1):
        code = int(rng.choice(catches)) if n % 10 == 0 else int(rng.choice(gabors))
        rs = stim[code]
        if C.CUES[task]:
            add(t, rs.cue_side or 1, n)
            add(t + 0.05, C.PHOTODIODE, n)
            t += 0.8
        contrast = np.nan if rs.catch else float(rng.uniform(0.01, 0.04))
        p_seen = 0.05 if rs.catch else (contrast - 0.01) / 0.03
        seen = bool(rng.random() < p_seen)
        rep = None if n in missing_report else (C.REPORT_SEEN if seen else C.REPORT_UNSEEN)
        ori = None
        if rep is not None and not rs.catch and rng.random() < 0.3:
            ori = rs.orient if rng.random() < 0.8 else 3 - rs.orient
        add(
            t,
            code,
            n,
            gabor_contrast=contrast,
            subjective_r=np.nan if rep is None else float(seen),
            objective_r=np.nan if ori is None else float(ori),
        )
        pd_t = None
        if n not in missing_photodiode:
            pd_t = q(q(t) + rng.integers(round(0.030 * sfreq), round(0.060 * sfreq) + 1) / sfreq)
            add(pd_t, C.PHOTODIODE, n)
        t += 0.9
        if rep is not None:
            add(t, rep, n)
            t += 0.6
        if ori is not None:
            add(t, C.ORIENT_VERTICAL if ori == C.VERTICAL else C.ORIENT_HORIZONTAL, n)
            t += 0.6
        truth.append(Truth(code, pd_t, rep, ori))
        t += 0.7
    df = pd.DataFrame(rows)
    df.attrs["sfreq"] = sfreq
    return df, truth


# ---------------------------------------------------------------------------------------------
# Step 8: continuous recordings with planted VAN and LP (end-to-end regression).
# ---------------------------------------------------------------------------------------------

SFREQ = 512.0
VAN_MS, VAN_SIGMA_MS = 230.0, 30.0
LP_MS, LP_SIGMA_MS = 450.0, 80.0
VAN_UV = {"seen": -2.0, "unseen": 0.0}  # at the peak electrode contralateral to the Gabor
LP_UV = {"seen": 4.0, "unseen": 1.0}  # at Pz


@dataclass
class VisionRecording:
    names: list[str]
    data_uV: np.ndarray  # (n_channels, n_times)
    events: pd.DataFrame  # aligned events frame, as stage.align returns
    van_topo: dict[int, np.ndarray]  # side -> weight per EEG channel (1 at the peak)
    lp_topo: np.ndarray
    n_eeg: int


def _eeg_names() -> list[str]:
    from pathlib import Path

    import yaml

    t = Path(__file__).parents[2] / "configs" / "template_ds006171.yaml"
    ren = yaml.safe_load(t.read_text())["channels"]["rename"]
    return [v for k, v in ren.items() if k[0] in "ABCD" and k[1:].isdigit()]


def _gauss_topo(xyz: np.ndarray, centre: np.ndarray, width_m: float) -> np.ndarray:
    return np.exp(-((np.linalg.norm(xyz - centre, axis=1) / width_m) ** 2))


def make_vision_recording(
    n_trials: int = 240,
    *,
    seed: int = 0,
    null: bool = False,
    noise_uV: float = 3.0,
) -> VisionRecording:
    """128-channel ds006171-like recording (cap names, standard_1005 positions) + VEOG1/2.

    Nocue task codes. Each Gabor appears at a photodiode onset 30-60 ms after its code; the
    VAN is a Gaussian at 230 ms centred on the occipital site contralateral to the Gabor
    (O2 for left, O1 for right), the LP a Gaussian at 450 ms centred on Pz. Seen trials have
    higher contrast (so matching is exercised). ``null=True``: seen and unseen get the same
    (unseen) amplitudes.
    """
    import mne

    rng = np.random.default_rng(seed)
    names = _eeg_names()
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    xyz = np.array([pos[n] for n in names])
    n_eeg = len(names)
    trial_s = 2.6
    n = int((n_trials * (trial_s + 0.2) + 6) * SFREQ)  # + per-trial jitter (max 0.2 s)
    t = np.arange(n) / SFREQ

    # background: spatially smooth pink noise + posterior alpha (as eegunify's generator)
    spec = rng.standard_normal((n_eeg, n // 2 + 1)) + 1j * rng.standard_normal((n_eeg, n // 2 + 1))
    f = np.fft.rfftfreq(n)
    spec[:, 1:] /= np.sqrt(f[1:])
    spec[:, 0] = 0
    pink = np.fft.irfft(spec, n)
    pink /= pink.std(axis=1, keepdims=True)
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    mix = np.exp(-((dist / 0.05) ** 2))
    mix /= np.linalg.norm(mix, axis=1, keepdims=True)
    data = noise_uV * mix @ pink + 0.3 * rng.standard_normal((n_eeg, n))
    # posterior alpha, weak and broad: a narrow 3 uV alpha made pyprep flag every occipital
    # channel on the dense cap (first Step 8 run), which interpolated away the planted VAN
    data += 1.5 * _gauss_topo(xyz, pos["Oz"], 0.12)[:, None] * np.sin(2 * np.pi * 10 * t)

    van_topo = {
        C.LEFT: _gauss_topo(xyz, pos["O2"], 0.04),
        C.RIGHT: _gauss_topo(xyz, pos["O1"], 0.04),
    }
    lp_topo = _gauss_topo(xyz, pos["Pz"], 0.07)
    tk = np.arange(int(-0.2 * SFREQ), int(0.9 * SFREQ)) / SFREQ * 1000
    van_w = np.exp(-0.5 * ((tk - VAN_MS) / VAN_SIGMA_MS) ** 2)
    lp_w = np.exp(-0.5 * ((tk - LP_MS) / LP_SIGMA_MS) ** 2)

    q = lambda x: round(x * SFREQ) / SFREQ  # noqa: E731
    rows: list[dict[str, float]] = []

    def add(tt: float, code: int, k: int, **kw: float) -> None:
        rows.append(
            {
                "onset_s": q(tt),
                "code": code,
                "trial_number": k,
                "gabor_contrast": np.nan,
                "subjective_r": np.nan,
                "objective_r": np.nan,
                **kw,
            }
        )

    gabors = [11, 12, 21, 22]
    tt = 3.0
    for k in range(1, n_trials + 1):
        catch = k % 10 == 0
        code = int(rng.choice([31, 32])) if catch else int(rng.choice(gabors))
        rs = C.RAW_STIMULI["nocue"][code]
        contrast = np.nan if catch else float(rng.uniform(0.01, 0.04))
        seen = bool(rng.random() < (0.05 if catch else 0.2 + 0.6 * (contrast - 0.01) / 0.03))
        add(tt, code, k, gabor_contrast=contrast, subjective_r=float(seen))
        onset = q(tt + rng.integers(int(0.03 * SFREQ), int(0.06 * SFREQ) + 1) / SFREQ)
        add(onset, C.PHOTODIODE, k)
        if not catch:
            cls = "unseen" if (null or not seen) else "seen"
            i0 = round(onset * SFREQ) + int(-0.2 * SFREQ)
            seg = slice(i0, i0 + len(tk))
            data[:, seg] += VAN_UV[cls] * van_topo[rs.side][:, None] * van_w[None]
            data[:, seg] += LP_UV[cls] * lp_topo[:, None] * lp_w[None]
        add(tt + 1.0, C.REPORT_SEEN if seen else C.REPORT_UNSEEN, k)
        tt += trial_s + float(rng.uniform(0, 0.2))
    veog = 3.0 * rng.standard_normal((2, n))
    events = pd.DataFrame(rows)
    events.attrs["sfreq"] = SFREQ
    return VisionRecording(
        [*names, "VEOG1", "VEOG2"], np.vstack([data, veog]), events, van_topo, lp_topo, n_eeg
    )


def window_mean(
    weights: np.ndarray,
    names: list[str],
    sites: tuple[str, ...],
    centre_ms: float,
    sigma_ms: float,
    win: tuple[float, float],
) -> float:
    """Noise-free mean of a planted Gaussian over ``sites`` and ``win`` (per µV of amplitude)."""
    tt = np.arange(win[0], win[1] + 1e-9, 1000 / SFREQ)
    w = np.exp(-0.5 * ((tt - centre_ms) / sigma_ms) ** 2).mean()
    return float(np.mean([weights[names.index(s)] for s in sites]) * w)
