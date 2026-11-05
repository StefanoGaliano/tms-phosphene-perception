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
