"""TrialSet and the ds006171 loader (PHASE2 §4-5, build step 4).

Phase 1 contract as built (README "Plan alignment"): one ``desc-stimraw`` epochs file per
subject x task under ``runs/<task>/derivatives/eegunify-epochs``, all trials, -1000..1000 ms,
256 Hz, no baseline, no low-pass; ICA applied (the pre-ICA copy is ``desc-preproc``, used by
C3). Metadata comes from the staged events via ``start_vision.trials.attach_metadata``.

The yes/no screen side was randomized per trial and not logged (paper, Methods), so
``resp_key_seen`` and ``resp_key_pressed`` are NaN here (decision 3; A6 recovers or drops).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from start_vision.decoding.config import DecodingConfig
from start_vision.trials import attach_metadata, staged_events

META_COLUMNS = (
    "subject", "dataset", "task", "trial_number", "event_code", "stim_present", "hemifield",
    "orientation", "gabor_contrast", "seen", "resp_key_seen", "resp_key_pressed", "rt_ms",
    "cue_valid", "objective_correct", "matched",
)


@dataclass(frozen=True)
class TrialSet:
    X: np.ndarray  # (n_trials, n_channels, n_times) float32, µV
    times: np.ndarray  # (n_times,) s
    ch_names: tuple[str, ...]
    sfreq: float
    meta: pd.DataFrame  # one row per trial, META_COLUMNS
    info: mne.Info

    def __post_init__(self) -> None:
        n, c, t = self.X.shape
        if len(self.meta) != n or len(self.ch_names) != c or len(self.times) != t:
            raise ValueError(f"TrialSet shapes disagree: X {self.X.shape}, meta {len(self.meta)},"
                             f" channels {len(self.ch_names)}, times {len(self.times)}")
        missing = set(META_COLUMNS) - set(self.meta.columns)
        if missing:
            raise ValueError(f"TrialSet meta lacks {sorted(missing)}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def select(self, mask: np.ndarray | pd.Series) -> TrialSet:
        m = np.asarray(mask, dtype=bool)
        return TrialSet(self.X[m], self.times, self.ch_names, self.sfreq,
                        self.meta[m].reset_index(drop=True), self.info)

    def labels(self, target: str = "seen") -> np.ndarray:
        y = self.meta[target]
        if y.isna().any():
            raise ValueError(f"label {target!r} has missing values in this TrialSet")
        return np.asarray(y.astype(int).to_numpy())

    def pick(self, channels: list[str] | tuple[str, ...]) -> TrialSet:
        idx = [self.ch_names.index(c) for c in channels]
        info = mne.pick_info(self.info, idx)
        return TrialSet(self.X[:, idx], self.times, tuple(channels), self.sfreq, self.meta, info)

    @staticmethod
    def concat(sets: list[TrialSet]) -> TrialSet:
        first = sets[0]
        for s in sets[1:]:
            if s.ch_names != first.ch_names or s.sfreq != first.sfreq or not np.allclose(
                s.times, first.times
            ):
                raise ValueError("TrialSets differ in channels, sfreq or times; harmonize first")
        return TrialSet(np.concatenate([s.X for s in sets]), first.times, first.ch_names,
                        first.sfreq, pd.concat([s.meta for s in sets], ignore_index=True),
                        first.info)


def _parse_tag(tt: str) -> tuple[str, str, str, bool]:
    """``stimulus/<cls>/<side>[/<orientation>/m0|m1]`` -> (cls, side, orientation, matched)."""
    parts = tt.split("/")
    cls, side = parts[1], parts[2]
    orient = parts[3] if len(parts) > 3 else "n/a"
    return cls, side, orient, len(parts) > 4 and parts[4] == "m1"


def meta_from_staged(staged_rows: pd.DataFrame, subject: str, task: str) -> pd.DataFrame:
    tags = staged_rows["trial_type"].astype(str).map(_parse_tag)
    cls = tags.map(lambda x: x[0])
    if not cls.isin(["seen", "unseen", "fa", "cr"]).all():
        raise ValueError(f"unexpected classes {sorted(set(cls))} in sub-{subject} {task}")
    side = tags.map(lambda x: x[1])
    return pd.DataFrame({
        "subject": subject,
        "dataset": "ds006171",
        "task": task,
        "trial_number": staged_rows["trial_number"].to_numpy(),
        "event_code": staged_rows["raw_code"].to_numpy(),
        "stim_present": cls.isin(["seen", "unseen"]).to_numpy(),
        # catch: the cued side; noninformative: unknown (Phase 1 decision 6)
        "hemifield": side.where(side != "sideunknown", "unknown").to_numpy(),
        "orientation": tags.map(lambda x: x[2]).to_numpy(),
        "gabor_contrast": staged_rows["gabor_contrast"].to_numpy(dtype=float),
        "seen": cls.isin(["seen", "fa"]).to_numpy(),
        "resp_key_seen": np.nan,
        "resp_key_pressed": np.nan,
        "rt_ms": staged_rows["rt_ms"].to_numpy(dtype=float),
        "cue_valid": staged_rows["cue_valid"].to_numpy(),
        "objective_correct": staged_rows["objective_correct"].to_numpy(),
        "matched": tags.map(lambda x: x[3]).to_numpy(),
    })


def prepare(epochs: mne.BaseEpochs, cfg: DecodingConfig) -> mne.BaseEpochs:
    """Low-pass then decimate (D4), crop to the decoding window. No baseline here."""
    e = epochs.copy().load_data()
    e.pick(["eeg", "eog"])
    e.filter(None, cfg.lowpass_hz, method="iir", iir_params={"order": 4, "ftype": "butter"},
             phase="zero", picks=["eeg", "eog"], verbose="error")
    factor = round(e.info["sfreq"] / cfg.sfreq)
    if abs(e.info["sfreq"] / factor - cfg.sfreq) > 1e-6:
        raise ValueError(f"sfreq {e.info['sfreq']} is not an integer multiple of {cfg.sfreq}")
    e.decimate(factor, verbose="error")
    return e.crop(*cfg.window_s, include_tmax=False)


def epochs_path(runs: Path, subject: str, task: str, desc: str = "stimraw") -> Path:
    return (runs / task / "derivatives" / "eegunify-epochs" / f"sub-{subject}" / "eeg"
            / f"sub-{subject}_task-{task}_desc-{desc}_epo.fif")


def load_trialset(runs: Path, staged: Path, subject: str, task: str,
                  cfg: DecodingConfig, eeg_only: bool = True) -> TrialSet:
    ep = mne.read_epochs(epochs_path(runs, subject, task, cfg.epoch_set), verbose="error")
    ep = attach_metadata(ep, staged_events(staged, subject, task))
    meta = meta_from_staged(ep.metadata, subject, task)
    ep = prepare(ep, cfg)
    if eeg_only:
        ep.pick("eeg")
    x = (ep.get_data() * 1e6).astype(np.float32)
    return TrialSet(x, ep.times.copy(), tuple(ep.ch_names), float(ep.info["sfreq"]), meta,
                    ep.info)
