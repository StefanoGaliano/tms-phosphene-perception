"""Per-trial metadata for eegunify epochs (Phase 1 -> Phase 2 hand-off).

eegunify keeps only ``onset, duration, trial_type, value, sample`` in rawdata and writes epochs
without metadata (results/phase1/notes_sub01.md, check d). The staged ``events.tsv``
(``sourcedata/staged/``) keeps every per-trial column, so this joins them back by onset.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd


def staged_events(staged: Path, sub: str, task: str) -> pd.DataFrame:
    return pd.read_csv(
        staged / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{task}_events.tsv",
        sep="\t",
        na_values="n/a",
        keep_default_na=False,
    )


def attach_metadata(epochs: mne.BaseEpochs, events: pd.DataFrame) -> mne.BaseEpochs:
    """Return ``epochs`` with one metadata row per epoch from the staged stimulus rows.

    Matched by onset (epoch sample relative to ``first_samp`` / sfreq) within one sample, and the
    trial_type must agree. Raises if any epoch has no unique match.
    """
    stim = events[events["trial_type"].astype(str).str.startswith("stimulus/")].reset_index(
        drop=True
    )
    sf = epochs.info["sfreq"]
    t_ep = (epochs.events[:, 0] - _first_samp(epochs)) / sf
    onsets = stim["onset"].to_numpy(float)
    idx = np.searchsorted(onsets, t_ep)
    rows = []
    inv: dict[int, str] = {int(v): str(k) for k, v in epochs.event_id.items()}
    for i, (t, j) in enumerate(zip(t_ep, idx, strict=True)):
        cand = [k for k in (j - 1, j) if 0 <= k < len(onsets) and abs(onsets[k] - t) <= 1 / sf]
        if len(cand) != 1:
            raise ValueError(f"epoch {i} at {t:.4f} s: {len(cand)} staged matches")
        k = cand[0]
        want, got = inv[int(epochs.events[i, 2])], str(stim.loc[k, "trial_type"])
        if got != want:
            raise ValueError(f"epoch {i}: trial_type {want} != staged {got}")
        rows.append(k)
    out = epochs.copy()
    out.metadata = stim.iloc[rows].reset_index(drop=True)
    return out


def _first_samp(epochs: mne.BaseEpochs) -> int:
    # eegunify epochs come from a raw starting at sample 0 unless cropped; the provenance of
    # first_samp is not stored in the FIF, so 0 is checked by the real-data test.
    return 0
