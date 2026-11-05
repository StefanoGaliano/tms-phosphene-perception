"""attach_metadata: staged per-trial columns joined onto eegunify epochs by onset."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from start_vision.trials import attach_metadata


def _epochs(onsets_s: list[float], types: list[str], sf: float = 256.0) -> mne.EpochsArray:
    info = mne.create_info(["Oz"], sf, "eeg")
    ids = {t: i + 1 for i, t in enumerate(sorted(set(types)))}
    ev = np.array([[round(t * sf), 0, ids[k]] for t, k in zip(onsets_s, types, strict=True)])
    return mne.EpochsArray(np.zeros((len(ev), 1, 10)), info, ev, event_id=ids, verbose="error")


def _staged() -> pd.DataFrame:
    return pd.DataFrame({
        "onset": [1.0, 1.6, 3.0, 3.7, 5.0],
        "trial_type": ["stimulus/seen/left", "response/seen", "stimulus/unseen/right",
                       "response/unseen", "stimulus/cr/left"],
        "gabor_contrast": [0.03, np.nan, 0.02, np.nan, np.nan],
    })


def test_ds006171_metadata_join_skips_rejected_epochs() -> None:
    ep = _epochs([1.002, 5.0], ["stimulus/seen/left", "stimulus/cr/left"])  # trial 2 rejected
    m = attach_metadata(ep, _staged()).metadata
    assert list(m["trial_type"]) == ["stimulus/seen/left", "stimulus/cr/left"]
    assert m["gabor_contrast"].iloc[0] == 0.03


def test_ds006171_metadata_join_rejects_type_mismatch() -> None:
    ep = _epochs([3.0], ["stimulus/seen/left"])
    with pytest.raises(ValueError, match="trial_type"):
        attach_metadata(ep, _staged())
