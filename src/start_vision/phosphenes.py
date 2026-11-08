"""Phase 3 helpers for the GIN Phosphenes_occipital data (Siviero et al. 2023).

User decision 10 (2026-11-08): the 5 contract channels this cap lacks (P9, P10, Iz, Fpz, AFz)
are interpolated here, before the frozen detector is called, so ``Detector.from_epochs`` sees
all 64 channels and its own ``max_missing_channels`` (4) rule is left untouched.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import mne
import numpy as np

from start_vision.decoding.channels import positions

Inst = TypeVar("Inst", mne.io.BaseRaw, mne.BaseEpochs)

#: Event codes, inferred from C3/C4 lateralization (results/phase3/notes_data_check.md);
#: not stated by the authors.
CODE_PRESENT = "S100"
CODE_ABSENT = "S200"


def complete_contract_channels(inst: Inst, channels: Sequence[str]) -> tuple[Inst, list[str]]:
    """Return a copy holding exactly ``channels`` (in order) and the names interpolated.

    Names are matched case-insensitively (the cap writes ``FP1``); every EEG channel gets its
    10-05 template position; absent channels are added as zeros and spherical-spline
    interpolated from the recorded ones. Non-contract channels (EOG, mastoid) are dropped.
    """
    out = inst.copy().load_data()
    want = {c.lower(): c for c in channels}
    out.rename_channels({n: want[n.lower()] for n in out.ch_names
                         if n.lower() in want and n != want[n.lower()]})
    present = [c for c in channels if c in out.ch_names]
    out.pick(present)
    missing = [c for c in channels if c not in present]
    if missing:
        shape = ((len(out), len(missing), len(out.times)) if isinstance(out, mne.BaseEpochs)
                 else (len(missing), out.n_times))
        info = mne.create_info(missing, out.info["sfreq"], "eeg")
        add: mne.io.BaseRaw | mne.BaseEpochs
        if isinstance(out, mne.BaseEpochs):
            add = mne.EpochsArray(np.zeros(shape), info, tmin=out.tmin, verbose="error")
        else:
            add = mne.io.RawArray(np.zeros(shape), info, verbose="error")
            add.set_meas_date(out.info["meas_date"])
        out.add_channels([add], force_update_info=True)
    out.set_montage(mne.channels.make_dig_montage(ch_pos=positions(list(channels)),
                                                  coord_frame="head"), verbose="error")
    if missing:
        out.info["bads"] = list(missing)
        out.interpolate_bads(reset_bads=True, verbose="error")
    out.reorder_channels(list(channels))
    return out, missing
