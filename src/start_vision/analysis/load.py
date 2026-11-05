"""Read eegunify derivatives for one task root (PHASE1_BLUEPRINT §7, analysis/load.py)."""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd


def erp_dir(run: Path, sub: str) -> Path:
    return run / "derivatives" / "eegunify-erp" / f"sub-{sub}" / "eeg"


def subjects(run: Path, epoch_set: str = "stim") -> list[str]:
    """Subjects whose erp stage wrote evokeds for ``epoch_set``."""
    d = run / "derivatives" / "eegunify-erp"
    return sorted(p.name[4:] for p in d.glob("sub-*")
                  if any((p / "eeg").glob(f"*_desc-{epoch_set}_ave.fif")))


def evokeds(run: Path, sub: str, task: str, epoch_set: str = "stim") -> dict[str, mne.Evoked]:
    f = erp_dir(run, sub) / f"sub-{sub}_task-{task}_desc-{epoch_set}_ave.fif"
    return {e.comment: e for e in mne.read_evokeds(f, verbose="error")}


def measures(run: Path, task: str, epoch_set: str = "stim") -> pd.DataFrame:
    files = sorted(run.glob(f"derivatives/eegunify-erp/sub-*/eeg/*_task-{task}_desc-{epoch_set}"
                            "_measures.tsv"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f, sep="\t", dtype={"subject": str}) for f in files),
                     ignore_index=True)


def group_stats(run: Path, task: str) -> pd.DataFrame:
    f = run / "derivatives" / "eegunify-group" / f"task-{task}" / f"task-{task}_stats.tsv"
    return pd.read_csv(f, sep="\t")


def epoch_log(run: Path, sub: str, task: str, epoch_set: str = "stim") -> pd.DataFrame:
    f = (run / "derivatives" / "eegunify-epochs" / f"sub-{sub}" / "eeg"
         / f"sub-{sub}_task-{task}_desc-{epoch_set}_log.tsv")
    return pd.read_csv(f, sep="\t")
