"""Loaders for the inputs on disk (blueprint §4) and the output-folder guard (D6).

``runs/`` is never used (user, 2026-11-09).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from start_vision.questions.params import EKLUND_CRITICAL, FIRST_LOOK_DIRS

QROOT = Path("results") / "questions_to_solve"


def eklund_trials(root: Path) -> pd.DataFrame:
    """Critical trials (condition 1-4) with the duration exclusion applied (P§2)."""
    d = pd.read_csv(root / "results" / "eklund2018" / "trial_scores.tsv.gz", sep="\t")
    keep = d["cls"].isin(EKLUND_CRITICAL) & d["exclusion"].isna()
    d = d.loc[keep, ["gabor", "fp", "cls", "condition", "rating", "obj_corr", "duration_ms",
                     "p_seen"]].reset_index(drop=True)
    return d


def _side(trial_type: str) -> str:
    parts = trial_type.split("/")
    return parts[2] if len(parts) > 2 else "unknown"


def _bool(v: object) -> float:
    s = str(v)
    if s in ("True", "1", "1.0"):
        return 1.0
    if s in ("False", "0", "0.0"):
        return 0.0
    return np.nan


def ds006171_trials(root: Path) -> pd.DataFrame:
    """LOSO held-out scores joined to the staged events (contrast, orientation answer, side)."""
    p = pd.read_csv(root / "derivatives" / "sv_decode_group" / "loso_predictions.tsv", sep="\t",
                    dtype={"subject": str})
    parts = []
    for (s, t), g in p.groupby(["subject", "task"], sort=True):
        f = (root / "sourcedata" / "staged" / "ds006171" / f"sub-{s}" / "eeg"
             / f"sub-{s}_task-{t}_events.tsv")
        e = pd.read_csv(f, sep="\t", dtype={"trial_number": float})
        e = e[e["trial_type"].str.startswith("stimulus/")]
        e = e.assign(side=e["trial_type"].map(_side),
                     correct=e["objective_correct"].map(_bool),
                     gabor_contrast=pd.to_numeric(e["gabor_contrast"], errors="coerce"))
        m = g.merge(e[["trial_number", "gabor_contrast", "correct", "side"]], on="trial_number",
                    how="left", validate="one_to_one")
        parts.append(m)
    d = pd.concat(parts, ignore_index=True)
    d["seen"] = d["seen"].astype(str).isin(["True", "1", "1.0"])
    return d


def gin_scores(root: Path) -> pd.DataFrame:
    """Phase 3 primary scores of the included recordings (P§1)."""
    r = root / "results" / "phase3"
    inc = pd.read_csv(r / "inclusion_primary.tsv", sep="\t", dtype={"subject": str})
    s = pd.read_csv(r / "scores_primary.tsv", sep="\t", dtype={"subject": str})
    keep = set(inc.loc[inc["included"], "recording"])
    return s[s["recording"].isin(keep)].reset_index(drop=True)


def gin_epochs_iter(root: Path, recordings: list[str]
                    ) -> Iterator[tuple[str, mne.BaseEpochs]]:
    for rid in recordings:
        f = root / "derivatives" / "sv_p3_clean" / "primary" / f"{rid}_desc-detector_epo.fif"
        yield rid, mne.read_epochs(f, verbose="error")


def out_dir(root: Path, name: str, force: bool = False) -> Path:
    """``results/questions_to_solve/prereg_<name>/``; refuses first-look folders and overwrites."""
    if name in FIRST_LOOK_DIRS:
        raise ValueError(f"{name} is a first-look folder (P§0): never written by the prereg run")
    d = root / QROOT / f"prereg_{name}"
    if d.exists() and any(d.iterdir()):
        if not force:
            raise FileExistsError(f"{d} exists; the analyses run once (P§1): use --force")
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def commit_hash(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_primary(d: Path, question: str, test: str, tail: str, row: dict[str, float]) -> None:
    """``primary.json`` read by the family step (D7)."""
    rec = {"question": question, "test": test, "tail": tail,
           **{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
              for k, v in row.items()}}
    (d / "primary.json").write_text(json.dumps(rec, indent=2) + "\n")
