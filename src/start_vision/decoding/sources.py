"""A12 dataset adapters (PHASE2 §5, §7 A12; build step 16, 2026-11-07).

Every analysis takes ``TrialSet``s, never files. A cross-dataset or cross-modality test is:
load each dataset through its adapter, ``channels.harmonize`` the sets (common channels, sfreq,
times), then ``across.transfer(train, test, cfg)`` (or ``Detector.predict_proba`` on a frozen
detector whose contract channels the new set covers).

Only ``Ds006171Adapter`` is implemented. The others are interface stubs: each one raises
``NotImplementedError`` naming what is missing before it can be written. Facts about the other
datasets come from ``docs/DATASETS.md`` (verified there); open items are in
``docs/A12_OPEN_ISSUES.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet

Modality = Literal["visual", "tactile", "tms-visual"]


@runtime_checkable
class DatasetAdapter(Protocol):
    """One dataset -> TrialSets. ``load`` must fill ``seen``, ``stim_present`` and the
    condition columns of ``data.META_COLUMNS`` (NaN where the dataset has no such field), and
    set ``dataset`` to ``name``."""

    name: str
    modality: Modality

    def subjects(self, root: Path) -> list[str]: ...

    def load(self, subject: str, root: Path) -> TrialSet: ...


class Ds006171Adapter:
    """The training dataset (Melcón et al. 2024): all included tasks of a subject, EEG only,
    all trials (stimulus-present and catch), from the Phase 1 ``stimraw`` epochs."""

    name = "ds006171"
    modality: Modality = "visual"

    def __init__(self, cfg: DecodingConfig | None = None) -> None:
        self.cfg = cfg or DecodingConfig()

    def subjects(self, root: Path) -> list[str]:
        from start_vision.decoding.step0 import included_tasks

        return sorted(included_tasks(root))

    def load(self, subject: str, root: Path) -> TrialSet:
        from start_vision.decoding.data import load_trialset
        from start_vision.decoding.stages import Project
        from start_vision.decoding.step0 import included_tasks

        proj = Project.default(root)
        tasks = included_tasks(root)[subject]
        return TrialSet.concat([load_trialset(proj.runs, proj.staged, subject, t, self.cfg)
                                for t in tasks])


class _Stub:
    name: str
    modality: Modality
    missing: str

    def subjects(self, root: Path) -> list[str]:
        raise NotImplementedError(f"{self.name}: {self.missing}")

    def load(self, subject: str, root: Path) -> TrialSet:
        raise NotImplementedError(f"{self.name}: {self.missing}")


class Eklund2018Adapter:
    """Eklund & Wiens 2018 (figshare doi:10.17045/sthlmuni.5418967; A12 issue #1, built
    2026-11-07). One "subject" per included session, ``<fp>_<large|small>``. Trials: aware correct
    (seen = 1) and unaware correct (seen = 0), the authors' contrast (user decision 9); epochs from
    ``runs/eklund2018`` (eegunify, ICA), prepared like ds006171 (``data.prepare``)."""

    name = "eklund2018"
    modality: Modality = "visual"

    def __init__(self, cfg: DecodingConfig | None = None) -> None:
        self.cfg = cfg or DecodingConfig()

    def subjects(self, root: Path) -> list[str]:
        import pandas as pd

        from start_vision.decoding.eklund_test import epochs_file

        t = pd.read_csv(root / "configs" / "eklund2018_sessions.tsv", sep="\t")
        t = t[t["authors_included"]]
        return [f"{int(f):02d}_{g}" for f, g in zip(t["fp"], t["gabor"], strict=True)
                if epochs_file(root / "runs" / "eklund2018", f"{int(f):02d}", str(g)).exists()]

    def load(self, subject: str, root: Path) -> TrialSet:
        import numpy as np
        import pandas as pd

        from start_vision.decoding.data import META_COLUMNS, prepare
        from start_vision.decoding.eklund_test import load_session

        sub, gabor = subject.split("_")
        ep, meta = load_session(root / "runs" / "eklund2018",
                                root / "sourcedata" / "staged" / "eklund2018", sub, gabor)
        keep = (meta["cls"].isin(["aware", "unaware"])
                & (meta["exclusion"].fillna("") == "")).to_numpy()
        e = prepare(ep[np.flatnonzero(keep)], self.cfg).pick("eeg")
        m = meta[keep].reset_index(drop=True)
        side = m["quadrant"].map({"UL": "left", "LL": "left", "UR": "right", "LR": "right"})
        out = pd.DataFrame({c: np.nan for c in META_COLUMNS}, index=m.index).astype(object)
        out = out.assign(subject=subject, dataset=self.name, task=gabor,
                         trial_number=m["trial_number"], event_code=m["condition"],
                         stim_present=True, hemifield=side, seen=(m["cls"] == "aware").astype(int),
                         objective_correct=True, matched=True)
        x = (e.get_data() * 1e6).astype(np.float32)
        return TrialSet(x, e.times.copy(), tuple(e.ch_names), float(e.info["sfreq"]), out, e.info)


class Ds005273Adapter(_Stub):
    """OpenNeuro ds005273 (Rodríguez-San Esteban, Chica, González-López; seen/unseen,
    33 subjects, 63 ch, BrainVision). Parked."""

    name = "ds005273"
    modality: Modality = "visual"
    missing = ("blocked: its events.tsv markers (e.g. 'Stimulus/S 70') have no code book and "
               "the paper was not found, so seen/unseen trials cannot be identified. "
               "See docs/A12_OPEN_ISSUES.md #2.")


class TactileAdapter(_Stub):
    """Near-threshold touch (cross-modality test of the detector). No dataset yet."""

    name = "tactile"
    modality: Modality = "tactile"
    missing = ("no open near-threshold tactile detection EEG dataset found (ds005291/ds005293 "
               "are suprathreshold laser pain ratings). See docs/A12_OPEN_ISSUES.md #3.")


class TmsPhospheneAdapter(_Stub):
    """Phase 3 input: TMS-EEG at phosphene threshold with per-trial reports. No data yet;
    the detector to apply is ``detectors/sv-perception-tms-robust-v1.0.0``."""

    name = "tms-phosphene"
    modality: Modality = "tms-visual"
    missing = ("no data: Phase 3 is blocked until TMS-EEG with phosphene reports exists "
               "(papers/authors or a lab study). See docs/A12_OPEN_ISSUES.md #4.")


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    a.name: a for a in (Ds006171Adapter, Eklund2018Adapter, Ds005273Adapter,
                        TactileAdapter, TmsPhospheneAdapter)
}
