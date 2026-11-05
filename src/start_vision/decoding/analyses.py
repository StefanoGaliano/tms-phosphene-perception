"""Registered within-subject analyses for ``sv_decode`` (PHASE2 §7; build step 14).

Each analysis takes a ``SubjectData`` (one subject, its included tasks) and returns named arrays,
saved as ``sub-XX_desc-<name>_scores.npz``. Extra views load lazily:

- ``main``: stimulus-present trials (``desc-stimraw``, 128 Hz, -500..1000 ms, ICA applied)
- ``catch``: stimulus-absent trials of the same epochs
- ``eog``: the bipolar EOG channels (VEOG = VEOG1-VEOG2, HEOG = F7-F8, Phase 1 decision 7)
- ``noica``: ``desc-preproc`` (before ICA) epoched at the same events as ``desc-stimraw``
  (re-epoching ``desc-clean`` that way reproduces ``desc-stimraw`` exactly, checked 2026-11-05)
- ``alpha_src``: main trials at 256 Hz, -1000..1000 ms (alpha from non-decimated data, C4)

Pre-set before any group result was seen (2026-11-05): window-level controls (C4, C5) use the
post-stimulus window ``WIN_S`` = 200-600 ms (the middle of PHASE2 §9's candidate windows);
the HEOG step rule of C3c is Phase 1's (32 µV, -50..350 ms, decision 5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import mne
import numpy as np

from start_vision.decoding import across, channels, confounds, labels, within
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import (
    TrialSet,
    epochs_path,
    load_trialset,
    meta_from_staged,
    prepare,
)
from start_vision.trials import attach_metadata, staged_events

WIN_S = (0.2, 0.6)
HEOG_STEP_UV, HEOG_WINDOW_S = 32.0, (-0.05, 0.35)


@dataclass
class SubjectData:
    runs: Path
    staged: Path
    subject: str
    tasks: list[str]
    cfg: DecodingConfig
    _cache: dict[str, TrialSet] = field(default_factory=dict)

    def _all(self, cfg: DecodingConfig, eeg_only: bool) -> TrialSet:
        return TrialSet.concat([load_trialset(self.runs, self.staged, self.subject, t, cfg,
                                              eeg_only=eeg_only) for t in self.tasks])

    @cached_property
    def full(self) -> TrialSet:  # EEG + EOG, all trials
        return self._all(self.cfg, eeg_only=False)

    def _eeg(self, ts: TrialSet) -> TrialSet:
        names = [c for c, t in zip(ts.ch_names, ts.info.get_channel_types(), strict=True)
                 if t == "eeg"]
        return ts.pick(names)

    @cached_property
    def main(self) -> TrialSet:
        return labels.main_set(self._eeg(self.full))

    @cached_property
    def catch(self) -> TrialSet:
        return labels.catch_set(self._eeg(self.full))

    @cached_property
    def eog(self) -> TrialSet:
        return labels.main_set(self.full.pick(["VEOG", "HEOG"]))

    @cached_property
    def alpha_src(self) -> TrialSet:
        cfg = self.cfg.model_copy(update={"sfreq": 256.0, "window_s": (-1.0, 1.0)})
        return labels.main_set(self._eeg(self._all(cfg, eeg_only=False)))

    @cached_property
    def noica(self) -> TrialSet:
        sets = []
        for task in self.tasks:
            ep = mne.read_epochs(epochs_path(self.runs, self.subject, task,
                                             self.cfg.epoch_set), preload=False,
                                 verbose="error")
            raw = mne.io.read_raw_fif(
                self.runs / task / "derivatives" / "eegunify-preproc" / f"sub-{self.subject}"
                / "eeg" / f"sub-{self.subject}_task-{task}_desc-preproc_eeg.fif",
                preload=True, verbose="error")
            pe = mne.Epochs(raw, ep.events, ep.event_id, tmin=ep.tmin, tmax=ep.tmax,
                            baseline=None, reject=None, flat=None, preload=True,
                            on_missing="ignore",
                            verbose="error")
            pe = attach_metadata(pe, staged_events(self.staged, self.subject, task))
            meta = meta_from_staged(pe.metadata, self.subject, task)
            pe = prepare(pe, self.cfg).pick("eeg")
            sets.append(TrialSet((pe.get_data() * 1e6).astype(np.float32), pe.times.copy(),
                                 tuple(pe.ch_names), float(pe.info["sfreq"]), meta, pe.info))
        return labels.main_set(TrialSet.concat(sets))


Analysis = Callable[[SubjectData], dict[str, np.ndarray]]


def _per_task(d: SubjectData, fn: Callable[[TrialSet], np.ndarray]) -> dict[str, np.ndarray]:
    out = {"pooled": fn(d.main)}
    for task in d.tasks:
        sub = d.main.select(d.main.meta["task"].to_numpy() == task)
        if labels.enough_trials(sub, d.cfg.min_trials_per_class)[0]:
            out[task] = fn(sub)
    return out


def a1_sliding(d: SubjectData) -> dict[str, np.ndarray]:
    return _per_task(d, lambda ts: within.sliding_auc(ts, d.cfg))


def a2_generalization(d: SubjectData) -> dict[str, np.ndarray]:
    return {"pooled": within.generalization_auc(d.main, d.cfg)}


def a3_patterns(d: SubjectData) -> dict[str, np.ndarray]:
    return {"patterns": within.patterns(d.main, d.cfg)}


def a4_within_bin(d: SubjectData) -> dict[str, np.ndarray]:
    auc, used = confounds.within_bin_auc(d.main, d.cfg)
    return {"pooled": auc, "n_used": np.array(sum(used.values()))}


def a5_contrast(d: SubjectData) -> dict[str, np.ndarray]:
    cd = confounds.contrast_decoding(d.main, d.cfg)
    return {"residualized": confounds.residualized_auc(d.main, d.cfg), **cd}


def c3_eog(d: SubjectData) -> dict[str, np.ndarray]:
    reps = within.splits_for(d.main, d.cfg)
    keep = ~confounds.heog_step_mask(d.eog.pick(["HEOG"]).X[:, 0], d.eog.times, HEOG_STEP_UV,
                                     HEOG_WINDOW_S)
    out = {"eog_only": confounds.eog_only_auc(d.eog, d.cfg, reps=reps),
           "no_ica": within.sliding_auc(d.noica, d.cfg, reps=reps),
           "n_heog_dropped": np.array(int((~keep).sum()))}
    if labels.enough_trials(d.main.select(keep), d.cfg.min_trials_per_class)[0]:
        out["heog_clean"] = within.sliding_auc(d.main.select(keep), d.cfg)
    return out


def c4_alpha(d: SubjectData) -> dict[str, np.ndarray]:
    r = confounds.alpha_models(d.main, d.cfg, WIN_S, alpha_ts=d.alpha_src)
    s = confounds.alpha_split_scores(d.main, d.cfg, WIN_S, alpha_ts=d.alpha_src)
    return {k: np.array(v) for k, v in {**r, **s}.items()}


def c5_balance_catch(d: SubjectData) -> dict[str, np.ndarray]:
    u = confounds.undersampled_auc(d.main, d.cfg, WIN_S, n_draws=20)
    c = confounds.catch_scores(d.main, d.catch, d.cfg, WIN_S) if len(d.catch) else None
    out = {"undersampled": u}
    if c is not None:
        out |= {"catch_scores": c["scores"], "catch_fa": d.catch.labels("seen"),
                "auc_fa_vs_cr": np.array(c["auc_fa_vs_cr"])}
    return out


def a8_cross_task(d: SubjectData) -> dict[str, np.ndarray]:
    if len(d.tasks) < 2:
        return {}
    m, tasks, n_train = across.cross_task_within(d.main, d.cfg, d.tasks)
    return {"matrix": m, "tasks": np.array(tasks), "n_train": np.array(n_train)}


def a9_cross_condition(d: SubjectData) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    lat = d.main.select(d.main.meta["hemifield"].isin(["left", "right"]).to_numpy())
    if len(lat) and labels.enough_trials(lat, 10)[0]:
        out["hemifield"] = across.cross_condition_auc(lat, d.cfg, "hemifield", "left",
                                                      "right")["mean"]
        right = lat.meta["hemifield"].to_numpy() == "right"
        mir = channels.mirror(lat)  # right-field trials mirrored to left-field geometry
        x = np.where(right[:, None, None], mir.X, lat.X)
        out["hemifield_mirrored"] = across.cross_condition_auc(
            TrialSet(x, lat.times, lat.ch_names, lat.sfreq, lat.meta, lat.info), d.cfg,
            "hemifield", "left", "right")["mean"]
    out["orientation"] = across.cross_condition_auc(d.main, d.cfg, "orientation", "vertical",
                                                    "horizontal")["mean"]
    return out


ANALYSES: dict[str, Analysis] = {
    "a1_sliding": a1_sliding, "a2_generalization": a2_generalization,
    "a3_patterns": a3_patterns, "a4_within_bin": a4_within_bin, "a5_contrast": a5_contrast,
    "c3_eog": c3_eog, "c4_alpha": c4_alpha, "c5_balance_catch": c5_balance_catch,
    "a8_cross_task": a8_cross_task, "a9_cross_condition": a9_cross_condition,
}
"""A6 (response key) is not registered: ds006171 has no key information (decision 3)."""
