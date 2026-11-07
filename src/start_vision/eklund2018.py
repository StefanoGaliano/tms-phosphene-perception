"""Stage Eklund & Wiens 2018 for the cross-dataset test (A12 issue #1; started 2026-11-07).

Data: figshare doi:10.17045/sthlmuni.5418967 (CC BY 4.0), raw BioSemi BDF
``VAN_Exp_<fp>_<sess>.bdf`` (64 ch A1-B32 + Nz, 512 Hz) and logs ``txt/VAN_Exp_<fp>_<sess>.txt``.
Paper: Eklund & Wiens 2018,
Cogn Affect Behav Neurosci 18:176-188, doi:10.3758/s13415-018-0562-z.

Everything here follows the authors' own code (``fieldtrip/trialfun_VAN.m``, ``analyze_VAN.m``):

- Status: ``trialfun_VAN.m`` reads up-going flanks of the Status value. Bit 0 = Light1
  (photodiode onset), bit 1 = Light2 (photodiode offset), bits 10-15 = TTL lines 1-6. A
  condition event has Light1 off and one TTL pattern: 1-4 critical UL/UR/LL/LR (TTL 1, 2, 3, 4),
  5-8 control (TTL 1+2, 2+3, 3+4, 4+5), 9 catch (TTL 6), 10 block (all six). The stimulus onset
  is the first Light1 flank within the next 6 events, its offset the first Light2 flank.
- The log's ``condition`` column must equal the Status sequence (580 trials, 5 blocks).
- Labels (``analyze_VAN.m``): aware correct = ``obj_corr == 1`` and ``Subj_resp > 1``; unaware
  correct = ``obj_corr == 1`` and ``Subj_resp == 1``; unaware incorrect = ``obj_corr == 0`` and
  ``Subj_resp == 1``. Trials whose photodiode duration is off the condition's mode by >= 2 ms
  are dropped (``analyze_VAN.m``, "keep only trials with good duration").
- Session inclusion (``analyze_VAN.m``, ``goodfps``): >= 5 trials in every quadrant for aware
  correct and unaware correct (``GMn``), control hit rate >= .70 (``GMnBeh.Hitrate``), and not
  in the bad control-N1 lists (small 9 14 16 31 32 33; large 27 31 32 33). Reproduces
  S = 15, L = 24, both = 13 (the code comment and the paper).

Each person's large- and small-Gabor sessions are staged as eegunify tasks ``large`` and
``small`` (sessions differ in Gabor size; order counterbalanced, from ``allGM.mat``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

QUADRANTS = {1: "UL", 2: "UR", 3: "LL", 4: "LR", 5: "UL", 6: "UR", 7: "LL", 8: "LR"}
# TTL bit pattern (lines 1-6 -> bits 0-5 of the TTL byte) -> condition, trialfun_VAN.m
TTL_TO_CONDITION = {0b000001: 1, 0b000010: 2, 0b000100: 3, 0b001000: 4,
                    0b000011: 5, 0b000110: 6, 0b001100: 7, 0b011000: 8,
                    0b100000: 9, 0b111111: 10}
N_TRIALS, N_BLOCKS, K_LOOKAHEAD = 580, 5, 6
DURATION_TOL_MS = 2.0
BAD_N1 = {"small": {9, 14, 16, 31, 32, 33}, "large": {27, 31, 32, 33}}
MIN_TRIALS_PER_QUADRANT, MIN_CONTROL_HITRATE = 5, 0.70

# composite event values for eegunify (class * 10 + quadrant)
CLASSES = {1: "aware", 2: "unaware", 3: "unaware_incorrect", 4: "aware_incorrect",
           5: "control", 6: "catch", 7: "excluded"}
QUAD_CODE = {"UL": 1, "UR": 2, "LL": 3, "LR": 4, "none": 0}


class EklundError(RuntimeError):
    """Status and log disagree, or a check of the authors' code fails."""


def trial_type(value: int) -> str:
    cls, q = divmod(value, 10)
    quad = {v: k for k, v in QUAD_CODE.items()}[q]
    side = {"UL": "left", "LL": "left", "UR": "right", "LR": "right", "none": "none"}[quad]
    return f"stimulus/{CLASSES[cls]}/{side}/{quad}"


def eegunify_mapping() -> dict[str, str]:
    vals = [c * 10 + q for c in CLASSES for q in (1, 2, 3, 4)] + [60, 70]
    return {str(v): trial_type(v) for v in vals}


def status_events(status: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Up-going flanks of the 16-bit Status value, as ft_read_event returns them for BioSemi."""
    v = status.astype(np.int64) & 0xFFFF
    idx = np.flatnonzero(v[1:] > v[:-1]) + 1
    val = v[idx]
    return pd.DataFrame({"sample": idx, "onset_s": idx / sfreq, "light1": val & 1,
                         "light2": (val >> 1) & 1, "ttl": (val >> 10) & 0x3F})


def find_trials(ev: pd.DataFrame, sfreq: float) -> pd.DataFrame:
    """Condition events with photodiode onset/offset (trialfun_VAN.m)."""
    cond = np.where(ev["light1"].to_numpy() == 0,
                    ev["ttl"].map(TTL_TO_CONDITION).to_numpy(dtype=float), np.nan)
    if int(np.sum(cond == 10)) != N_BLOCKS:
        raise EklundError(f"{int(np.sum(cond == 10))} block codes, expected {N_BLOCKS}")
    rows = []
    l1, l2, smp = ev["light1"].to_numpy(), ev["light2"].to_numpy(), ev["sample"].to_numpy()
    for i in np.flatnonzero(cond < 10):
        nxt = slice(i, min(i + K_LOOKAHEAD, len(ev)))
        on = np.flatnonzero(l1[nxt])
        off = np.flatnonzero(l2[nxt])
        if not len(on) or not len(off):
            raise EklundError(f"event row {i}: no photodiode flank within {K_LOOKAHEAD} events")
        s_on, s_off = smp[i + on[0]], smp[i + off[0]]
        rows.append({"condition": int(cond[i]), "code_sample": int(smp[i]),
                     "sample": int(s_on), "onset_s": s_on / sfreq,
                     "delay_ms": (s_on - smp[i]) / sfreq * 1000,
                     "duration_ms": (s_off - s_on) / sfreq * 1000})
    t = pd.DataFrame(rows)
    if (t["code_sample"].to_numpy()[1:] <= t["sample"].to_numpy()[:-1] + (
            t["duration_ms"].to_numpy()[:-1] * sfreq / 1000)).any():
        raise EklundError("light sensor after the next trial")
    return t


def read_log(txt: Path) -> pd.DataFrame:
    """The authors' log: one row per trial (obj_resp/corrAns are text; trialfun skips them)."""
    log = pd.read_csv(txt, sep="\t")
    need = {"FP", "Session", "trialNr", "condition", "blockNr", "frames", "obj_corr",
            "Subj_resp", "obj_resp_time", "sub_resp_time"}
    if missing := need - set(log.columns):
        raise EklundError(f"{txt.name}: log lacks {sorted(missing)}")
    return log


def label(trials: pd.DataFrame) -> pd.DataFrame:
    """Class per trial (analyze_VAN.m) and the duration exclusion (mode +- 2 ms, per condition)."""
    t = trials.copy()
    crit = t["condition"].between(1, 4)
    oc, sr = t["obj_corr"], t["Subj_resp"]
    cls = np.select(
        [crit & (oc == 1) & (sr > 1), crit & (oc == 1) & (sr == 1),
         crit & (oc == 0) & (sr == 1), crit & (oc == 0) & (sr > 1),
         t["condition"].between(5, 8), t["condition"] == 9],
        ["aware", "unaware", "unaware_incorrect", "aware_incorrect", "control", "catch"],
        default="excluded")
    t["cls"] = cls
    t["quadrant"] = t["condition"].map(QUADRANTS).fillna("none")
    t["exclusion"] = np.where(t["cls"] == "excluded", "label", "")
    for _, g in t.groupby("condition"):
        mode = g["duration_ms"].round(3).mode().iloc[0]
        dur = g["duration_ms"]
        bad = (dur <= mode - DURATION_TOL_MS) | (dur >= mode + DURATION_TOL_MS)
        t.loc[g.index[bad.to_numpy()], "exclusion"] = "duration"
    return t


def stage_trials(bdf: Path, txt: Path) -> tuple[pd.DataFrame, float]:
    raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="error", include=["Status"])
    sfreq = float(raw.info["sfreq"])
    st = raw.get_data(picks=["Status"])[0]
    trials = find_trials(status_events(st, sfreq), sfreq)
    log = read_log(txt)
    if len(trials) != N_TRIALS or len(log) != N_TRIALS:
        raise EklundError(f"{bdf.name}: {len(trials)} Status / {len(log)} log trials, "
                          f"expected {N_TRIALS}")
    if not np.array_equal(trials["condition"].to_numpy(), log["condition"].to_numpy()):
        raise EklundError(f"{bdf.name}: Status condition sequence differs from the log")
    t = pd.concat([trials, log.drop(columns=["condition"]).reset_index(drop=True)], axis=1)
    return label(t), sfreq


def staged_events(t: pd.DataFrame, gabor: str) -> pd.DataFrame:
    cls_code = {v: k for k, v in CLASSES.items()}
    value = [(cls_code["excluded"] if e else cls_code[c]) * 10 + QUAD_CODE[q]
             for c, q, e in zip(t["cls"], t["quadrant"], t["exclusion"] != "", strict=True)]
    return pd.DataFrame({
        "onset": t["onset_s"].round(6), "duration": 0.0, "value": value,
        "trial_type": [trial_type(v) for v in value], "trial_number": t["trialNr"],
        "condition": t["condition"], "cls": t["cls"], "quadrant": t["quadrant"],
        "rating": t["Subj_resp"], "obj_corr": t["obj_corr"], "frames": t["frames"],
        "photodiode_delay_ms": t["delay_ms"].round(3),
        "duration_ms": t["duration_ms"].round(3), "exclusion": t["exclusion"], "gabor": gabor,
    })


@dataclass
class StageReport:
    fp: int
    session: int
    gabor: str
    sfreq: float
    counts: dict[str, int]
    n_duration_excluded: int
    delay_ms: dict[str, float]


def stage_session(bdf: Path, txt: Path, dst_root: Path, fp: int, session: int,
                  gabor: str) -> StageReport:
    """Write ``sub-<fp>/eeg/sub-<fp>_task-<gabor>_{eeg.bdf (symlink),events.tsv,stage.json}``."""
    t, sfreq = stage_trials(bdf, txt)
    sub = f"{fp:02d}"
    d = dst_root / f"sub-{sub}" / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{sub}_task-{gabor}"
    staged_events(t, gabor).to_csv(d / f"{stem}_events.tsv", sep="\t", index=False,
                                   na_rep="n/a")
    link = d / f"{stem}_eeg.bdf"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(bdf.resolve(), link)
    ok = t["exclusion"] == ""
    rep = StageReport(fp=fp, session=session, gabor=gabor, sfreq=sfreq,
                      counts={str(k): int(v) for k, v in t.loc[ok, "cls"].value_counts().items()},
                      n_duration_excluded=int((t["exclusion"] == "duration").sum()),
                      delay_ms={"min": float(t["delay_ms"].min()),
                                "median": float(t["delay_ms"].median()),
                                "max": float(t["delay_ms"].max())})
    (d / f"{stem}_stage.json").write_text(json.dumps(asdict(rep), indent=1))
    return rep


def sessions_table(fieldtrip_dir: Path) -> pd.DataFrame:
    """Per session: Gabor size, the authors' counts and their inclusion rule (``goodfps``)."""
    import scipy.io as sio

    gm = sio.loadmat(fieldtrip_dir / "allGM.mat", simplify_cells=True)
    hit = sio.loadmat(fieldtrip_dir / "GMnBeh.mat", simplify_cells=True)["Hitrate"]
    gmn = np.asarray(gm["GMn"], dtype=float)
    # GMn(:,[3,4,6,7,9,10,12,13]): aware/unaware correct per quadrant (analyze_VAN.m)
    ntr = gmn[:, [2, 3, 5, 6, 8, 9, 11, 12]].reshape(len(gmn), 2, 4, order="F")
    rows: list[dict[str, Any]] = []
    for k, g in enumerate(gm["GM"]):
        size = {"S": "small", "L": "large"}[g["gabor"]]
        nmin = int(ntr[k].min())
        rows.append({"fp": int(g["fp"]), "session": int(g["sess"]), "gabor": size,
                     "authors_ntr_min": nmin, "control_hitrate": float(hit[k]),
                     "authors_included": bool(nmin >= MIN_TRIALS_PER_QUADRANT
                                              and hit[k] >= MIN_CONTROL_HITRATE
                                              and int(g["fp"]) not in BAD_N1[size])})
    return pd.DataFrame(rows)


def ab_to_1020(fieldtrip_dir: Path) -> dict[str, str]:
    """BioSemi positional names A1-B32 -> 10-20 names (the authors' Biosemi_64_AB_to_1020.mat)."""
    import scipy.io as sio

    m = sio.loadmat(fieldtrip_dir / "Biosemi_64_AB_to_1020.mat", simplify_cells=True)
    return {str(a): str(b) for a, b in m["BiosemiAB_to_1020"]}


def eegunify_config(gabor: str, subjects: list[str], rename: dict[str, str]) -> dict[str, Any]:
    """eegunify config for one Gabor size; preprocessing as Phase 1 (configs/template_ds006171.yaml)
    so the frozen detector sees the same kind of input. Differences are noted inline."""
    eeg = list(rename.values())
    return {
        "schema_version": 1,
        "dataset": {"name": f"eklund2018 {gabor}", "bids_root": "rawdata",
                    "source_root": "../../../sourcedata/staged/eklund2018", "task": gabor,
                    "authors": ["Eklund", "Wiens"], "license": "CC BY 4.0"},
        "recordings": {"pattern": f"sub-{{subject}}/eeg/sub-{{subject}}_task-{gabor}_eeg.bdf",
                       "subjects": subjects},
        "reference": {"online": "CMS/DRL (reference-free)", "scheme": "average"},
        "channels": {
            "rename": rename | {"Nz": "Nz"},
            # Nz: a real electrode the authors kept in their average reference; the detector
            # was trained without it, so it is misc here (not EEG, not in the reference).
            "types": {"Nz": "misc"},
            # EXG1-8: not connected in this dataset (flat, saturated offset; checked on 1_1).
            "drop": [f"EXG{i}" for i in range(1, 9)]
                    + ["GSR1", "GSR2", "Erg1", "Erg2", "Resp", "Plet", "Temp"],
        },
        "montage": {"source": "standard", "standard_name": "standard_1005", "on_missing": "raise"},
        "events": {"source": "bids_events", "mapping": eegunify_mapping(), "unmapped": "error",
                   "expected_counts": {}},
        "preproc": {"resample_hz": 256, "highpass_hz": 0.1, "lowpass_hz": None,
                    "filter": {"method": "iir", "iir_order": 2, "phase": "zero"},
                    "bad_channels": {"method": "pyprep_robust"}, "interpolate": True},
        "artifacts": {
            "method": "ica",
            # no EOG electrodes: VEOG from frontal minus central EEG (the authors built their
            # blink channel from frontal sites); HEOG F7-F8 as Phase 1 (decision 7)
            "eog_bipolar": {"VEOG": ["Fpz", "Cz"], "HEOG": ["F7", "F8"]},
            "ica": {"fit_exclude": {}},
            "post_bad_channels": {"max_ptp_ratio": 2.0},
        },
        "epochs": {
            "sets": {
                "stim": {"lock": "stimulus/*", "tmin_ms": -1000, "tmax_ms": 1000,
                         "baseline_ms": [-100, 0]},
                "stimraw": {"lock": "stimulus/*", "tmin_ms": -1000, "tmax_ms": 1000,
                            "baseline_ms": None},
            },
            "reject": {"method": "rules", "rules": [
                {"test": "ptp", "channels": eeg, "threshold_uv": 100,
                 "window_ms": [-100, 600], "win_ms": 200, "step_ms": 100},
                {"test": "step", "channels": ["HEOG"], "threshold_uv": 32,
                 "window_ms": [-50, 350], "win_ms": 100, "step_ms": 10},
            ]},
        },
        "conditions": {
            "aware": ["stimulus/aware"], "unaware": ["stimulus/unaware"],
            "aware_left": ["stimulus/aware/left"], "aware_right": ["stimulus/aware/right"],
            "unaware_left": ["stimulus/unaware/left"],
            "unaware_right": ["stimulus/unaware/right"],
        },
        "contrasts": {
            "aware_minus_unaware": {"plus": "aware", "minus": "unaware"},
            "aware_lat": {"type": "lateralized", "left": "aware_left", "right": "aware_right",
                          "pairs": [["O1", "O2"]], "contra_wave": "aware_contra",
                          "ipsi_wave": "aware_ipsi"},
            "unaware_lat": {"type": "lateralized", "left": "unaware_left",
                            "right": "unaware_right", "pairs": [["O1", "O2"]],
                            "contra_wave": "unaware_contra", "ipsi_wave": "unaware_ipsi"},
        },
        "components": {"library_paths": ["../../../configs/components/awareness_eklund2018.yaml"],
                       "preset": "awareness_eklund2018", "use": ["van", "lp"]},
        "group": {"tests": ["paired_t", "wilcoxon"],
                  "cluster": {"enabled": True, "contrast": "aware_minus_unaware", "tmin_ms": 0,
                              "tmax_ms": 800, "n_permutations": 5000, "tail": 0},
                  "reference_values": "none"},
    }
