"""Stage one ds006171 recording: photodiode onsets, labels, contrast matching (PHASE1 D1-D7).

Pipeline: Status events (timing) and events.tsv (behaviour) are aligned code by code; one row
is built per stimulus; a new events.tsv with composite codes is written next to a symlink of
the BDF, which eegunify then reads with ``events.source: bids_events``.

Data facts this relies on (docs/DATASETS.md, Step 3 checks):

- codes outside the task's known set are trigger glitches (sub-23 nocue: one code 6) and are
  dropped from both sources before alignment;
- events.tsv onsets have 6 significant digits (10 ms precision after 1000 s), so the
  cross-check tolerance is 5 ms and the Status onsets are used;
- the sampling rate is 1024 Hz except in 4 recordings (2048 Hz): always use the file's own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mne
import numpy as np
import pandas as pd

from start_vision import codes as C  # noqa: N812
from start_vision.matching import balance_report, match

log = logging.getLogger(__name__)

ONSET_TOL_S = 0.005  # events.tsv precision (see module docstring)
PHOTODIODE_WINDOW_S = (0.010, 0.120)  # D3
SEED = 97


class StageError(RuntimeError):
    """The two event sources disagree, or a report contradicts subjective_r."""


def read_status_events(bdf: Path) -> pd.DataFrame:
    """Status events (onset_s, sample, code), code masked to the low byte as eegunify does."""
    raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="error", include=["Status"])
    ev = mne.find_events(
        raw, stim_channel="Status", shortest_event=1, initial_event=True, verbose="error"
    )
    code = ev[:, 2] & 0xFF
    keep = code != 0
    sfreq = float(raw.info["sfreq"])
    sample = ev[keep, 0] - raw.first_samp
    return _with_sfreq(
        pd.DataFrame({"onset_s": sample / sfreq, "sample": sample, "code": code[keep]}), sfreq
    )


def _with_sfreq(df: pd.DataFrame, sfreq: float) -> pd.DataFrame:
    df.attrs["sfreq"] = sfreq
    return df


def read_bids_events(tsv: Path) -> pd.DataFrame:
    """events.tsv without its corrupted ``sample`` column; ``code`` is the integer ``value``."""
    df = pd.read_csv(tsv, sep="\t").drop(columns=["sample"], errors="ignore")
    df["code"] = df["value"].astype(int)
    return df


def drop_unknown(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, list[int]]:
    known = C.known_codes(task)
    bad = ~df["code"].isin(known)
    return df[~bad].reset_index(drop=True), [int(x) for x in df.loc[bad, "code"]]


def align(status: pd.DataFrame, bids: pd.DataFrame) -> pd.DataFrame:
    """Join row by row; code sequences must be identical and onsets within ONSET_TOL_S."""
    if len(status) != len(bids) or not np.array_equal(status["code"], bids["code"]):
        n = min(len(status), len(bids))
        sc, bc = status["code"].to_numpy(), bids["code"].to_numpy()
        i = next((k for k in range(n) if sc[k] != bc[k]), n)
        raise StageError(
            f"code sequences differ at event {i}: Status {status['code'].iloc[i:i + 3].tolist()}"
            f" vs events.tsv {bids['code'].iloc[i:i + 3].tolist()}"
            f" (lengths {len(status)} / {len(bids)})"
        )
    d = np.abs(status["onset_s"].to_numpy() - bids["onset"].to_numpy())
    if d.max() > ONSET_TOL_S + 1e-9:
        i = int(d.argmax())
        raise StageError(f"onset of event {i} differs by {d[i] * 1000:.2f} ms")
    out = bids.drop(columns=["onset", "code"]).copy()
    out.insert(0, "onset_s", status["onset_s"].to_numpy())
    out.insert(1, "code", status["code"].to_numpy())
    out.attrs["sfreq"] = status.attrs.get("sfreq")
    return out


def build_trials(events: pd.DataFrame, task: str) -> pd.DataFrame:
    """One row per stimulus code (D3-D5): photodiode onset, report, orientation, class."""
    stim = C.RAW_STIMULI[task]
    t = events["onset_s"].to_numpy(dtype=float)
    code = events["code"].to_numpy(dtype=int)
    nan = np.full(len(events), np.nan)
    tn = events["trial_number"].to_numpy(dtype=float) if "trial_number" in events else nan
    gc = events["gabor_contrast"].to_numpy(dtype=float)
    subj = events["subjective_r"].to_numpy(dtype=float)
    obj = events["objective_r"].to_numpy(dtype=float)
    is_stim = np.isin(code, list(stim))
    stim_idx = np.flatnonzero(is_stim)
    rows = []
    for n, i in enumerate(stim_idx):
        end = stim_idx[n + 1] if n + 1 < len(stim_idx) else len(code)
        rs = stim[int(code[i])]
        r: dict[str, object] = {
            "raw_code": int(code[i]),
            "code_onset_s": float(t[i]),
            "side": rs.side,
            "orient": rs.orient,
            "cue_valid": rs.cue_valid,
            "trial_number": tn[i],
            "gabor_contrast": gc[i],
            "subjective_r": subj[i],
            "objective_r": obj[i],
        }
        lo, hi = t[i] + PHOTODIODE_WINDOW_S[0], t[i] + PHOTODIODE_WINDOW_S[1]
        pd_hits = [j for j in range(i + 1, end) if code[j] == C.PHOTODIODE and lo <= t[j] <= hi]
        reps = [j for j in range(i + 1, end) if code[j] in (C.REPORT_SEEN, C.REPORT_UNSEEN)]
        r["onset_s"] = float(t[pd_hits[0]]) if pd_hits else np.nan
        r["photodiode_delay_ms"] = (t[pd_hits[0]] - t[i]) * 1000 if pd_hits else np.nan
        report = int(code[reps[0]]) if reps else None
        r["report"] = report
        r["report_onset_s"] = float(t[reps[0]]) if reps else np.nan
        after = reps[0] if reps else i
        ori = [
            j for j in range(after + 1, end)
            if code[j] in (C.ORIENT_VERTICAL, C.ORIENT_HORIZONTAL)
        ]
        r["orient_response"] = (
            C.VERTICAL if ori and code[ori[0]] == C.ORIENT_VERTICAL
            else C.HORIZONTAL if ori else None
        )
        r["orient_onset_s"] = float(t[ori[0]]) if ori else np.nan
        reason = "" if pd_hits else "no_photodiode"
        if report is None:
            reason = reason or "no_report"
        if report is not None and np.isfinite(subj[i]):
            expected = C.REPORT_SEEN if int(subj[i]) == 1 else C.REPORT_UNSEEN
            if report != expected:
                raise StageError(
                    f"trial {r['trial_number']} (event {i}): report {report} but subjective_r "
                    f"{r['subjective_r']}"
                )
        seen = report == C.REPORT_SEEN
        if reason:
            r["cls"] = C.EXCLUDED
        elif rs.catch:
            r["cls"] = C.FA if seen else C.CR
        else:
            r["cls"] = C.SEEN if seen else C.UNSEEN
        r["exclusion"] = reason
        rows.append(r)
    df = pd.DataFrame(rows)
    gabor = df["orient"] != C.NO_GABOR
    answered = (gabor & df["orient_response"].notna()).to_numpy()
    correct = (df["orient_response"] == df["orient"]).to_numpy()
    df["objective_correct"] = [
        bool(c) if a else None for a, c in zip(answered, correct, strict=True)
    ]
    df["rt_ms"] = (df["report_onset_s"] - df["onset_s"]) * 1000
    return df


def staged_events(trials: pd.DataFrame, events: pd.DataFrame, task: str) -> pd.DataFrame:
    """The staged events.tsv: one composite-coded row per stimulus, plus the response events."""
    rows: list[dict[str, Any]] = []
    recs: list[dict[Any, Any]] = trials.to_dict("records")
    for d in recs:
        r = SimpleNamespace(**d)
        no_pd = not np.isfinite(r.onset_s)
        onset = r.code_onset_s if no_pd else r.onset_s  # excluded without photodiode: code onset
        value = C.encode(int(r.cls), int(r.side), int(r.orient), bool(r.matched))
        rows.append(
            {
                "onset": onset, "duration": 0.0, "value": value,
                "trial_type": C.trial_type(value), "trial_number": r.trial_number,
                "gabor_contrast": r.gabor_contrast, "subjective_r": r.subjective_r,
                "objective_r": r.objective_r, "objective_correct": r.objective_correct,
                "cue_valid": r.cue_valid, "photodiode_delay_ms": r.photodiode_delay_ms,
                "rt_ms": r.rt_ms, "raw_code": r.raw_code, "exclusion": r.exclusion, "task": task,
            }
        )
    resp = events[events["code"].isin(list(C.RESPONSES))]
    for o, c in zip(resp["onset_s"], resp["code"], strict=True):
        rows.append(
            {"onset": o, "duration": 0.0, "value": C.RESPONSE_OFFSET + int(c),
             "trial_type": C.RESPONSES[int(c)],
             "raw_code": int(c), "task": task}
        )
    out = pd.DataFrame(rows).sort_values("onset", kind="stable").reset_index(drop=True)
    out["onset"] = out["onset"].round(6)
    return out


@dataclass
class StageReport:
    subject: str
    task: str
    sfreq: float
    n_trials: int
    counts: dict[str, int]
    excluded: dict[str, int]
    dropped_codes: list[int]
    delay_ms: dict[str, float]
    balance: dict[str, float]
    input_sha1: str
    extra: dict[str, object] = field(default_factory=dict)


def _sha1(*paths: Path) -> str:
    h = hashlib.sha1()
    for p in paths:
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def stage_trials(bdf: Path, task: str) -> tuple[pd.DataFrame, pd.DataFrame, list[int], float]:
    status = read_status_events(bdf)
    sfreq = float(status.attrs["sfreq"])
    bids = read_bids_events(bdf.with_name(bdf.name.replace("_eeg.bdf", "_events.tsv")))
    status, dropped_s = drop_unknown(status, task)
    status.attrs["sfreq"] = sfreq
    bids, dropped_b = drop_unknown(bids, task)
    events = align(status, bids)
    trials = build_trials(events, task)
    trials["matched"] = match(trials, seed=SEED)
    return trials, events, sorted(dropped_s + dropped_b), sfreq


def stage_recording(src_bdf: Path, dst_dir: Path, task: str, subject: str) -> StageReport:
    """Write the staged recording into ``dst_dir`` and return its report. Idempotent."""
    tsv = src_bdf.with_name(src_bdf.name.replace("_eeg.bdf", "_events.tsv"))
    sha = _sha1(src_bdf, tsv)
    dst_dir.mkdir(parents=True, exist_ok=True)
    report_path = dst_dir / src_bdf.name.replace("_eeg.bdf", "_stage.json")
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old.get("input_sha1") == sha:
            return StageReport(**old)
    trials, events, dropped, sfreq = stage_trials(src_bdf, task)
    out = staged_events(trials, events, task)
    out.to_csv(dst_dir / tsv.name, sep="\t", index=False, na_rep="n/a")
    (dst_dir / tsv.name.replace(".tsv", ".json")).write_text(
        json.dumps({"value": {"Description": "start_vision.codes composite code",
                              "Levels": C.eegunify_mapping()}}, indent=1)
    )
    link = dst_dir / src_bdf.name
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(src_bdf.resolve(), link)
    for side in src_bdf.parent.glob(src_bdf.name.replace("_eeg.bdf", "_*")):
        if side.suffix in (".json", ".tsv") and "_events." not in side.name:
            shutil.copy2(side, dst_dir / side.name)
    for el in src_bdf.parent.glob("*_electrodes.tsv"):
        shutil.copy2(el, dst_dir / el.name)
    labels = {C.SEEN: "seen", C.UNSEEN: "unseen", C.FA: "fa", C.CR: "cr", C.EXCLUDED: "excluded"}
    d = trials["photodiode_delay_ms"].dropna()
    delays = {"min": float(d.min()), "median": float(d.median()), "max": float(d.max())} if len(
        d
    ) else {}
    reasons = trials.loc[trials["exclusion"] != "", "exclusion"].value_counts()
    rep = StageReport(
        subject=subject, task=task, sfreq=sfreq, n_trials=len(trials),
        counts={labels[k]: int((trials["cls"] == k).sum()) for k in labels},
        excluded={str(k): int(v) for k, v in reasons.items()},
        dropped_codes=dropped,
        delay_ms=delays,
        balance=balance_report(trials, trials["matched"]),
        input_sha1=sha,
        extra={"n_matched": int(trials["matched"].sum()),
               "objective_accuracy_seen": _acc(trials, C.SEEN),
               "objective_accuracy_unseen": _acc(trials, C.UNSEEN)},
    )
    report_path.write_text(json.dumps(asdict(rep), indent=1))
    return rep


def _acc(trials: pd.DataFrame, cls: int) -> float:
    sel = (trials["cls"] == cls) & trials["objective_correct"].notna()
    x = trials.loc[sel, "objective_correct"]
    return float(x.astype(bool).mean()) if len(x) else float("nan")
