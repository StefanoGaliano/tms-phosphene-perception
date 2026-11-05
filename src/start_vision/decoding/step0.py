"""PHASE2 §7 Step 0 on real data: data contract, inclusion, descriptives (build step 13).

For every subject x task recording: load the TrialSet (``data.load_trialset``), check the §4
contract as built (128 EEG channels with the same names everywhere, sfreq 128 Hz, the
-500..1000 ms grid, required meta columns filled), then decide inclusion:

1. Phase 1 rules (``results/phase1/exclusions.tsv``): subject rule (user decision 2: pooled hit
   rate 30-70 %, FA <= 20 %) and recording rules (> 45 % epochs rejected; < 30 matched epochs or
   epochs stage failed). Kept so both phases analyse the same recordings.
2. PHASE2 §6: >= ``min_trials_per_class`` seen and unseen stimulus-present trials after rejection.

Writes ``results/phase2/inclusion.tsv`` (one row per recording, reason when excluded) and
``results/phase2/stats_descriptives.tsv`` (T1, all loaded recordings, with an ``included`` column).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from start_vision.decoding import labels
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import epochs_path, load_trialset
from start_vision.decoding.figures import t1_descriptives

REQUIRED_FILLED = ("subject", "task", "trial_number", "stim_present", "seen", "hemifield",
                   "orientation", "rt_ms")


def contract_problems(ts: object, cfg: DecodingConfig, ref_channels: tuple[str, ...] | None
                      ) -> list[str]:
    from start_vision.decoding.data import TrialSet

    assert isinstance(ts, TrialSet)
    out = []
    if ts.X.shape[1] != 128:
        out.append(f"{ts.X.shape[1]} channels")
    if ref_channels is not None and ts.ch_names != ref_channels:
        out.append("channel names/order differ")
    if abs(ts.sfreq - cfg.sfreq) > 1e-9:
        out.append(f"sfreq {ts.sfreq}")
    n_expected = round((cfg.window_s[1] - cfg.window_s[0]) * cfg.sfreq)
    if len(ts.times) != n_expected or abs(ts.times[0] - cfg.window_s[0]) > 1e-9:
        out.append(f"time grid {ts.times[0]:.4f}+{len(ts.times)} samples")
    for c in REQUIRED_FILLED:
        if ts.meta[c].isna().any():
            out.append(f"meta {c} has NaN")
    pres = ts.meta["stim_present"].astype(bool)
    if ts.meta.loc[pres, "gabor_contrast"].isna().any():
        out.append("gabor_contrast missing on stimulus trials")
    if not np.isfinite(ts.X).all():
        out.append("non-finite data")
    return out


def run(root: Path, cfg: DecodingConfig, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, staged = root / "runs", root / "sourcedata" / "staged" / "ds006171"
    ph1 = pd.read_csv(root / "results" / "phase1" / "exclusions.tsv", sep="\t",
                      dtype={"subject": str}).fillna("")
    rows, metas = [], []
    ref: tuple[str, ...] | None = None
    for _, r in ph1.sort_values(["subject", "task"]).iterrows():
        sub, task = str(r["subject"]), str(r["task"])
        row = {"subject": sub, "task": task, "loaded": False, "contract": "",
               "phase1_rule": r["subject_rule"] or r["task_rule"], "phase2_rule": "",
               "n_trials": 0, "included": False}
        if not epochs_path(runs, sub, task, cfg.epoch_set).exists():
            row["contract"] = "no stimraw epochs (eegunify epochs stage failed)"
            rows.append(row)
            continue
        try:
            ts = load_trialset(runs, staged, sub, task, cfg)
        except Exception as e:  # recorded, never silently dropped
            row["contract"] = f"load error: {e}"
            rows.append(row)
            continue
        ref = ref or ts.ch_names
        problems = contract_problems(ts, cfg, ref)
        ok, why = labels.enough_trials(ts, cfg.min_trials_per_class)
        row.update(loaded=True, contract="; ".join(problems), phase2_rule=why,
                   n_trials=len(ts))
        row["included"] = not problems and not row["phase1_rule"] and ok
        rows.append(row)
        metas.append(ts.meta.assign(included=row["included"]))
    inc = pd.DataFrame(rows)
    t1 = t1_descriptives(pd.concat(metas, ignore_index=True))
    t1 = t1.merge(inc[["subject", "task", "included"]], on=["subject", "task"])
    out.mkdir(parents=True, exist_ok=True)
    inc.to_csv(out / "inclusion.tsv", sep="\t", index=False)
    t1.to_csv(out / "stats_descriptives.tsv", sep="\t", index=False)
    return inc, t1


def included_tasks(root: Path) -> dict[str, list[str]]:
    """subject -> included tasks, from ``results/phase2/inclusion.tsv``."""
    inc = pd.read_csv(root / "results" / "phase2" / "inclusion.tsv", sep="\t",
                      dtype={"subject": str})
    inc = inc[inc["included"].astype(bool)]
    return {str(s): sorted(map(str, g["task"])) for s, g in inc.groupby("subject")}
