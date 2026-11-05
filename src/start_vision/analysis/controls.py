"""Behavioural and design controls (PHASE1 §7 analysis/controls.py), from the staged events.

(a) false alarms: FA rate per subject x task (FA-minus-CR waves only with >= MIN_FA trials);
(b) contrast balance before/after matching (from stage) and after epoch rejection (sanity);
(c) orientation accuracy seen vs unseen; (d) hit rate per task (staircase target near 50 %).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from start_vision import codes as C  # noqa: N812

MIN_FA = 10


def behaviour(staged: Path, sub: str, task: str) -> dict[str, float]:
    f = staged / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{task}_events.tsv"
    ev = pd.read_csv(f, sep="\t")
    ev = ev[ev["value"] < C.RESPONSE_OFFSET]
    cls = ev["value"].astype(int).map(lambda v: C.decode(v).cls)
    n = {k: int((cls == k).sum()) for k in (C.SEEN, C.UNSEEN, C.FA, C.CR, C.EXCLUDED)}
    oc = ev["objective_correct"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    def acc(k: int) -> float:
        x = oc[(cls == k) & oc.notna()]
        return float(x.mean()) if len(x) else float("nan")

    def n_or(k: int) -> int:
        return int(((cls == k) & oc.notna()).sum())

    gab, catch = n[C.SEEN] + n[C.UNSEEN], n[C.FA] + n[C.CR]
    return {
        "n_seen": n[C.SEEN], "n_unseen": n[C.UNSEEN], "n_fa": n[C.FA], "n_cr": n[C.CR],
        "n_excluded": n[C.EXCLUDED],
        "hit_rate": n[C.SEEN] / gab if gab else np.nan,
        "fa_rate": n[C.FA] / catch if catch else np.nan,
        "fa_wave_ok": float(n[C.FA] >= MIN_FA),
        "orient_acc_seen": acc(C.SEEN), "orient_n_seen": n_or(C.SEEN),
        "orient_acc_unseen": acc(C.UNSEEN), "orient_n_unseen": n_or(C.UNSEEN),
    }
