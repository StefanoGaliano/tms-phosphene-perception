"""Pool one subject's per-task evokeds (D6): conditions by trial count, contrasts recomputed.

Contrasts are recomputed from the pooled conditions, never averaged across tasks. Tasks
without a condition (noninformative has no left/right or lateralized waves, decision 6)
simply do not contribute to it.
"""

from __future__ import annotations

from collections.abc import Mapping

import mne

CONDITIONS = ("seen", "unseen", "seen_m", "unseen_m", "seen_left", "seen_right",
              "unseen_left", "unseen_right", "fa", "cr", "seen_contra", "seen_ipsi",
              "unseen_contra", "unseen_ipsi")
CONTRASTS = {"seen_minus_unseen": ("seen", "unseen"),
             "seen_minus_unseen_m": ("seen_m", "unseen_m"),
             "fa_minus_cr": ("fa", "cr"),
             "seen_lat": ("seen_contra", "seen_ipsi"),
             "unseen_lat": ("unseen_contra", "unseen_ipsi")}


def pool_subject(by_task: Mapping[str, Mapping[str, mne.Evoked]]) -> dict[str, mne.Evoked]:
    out: dict[str, mne.Evoked] = {}
    for c in CONDITIONS:
        evs = [e[c] for e in by_task.values() if c in e and e[c].nave > 0]
        if not evs:
            continue
        common = [ch for ch in evs[0].ch_names if all(ch in e.ch_names for e in evs)]
        evs = [e.copy().pick(common) for e in evs]
        p = evs[0] if len(evs) == 1 else mne.combine_evoked(evs, weights="nave")
        p.comment = c
        out[c] = p
    for name, (a, b) in CONTRASTS.items():
        if a in out and b in out:
            d = mne.combine_evoked([out[a], out[b]], weights=[1, -1])
            d.comment = name
            out[name] = d
    return out
