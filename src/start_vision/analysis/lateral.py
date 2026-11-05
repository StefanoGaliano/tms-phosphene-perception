"""Contralateral VAN (primary, decision 4) and the double difference (secondary, D8)."""

from __future__ import annotations

import mne

from start_vision.analysis.sanity import VAN_WIN

PAIR = ("O1", "O2")  # both pair members carry the pair's value in eegunify's lateralized waves


def window_mean(ev: mne.Evoked, ch: str = PAIR[0], win: tuple[int, int] = VAN_WIN) -> float:
    sel = (ev.times * 1000 >= win[0]) & (ev.times * 1000 <= win[1])
    return float(ev.copy().pick([ch]).data[0, sel].mean() * 1e6)


def contra_difference(evs: dict[str, mne.Evoked]) -> mne.Evoked:
    """contra(seen) - contra(unseen): the primary VAN wave (Eklund's measure)."""
    d = mne.combine_evoked([evs["seen_contra"], evs["unseen_contra"]], weights=[1, -1])
    d.comment = "seen_minus_unseen_contra"
    return d


def double_difference(evs: dict[str, mne.Evoked]) -> mne.Evoked:
    """(contra - ipsi)(seen) - (contra - ipsi)(unseen)."""
    d = mne.combine_evoked([evs["seen_lat"], evs["unseen_lat"]], weights=[1, -1])
    d.comment = "seen_lat_minus_unseen_lat"
    return d
