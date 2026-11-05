"""Channel sets (PHASE2 §7 A9, A10, A12; build step 9).

ds006171's 128 channels carry 10-05 names (BDF A1..D32 renamed in configs/template_ds006171.yaml).
Target positions come from MNE's 10-05 template (``colin27_1005``, formerly ``standard_1005``).
Cap positions are the measured ``electrodes.tsv`` aligned to that template by a similarity
Procrustes over the shared names (``cap_positions``), the method of PHASE1_REFERENCES §4 (median
residual 8.7 mm; P3 -> CPP3 10.8 mm), so Phase 1 and Phase 2 map sites the same way.

- ``nearest_subset``: map a target layout (BioSemi 64 / 32, 10-10 names) to cap channels: exact
  name first, then the nearest unused cap channel (global greedy by distance, ties by name), so
  the mapping is unique and deterministic. ``mapping_table`` writes it with distances.
- ``mirror_index`` / ``mirror``: left/right swap by 10-05 name (odd <-> even number, ``z``
  unchanged) for the mirrored-montage A9 variant.
- ``harmonize``: bring TrialSets to the common channels, sfreq and time grid (A12).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from start_vision.decoding.data import TrialSet


def _template() -> mne.channels.DigMontage:
    try:
        return mne.channels.make_standard_montage("colin27_1005")
    except ValueError:  # older MNE
        return mne.channels.make_standard_montage("standard_1005")


def positions(names: Sequence[str]) -> dict[str, np.ndarray]:
    pos = _template().get_positions()["ch_pos"]
    missing = [n for n in names if n not in pos]
    if missing:
        raise ValueError(f"no 10-05 template position for {missing}")
    return {n: np.asarray(pos[n]) for n in names}


def cap_positions(electrodes_tsv: Path) -> tuple[dict[str, np.ndarray], float]:
    """Measured cap positions aligned to the template (m), and the median residual (mm)."""
    df = pd.read_csv(electrodes_tsv, sep="\t")
    df = df[df["type"].astype(str).str.upper() == "EEG"]
    names = list(df["name"])
    src = df[["x", "y", "z"]].to_numpy(float)
    tmpl = positions(names)
    dst = np.array([tmpl[n] for n in names])
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    corr = np.diag([1.0, 1.0, d])
    rot = u @ corr @ vt
    scale = float((sv * np.diag(corr)).sum() / (a**2).sum())
    aligned = scale * a @ rot + mu_d
    resid = float(np.median(np.linalg.norm(aligned - dst, axis=1)) * 1000)
    return {n: aligned[i] for i, n in enumerate(names)}, resid


def target_layout(n: int) -> list[str]:
    if n not in (64, 32):
        raise ValueError("target layouts: 64 or 32 (BioSemi 10-10)")
    return list(mne.channels.make_standard_montage(f"biosemi{n}").ch_names)


def nearest_subset(cap: Sequence[str], targets: Sequence[str],
                   cap_pos: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """One row per target: target, cap channel, distance (mm), exact (bool).

    ``cap_pos``: aligned measured positions (``cap_positions``); template positions otherwise."""
    cap_set = set(cap)
    rows = {t: (t, 0.0, True) for t in targets if t in cap_set}
    used = {c for c, _, _ in rows.values()}
    rest = [t for t in targets if t not in rows]
    if rest:
        tpos = positions(rest)
        cpos = cap_pos if cap_pos is not None else positions(list(cap))
        pairs = sorted(
            (float(np.linalg.norm(tpos[t] - cpos[c]) * 1000), t, c)
            for t in rest for c in cap if c not in used
        )
        for d, t, c in pairs:
            if t in rows or c in used:
                continue
            rows[t] = (c, d, False)
            used.add(c)
    if len(rows) != len(targets):
        raise ValueError("cap has fewer channels than the target layout")
    return pd.DataFrame([{"target": t, "cap": rows[t][0], "distance_mm": round(rows[t][1], 1),
                          "exact": rows[t][2]} for t in targets])


def mapping_table(electrodes_tsv: Path, n: int, out: Path | None = None) -> pd.DataFrame:
    cap_pos, _ = cap_positions(electrodes_tsv)
    df = nearest_subset(list(cap_pos), target_layout(n), cap_pos)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, sep="\t", index=False)
    return df


def subset(ts: TrialSet, table: pd.DataFrame) -> TrialSet:
    """Pick the mapped cap channels and rename them to the target names."""
    picked = ts.pick(list(table["cap"]))
    info = picked.info.copy()
    mne.rename_channels(info, dict(zip(table["cap"], table["target"], strict=True)))
    return TrialSet(picked.X, picked.times, tuple(table["target"]), picked.sfreq, picked.meta,
                    info)


_NUM = re.compile(r"^(?P<stem>[A-Za-z]+?)(?P<num>\d+|z)(?P<h>h?)$")


def mirror_name(name: str) -> str:
    """10-05 left/right counterpart: odd n -> n+1, even n -> n-1, ``z`` unchanged."""
    m = _NUM.match(name)
    if m is None:
        raise ValueError(f"cannot mirror channel name {name!r}")
    num = m["num"]
    if num == "z":
        return name
    k = int(num)
    return f"{m['stem']}{k + 1 if k % 2 else k - 1}{m['h']}"


def mirror_index(ch_names: Sequence[str]) -> np.ndarray:
    """Index such that ``X[:, idx]`` swaps every channel with its mirror."""
    pos = {c: i for i, c in enumerate(ch_names)}
    try:
        return np.array([pos[mirror_name(c)] for c in ch_names])
    except KeyError as e:
        raise ValueError(f"mirror channel {e} not in the set") from e


def mirror(ts: TrialSet) -> TrialSet:
    """Same channel names, data swapped left <-> right (A9 mirrored-montage variant)."""
    idx = mirror_index(ts.ch_names)
    return TrialSet(ts.X[:, idx], ts.times, ts.ch_names, ts.sfreq, ts.meta, ts.info)


def harmonize(sets: Sequence[TrialSet], sfreq: float | None = None) -> list[TrialSet]:
    """Common channels (order of the first set), common sfreq (integer decimation only) and
    the common time grid (intersection)."""
    common = [c for c in sets[0].ch_names if all(c in s.ch_names for s in sets)]
    if not common:
        raise ValueError("no common channels")
    target = sfreq or min(s.sfreq for s in sets)
    out = []
    for s in sets:
        s = s.pick(common)
        factor = s.sfreq / target
        if abs(factor - round(factor)) > 1e-9:
            raise ValueError(f"sfreq {s.sfreq} is not an integer multiple of {target}")
        k = round(factor)
        if k > 1:  # data are low-passed below the new Nyquist upstream (data.prepare)
            s = TrialSet(s.X[..., ::k], s.times[::k], s.ch_names, target, s.meta, s.info)
        out.append(s)
    t0 = max(s.times[0] for s in out)
    t1 = min(s.times[-1] for s in out)
    res = []
    for s in out:
        m = (s.times >= t0 - 1e-9) & (s.times <= t1 + 1e-9)
        res.append(TrialSet(s.X[..., m], s.times[m], s.ch_names, s.sfreq, s.meta, s.info))
    if len({len(s.times) for s in res}) != 1 or not all(
        np.allclose(r.times, res[0].times) for r in res
    ):
        raise ValueError("time grids do not align after harmonization")
    return res
