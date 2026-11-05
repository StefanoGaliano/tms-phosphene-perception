"""Phase 1 tables (PHASE1 §7 analysis/report.py) -> results/phase1/.

Per subject and scope (each task, and ``pooled`` = trial-weighted across the subject's tasks,
D6) the window means of the seen-minus-unseen measures; group stats with ``stats.paired``;
controls; a per-task check that ``stats.paired`` reproduces eegunify's group stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import pandas as pd

from start_vision.analysis import controls, lateral, load, pool, stats
from start_vision.analysis.sanity import LP_SITES, LP_WIN, VAN_SITES, VAN_WIN
from start_vision.codes import TASKS


def _mean(ev: mne.Evoked, sites: tuple[str, ...], win: tuple[int, int]) -> float:
    sel = (ev.times * 1000 >= win[0]) & (ev.times * 1000 <= win[1])
    return float(ev.copy().pick(list(sites)).data[:, sel].mean() * 1e6)


def subject_measures(evs: dict[str, mne.Evoked]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(comp: str, form: str, seen: mne.Evoked, unseen: mne.Evoked, diff: float,
            s: float, u: float) -> None:
        rows.append({"component": comp, "form": form, "seen": s, "unseen": u, "diff": diff,
                     "n_seen": seen.nave, "n_unseen": unseen.nave})

    if "seen_m" in evs and "unseen_m" in evs:
        s, u = evs["seen_m"], evs["unseen_m"]
        for comp, sites, win in (("van", VAN_SITES, VAN_WIN), ("lp", LP_SITES, LP_WIN)):
            a, b = _mean(s, sites, win), _mean(u, sites, win)
            add(comp, "bilateral_matched", s, u, a - b, a, b)
    if "seen_contra" in evs and "unseen_contra" in evs:
        s, u = evs["seen_contra"], evs["unseen_contra"]
        a, b = lateral.window_mean(s), lateral.window_mean(u)
        add("van", "contralateral", s, u, a - b, a, b)  # primary (decision 4)
    if "seen_lat" in evs and "unseen_lat" in evs:
        s, u = evs["seen_lat"], evs["unseen_lat"]
        a, b = lateral.window_mean(s), lateral.window_mean(u)
        add("van", "double_difference", s, u, a - b, a, b)
    return rows


def build(runs: Path, staged: Path, out: Path, tasks: tuple[str, ...] = TASKS) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    per: list[dict[str, Any]] = []
    ctl: list[dict[str, Any]] = []
    by_sub: dict[str, dict[str, dict[str, mne.Evoked]]] = {}
    for task in tasks:
        run = runs / task
        for sub in load.subjects(run) if run.exists() else []:
            evs = load.evokeds(run, sub, task)
            by_sub.setdefault(sub, {})[task] = evs
            per += [{"subject": sub, "scope": task, **r} for r in subject_measures(evs)]
            ctl.append({"subject": sub, "task": task, **controls.behaviour(staged, sub, task)})
    for sub, tasks_evs in by_sub.items():
        pooled = pool.pool_subject(tasks_evs)
        per += [{"subject": sub, "scope": "pooled", "tasks": "+".join(sorted(tasks_evs)), **r}
                for r in subject_measures(pooled)]
    per_df = pd.DataFrame(per)
    grp = []
    for (scope, comp, form), g in per_df.groupby(["scope", "component", "form"]):
        grp.append({"scope": scope, "component": comp, "form": form, **stats.paired(g["diff"])})
    grp_df = pd.DataFrame(grp)
    ctl_df = pd.DataFrame(ctl)
    check = reproduce_eegunify(runs, tasks)
    per_df.to_csv(out / "per_subject.tsv", sep="\t", index=False)
    grp_df.to_csv(out / "group.tsv", sep="\t", index=False)
    ctl_df.to_csv(out / "controls.tsv", sep="\t", index=False)
    check.to_csv(out / "eegunify_reproduction.tsv", sep="\t", index=False)
    return {"per_subject": per_df, "group": grp_df, "controls": ctl_df, "check": check}


def reproduce_eegunify(runs: Path, tasks: tuple[str, ...]) -> pd.DataFrame:
    """stats.paired on eegunify's own per-subject measures vs its group stats.tsv."""
    rows = []
    for task in tasks:
        run = runs / task
        try:
            g = load.group_stats(run, task)
        except FileNotFoundError:
            continue
        m = load.measures(run, task)
        m = m[(m.roi == "primary") & (m.measure == "mean_amplitude")]
        for (comp, wave), x in m.groupby(["component", "wave"]):
            ref = g[(g.component == comp) & (g.wave == wave) & (g.measure == "mean_amplitude")
                    & (g.test == "paired_t")]
            if ref.empty:
                continue
            mine = stats.paired(x["value"])
            r = ref.iloc[0]
            rows.append({"task": task, "component": comp, "wave": wave, "n": mine["n"],
                         "t_mine": mine["t"], "t_eegunify": float(r.statistic),
                         "dz_mine": mine["d_z"], "dz_eegunify": float(r.d_z),
                         "abs_err_t": abs(mine["t"] - float(r.statistic)),
                         "abs_err_dz": abs(mine["d_z"] - float(r.d_z))})
    df = pd.DataFrame(rows)
    if len(df):
        df["ok_1e-6"] = (df[["abs_err_t", "abs_err_dz"]].max(axis=1) < 1e-6) | (
            df["t_mine"].isna() & df["t_eegunify"].isna())
    return df
