"""Phase 1 tables (PHASE1 §7 analysis/report.py) -> results/phase1/.

Per subject and scope (each task, and ``pooled`` = trial-weighted across the subject's tasks,
D6) the window means of the seen-minus-unseen measures; group stats with ``stats.paired``;
controls; a per-task check that ``stats.paired`` reproduces eegunify's group stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from start_vision.analysis import controls, lateral, load, pool, stats
from start_vision.analysis.sanity import LP_SITES, LP_WIN, VAN_SITES, VAN_WIN
from start_vision.codes import TASKS


def _mean(ev: mne.Evoked, sites: tuple[str, ...], win: tuple[int, int]) -> float:
    sel = (ev.times * 1000 >= win[0]) & (ev.times * 1000 <= win[1])
    return float(ev.copy().pick(list(sites)).data[:, sel].mean() * 1e6)


def subject_measures(evs: dict[str, mne.Evoked]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        comp: str, form: str, seen: mne.Evoked, unseen: mne.Evoked, diff: float, s: float, u: float
    ) -> None:
        rows.append(
            {
                "component": comp,
                "form": form,
                "seen": s,
                "unseen": u,
                "diff": diff,
                "n_seen": seen.nave,
                "n_unseen": unseen.nave,
            }
        )

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


MIN_TRIALS, MAX_REJECTED = 30, 0.45  # PHASE1_REFERENCES §7 rules 2-3
HIT_RANGE, MAX_FA = (0.30, 0.70), 0.20  # rule 1 (user decision 2)
TARGETS = {  # PHASE1_REFERENCES §2 (Eklund & Wiens 2018, Table 2): mean, CI low, CI high
    "van": {"large": (-1.58, -2.37, -0.79), "small": (-1.41, -2.04, -0.77)},
    "lp": {"large": (1.35, 0.96, 1.74), "small": (1.00, 0.50, 1.50)},
}


def _exclusions(runs: Path, staged: Path, tasks: tuple[str, ...]) -> pd.DataFrame:
    """One row per subject x task present in the inventory of staged recordings, with the rule."""
    rows = []
    for task in tasks:
        run = runs / task
        for d in sorted(staged.glob("sub-*")):
            sub = d.name[4:]
            if not (d / "eeg" / f"sub-{sub}_task-{task}_eeg.bdf").exists():
                continue
            b = controls.behaviour(staged, sub, task)
            r: dict[str, Any] = {
                "subject": sub,
                "task": task,
                "n_gabor": b["n_seen"] + b["n_unseen"],
                "n_seen_rep": b["n_seen"],
                "n_catch": b["n_fa"] + b["n_cr"],
                "n_fa": b["n_fa"],
                "rejected_frac": np.nan,
                "n_seen_m": np.nan,
                "n_unseen_m": np.nan,
                "task_rule": "",
            }
            try:
                log = load.epoch_log(run, sub, task)
                r["rejected_frac"] = 1 - float(log["kept"].mean())
            except FileNotFoundError:
                r["task_rule"] = "not processed"
            if sub in (load.subjects(run) if run.exists() else []):
                evs = load.evokeds(run, sub, task)
                r["n_seen_m"], r["n_unseen_m"] = evs["seen_m"].nave, evs["unseen_m"].nave
            elif not r["task_rule"]:
                r["task_rule"] = "3: < 30 matched epochs (eegunify epochs stage failed)"
            if not r["task_rule"] and r["rejected_frac"] > MAX_REJECTED:
                r["task_rule"] = f"2: {r['rejected_frac']:.0%} epochs rejected (> 45 %)"
            if not r["task_rule"] and min(r["n_seen_m"], r["n_unseen_m"]) < MIN_TRIALS:
                r["task_rule"] = "3: < 30 epochs in seen_m or unseen_m"
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = df.groupby("subject")[["n_gabor", "n_seen_rep", "n_catch", "n_fa"]].sum()
    agg["hit_rate"] = agg["n_seen_rep"] / agg["n_gabor"]
    agg["fa_rate"] = agg["n_fa"] / agg["n_catch"]
    bad = ~agg["hit_rate"].between(*HIT_RANGE) | (agg["fa_rate"] > MAX_FA)
    agg["subject_rule"] = np.where(
        bad,
        "1: pooled hit rate "
        + agg["hit_rate"].map("{:.0%}".format)
        + ", FA rate "
        + agg["fa_rate"].map("{:.0%}".format),
        "",
    )
    return df.merge(agg[["hit_rate", "fa_rate", "subject_rule"]], on="subject")


def build(
    runs: Path,
    staged: Path,
    out: Path,
    tasks: tuple[str, ...] = TASKS,
    figures_out: Path | None = None,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    excl = _exclusions(runs, staged, tasks)
    keep = excl[(excl["task_rule"] == "") & (excl["subject_rule"] == "")]
    per: list[dict[str, Any]] = []
    ctl: list[dict[str, Any]] = []
    by_sub: dict[str, dict[str, dict[str, mne.Evoked]]] = {}
    for sub, task in zip(keep["subject"].astype(str), keep["task"].astype(str), strict=True):
        evs = load.evokeds(runs / task, sub, task)
        by_sub.setdefault(sub, {})[task] = evs
        per += [{"subject": sub, "scope": task, **m} for m in subject_measures(evs)]
    for sub, task in zip(excl["subject"].astype(str), excl["task"].astype(str), strict=True):
        ctl.append({"subject": sub, "task": task, **controls.behaviour(staged, sub, task)})
    pooled: dict[str, dict[str, mne.Evoked]] = {}
    for sub, tasks_evs in by_sub.items():
        pooled[sub] = pool.pool_subject(tasks_evs)
        per += [
            {"subject": sub, "scope": "pooled", "tasks": "+".join(sorted(tasks_evs)), **m}
            for m in subject_measures(pooled[sub])
        ]
    per_df = pd.DataFrame(per)
    grp = [
        {"scope": sc, "component": c, "form": f, **stats.paired(g["diff"])}
        for (sc, c, f), g in per_df.groupby(["scope", "component", "form"])
    ]
    grp_df = pd.DataFrame(grp)
    clusters = []
    if len(pooled) >= 2:
        cl = stats.cluster([e["seen_minus_unseen_m"] for e in pooled.values()])
        clusters.append(cl.assign(scope="pooled", contrast="seen_minus_unseen_m"))
    # Per-task clusters computed here too (eegunify 0.9.2 fed the contrast of
    # every epoch set into its cluster test, each subject twice; fixed in 0.9.3), so the
    # numbers do not depend on the eegunify version (results/phase1/summary.md).
    for task in tasks:
        task_waves = [e[task]["seen_minus_unseen_m"] for e in by_sub.values() if task in e]
        if len(task_waves) >= 2:
            clusters.append(
                stats.cluster(task_waves).assign(scope=task, contrast="seen_minus_unseen_m")
            )
    cl_df = pd.concat(clusters, ignore_index=True) if clusters else pd.DataFrame()
    ctl_df = pd.DataFrame(ctl)
    check = reproduce_eegunify(runs, tasks)
    per_df.to_csv(out / "per_subject.tsv", sep="\t", index=False)
    grp_df.to_csv(out / "group.tsv", sep="\t", index=False)
    ctl_df.to_csv(out / "controls.tsv", sep="\t", index=False)
    excl.to_csv(out / "exclusions.tsv", sep="\t", index=False)
    cl_df.to_csv(out / "clusters.tsv", sep="\t", index=False)
    check.to_csv(out / "eegunify_reproduction.tsv", sep="\t", index=False)
    figs: list[Path] = []
    if figures_out is not None and pooled:
        from start_vision.analysis import figures

        figures_out.mkdir(parents=True, exist_ok=True)
        figs = [
            figures.waves(pooled, "pooled", figures_out),
            figures.dots(per_df, grp_df, "pooled", figures_out),
            figures.topomaps(pooled, "pooled", figures_out),
        ]
    return {
        "per_subject": per_df,
        "group": grp_df,
        "controls": ctl_df,
        "exclusions": excl,
        "clusters": cl_df,
        "check": check,
        "figures": figs,
    }


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
            ref = g[
                (g.component == comp)
                & (g.wave == wave)
                & (g.measure == "mean_amplitude")
                & (g.test == "paired_t")
            ]
            if ref.empty:
                continue
            mine = stats.paired(x["value"])
            r = ref.iloc[0]
            rows.append(
                {
                    "task": task,
                    "component": comp,
                    "wave": wave,
                    "n": mine["n"],
                    "t_mine": mine["t"],
                    "t_eegunify": float(r.statistic),
                    "dz_mine": mine["d_z"],
                    "dz_eegunify": float(r.d_z),
                    "abs_err_t": abs(mine["t"] - float(r.statistic)),
                    "abs_err_dz": abs(mine["d_z"] - float(r.d_z)),
                }
            )
    df = pd.DataFrame(rows)
    if len(df):
        df["ok_1e-6"] = (df[["abs_err_t", "abs_err_dz"]].max(axis=1) < 1e-6) | (
            df["t_mine"].isna() & df["t_eegunify"].isna()
        )
    return df
