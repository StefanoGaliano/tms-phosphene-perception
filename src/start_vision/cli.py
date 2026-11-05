"""Command-line entry points (sv-inventory, sv-stage, sv-configs, sv-analyze).

Each command is filled in by its build step in plans/PHASE1_BLUEPRINT.md §8.
"""

from __future__ import annotations

import sys


def _not_built(name: str, step: int) -> None:
    print(f"{name}: not built yet (PHASE1_BLUEPRINT §8 step {step})", file=sys.stderr)
    raise SystemExit(2)


def inventory() -> None:
    """Write results/phase1/inventory.tsv from the S3 listing of ds006171."""
    from pathlib import Path

    from start_vision.inventory import list_remote

    df = list_remote()
    out = Path("results/phase1/inventory.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"{len(df)} subjects, {df['bytes'].sum() / 1e9:.1f} GB -> {out}")
    print(df["n_tasks"].value_counts().sort_index().to_string())


def stage() -> None:
    """sv-stage [-s 01 ...] [--task all|nocue|...]: write sourcedata/staged/ds006171/."""
    import argparse
    from pathlib import Path

    import pandas as pd

    from start_vision.codes import TASKS
    from start_vision.stage import stage_recording

    p = argparse.ArgumentParser(prog="sv-stage")
    p.add_argument("-s", "--subject", nargs="*", default=None, help="e.g. 01 03; default all")
    p.add_argument("--task", default="all", choices=["all", *TASKS])
    p.add_argument("--src", default="sourcedata/ds006171")
    p.add_argument("--dst", default="sourcedata/staged/ds006171")
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    subs = a.subject or sorted(d.name[4:] for d in src.glob("sub-*") if d.is_dir())
    tasks = TASKS if a.task == "all" else (a.task,)
    rows = []
    for s in subs:
        for t in tasks:
            bdf = src / f"sub-{s}" / "eeg" / f"sub-{s}_task-{t}_eeg.bdf"
            if not bdf.exists():
                continue
            r = stage_recording(bdf, dst / f"sub-{s}" / "eeg", t, s)
            rows.append({"subject": s, "task": t, "sfreq": r.sfreq, "n_trials": r.n_trials,
                         **r.counts, "no_report": r.excluded.get("no_report", 0),
                         "no_photodiode": r.excluded.get("no_photodiode", 0),
                         "dropped_codes": " ".join(map(str, r.dropped_codes)),
                         "delay_min_ms": r.delay_ms.get("min"),
                         "delay_max_ms": r.delay_ms.get("max"),
                         "n_matched": r.extra["n_matched"],
                         "smd_all": r.balance["smd_all"], "smd_matched": r.balance["smd_matched"]})
            print(f"sub-{s} {t}: {r.counts}", flush=True)
    if not rows:
        raise SystemExit("no recordings found")
    out = Path("results/phase1/stage_summary.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out, sep="\t", dtype={"subject": str})
        done = new.set_index(["subject", "task"]).index
        keep = ~old.set_index(["subject", "task"]).index.isin(done)
        new = pd.concat([old[keep], new]).sort_values(["subject", "task"])
    new.to_csv(out, sep="\t", index=False)
    print(f"{len(rows)} recordings staged -> {dst}; summary {out}")


def configs() -> None:
    _not_built("sv-configs", 5)


def analyze() -> None:
    _not_built("sv-analyze", 9)
