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
    """sv-configs: write configs/generated/ds006171_<task>.yaml from the template."""
    from pathlib import Path

    from start_vision.configs import write_all

    for p in write_all(Path("configs/template_ds006171.yaml"),
                       Path("results/phase1/inventory.tsv"), Path("configs/generated")):
        print(p)


def analyze() -> None:
    """sv-analyze: results/phase1/{per_subject,group,controls,eegunify_reproduction}.tsv."""
    from pathlib import Path

    from start_vision.analysis.report import build

    r = build(Path("runs"), Path("sourcedata/staged/ds006171"), Path("results/phase1"),
              figures_out=Path("results/phase1/figures"))
    print(r["group"].to_string(index=False))
    if len(r["check"]):
        print(f"eegunify reproduction to 1e-6: {int(r['check']['ok_1e-6'].sum())}"
              f"/{len(r['check'])} rows")


def decode() -> None:
    """sv-decode: Phase 2 stages (sv_decode per subject, then sv_decode_group), cached."""
    import argparse
    from pathlib import Path

    from start_vision.decoding import config, stages

    p = argparse.ArgumentParser(prog="sv-decode", description=decode.__doc__)
    p.add_argument("-s", "--subject", nargs="*", default=None, help="e.g. 01 03; default all")
    p.add_argument("--config", default="configs/decoding.yaml")
    p.add_argument("--analyses", nargs="*", default=list(stages.SUBJECT_ANALYSES),
                   choices=list(stages.SUBJECT_ANALYSES))
    p.add_argument("--force", action="store_true", help="rerun even on a cache hit")
    p.add_argument("--no-group", action="store_true")
    a = p.parse_args()
    cfg = config.load(Path(a.config))
    proj = stages.Project.default(Path.cwd())
    subs = a.subject or sorted(
        d.name[4:] for d in proj.staged.iterdir() if d.name.startswith("sub-"))
    for o in stages.run(proj, subs, cfg, a.analyses, a.force, group=not a.no_group):
        sub = f"sub-{o.subject}" if o.subject else "group"
        print(f"{o.stage:16s} {sub:8s} {o.status:8s} {o.message}")
