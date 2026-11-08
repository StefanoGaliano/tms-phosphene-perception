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


def step0() -> None:
    """sv-step0: Phase 2 Step 0 (contract, inclusion, descriptives) -> results/phase2/."""
    from pathlib import Path

    from start_vision.decoding import config
    from start_vision.decoding import step0 as s0

    inc, _ = s0.run(Path.cwd(), config.load(Path("configs/decoding.yaml")),
                    Path("results/phase2"))
    print(f"{int(inc.included.sum())}/{len(inc)} recordings included, "
          f"{inc[inc.included].subject.nunique()} subjects")


def export() -> None:
    """sv-export: A11, choose and export the frozen detectors into detectors/ (immutable)."""
    import argparse
    from pathlib import Path

    from start_vision.decoding import config, stages
    from start_vision.decoding import export as ex
    from start_vision.decoding.step0 import included_tasks

    p = argparse.ArgumentParser(prog="sv-export", description=export.__doc__)
    p.add_argument("--variant", nargs="*", default=["tms-robust", "full"],
                   choices=["tms-robust", "full"])
    p.add_argument("--config", default="configs/decoding.yaml")
    p.add_argument("--backend", default="sklearn", choices=["sklearn", "torch"],
                   help="selection fits: sklearn (CPU threads) or torch (GPU); the exported "
                        "detector is fitted by sklearn either way")
    a = p.parse_args()
    proj = stages.Project.default(Path.cwd())
    subs = sorted(included_tasks(proj.root))
    for f in ex.run(proj, subs, config.load(Path(a.config)), a.variant, a.backend):
        print(f"exported {f}")


def eklund() -> None:
    """sv-eklund: frozen tms-robust detector on Eklund & Wiens 2018 (A12 issue #1)."""
    from pathlib import Path

    from start_vision.decoding import config
    from start_vision.decoding import eklund_test as et

    root = Path.cwd()
    et.run(root, config.load(root / "configs" / "decoding.yaml"),
           root / "results" / "eklund2018")


def p3_inventory() -> None:
    """sv-p3-inventory: GIN Phosphenes_occipital files vs annex keys (Phase 3 step 1)."""
    import argparse
    import shutil
    from pathlib import Path

    from start_vision.phase3 import inventory as inv

    p = argparse.ArgumentParser(prog="sv-p3-inventory")
    p.add_argument("--hash", action="store_true", help="also check MD5 of every local file (slow)")
    a = p.parse_args()
    root = Path.cwd()
    src = root / "sourcedata"
    df, note = inv.run(src / "phosphenes_occipital_index", src / "phosphenes_occipital",
                       root / "results" / "phase3" / "inventory.tsv", a.hash)
    print(df[["site", "subject", "status"]].to_string(index=False))
    print(note)
    need = df.loc[df["status"] == "missing", ["set_bytes_key", "fdt_bytes_key"]].fillna(0)
    print(f"to download: {(df['status'] == 'missing').sum()} recordings, "
          f"{need.to_numpy().sum() / 1e9:.1f} GB; free: {shutil.disk_usage(src).free / 1e9:.0f} GB")


def p3_clean() -> None:
    """sv-p3-clean: TMS cleaning + detector-matched chain, cached (Phase 3 step 9)."""
    import argparse
    from pathlib import Path

    from start_vision.phase3 import config, stages

    p = argparse.ArgumentParser(prog="sv-p3-clean")
    p.add_argument("-s", "--subject", action="append", required=True, help="e.g. 03 (repeatable)")
    p.add_argument("--site", action="append", choices=["O1", "O2"], default=None)
    p.add_argument("--pipeline", default="primary", choices=["primary", "paperlike"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    root = Path.cwd()
    cfg = config.load(root / "configs" / "phase3.yaml")
    for sub in a.subject:
        for site in a.site or ["O1", "O2"]:
            path, ran = stages.run(cfg, root, sub, site, a.pipeline, a.force)
            print(f"{'cleaned' if ran else 'cached '} {path}")


def p3_score() -> None:
    """sv-p3-score: frozen detector on every cleaned recording (Phase 3 step 10)."""
    import argparse
    from pathlib import Path

    from start_vision.phase3 import config, score

    p = argparse.ArgumentParser(prog="sv-p3-score")
    p.add_argument("--pipeline", default="primary", choices=["primary", "paperlike"])
    a = p.parse_args()
    root = Path.cwd()
    cfg = config.load(root / "configs" / "phase3.yaml")
    df = score.score_all(cfg, root, a.pipeline)
    out = root / "results" / "phase3" / f"scores_{a.pipeline}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"{df['recording'].nunique()} recordings, {len(df)} trials -> {out}")


def p3_analyze() -> None:
    """sv-p3-analyze: inclusion, primary/secondary tests and controls (Phase 3 step 11)."""
    import argparse
    from pathlib import Path

    import pandas as pd

    from start_vision.phase3 import analyses as an
    from start_vision.phase3 import config, stages

    p = argparse.ArgumentParser(prog="sv-p3-analyze")
    p.add_argument("--pipeline", default="primary", choices=["primary", "paperlike"])
    a = p.parse_args()
    root = Path.cwd()
    cfg = config.load(root / "configs" / "phase3.yaml")
    res = root / "results" / "phase3"
    scores = pd.read_csv(res / f"scores_{a.pipeline}.tsv", sep="\t", dtype={"subject": str})
    cpath = res / "code_check.tsv"
    codes = (pd.read_csv(cpath, sep="\t") if cpath.exists()
             else pd.DataFrame(columns=["recording", "code_diff_uv", "code_p"]))
    for rid in sorted(set(scores["recording"]) - set(codes["recording"])):
        sub, site = rid.split("_")[0][4:], rid.split("_")[1][5:]
        r = an.code_check(stages.set_path(cfg, root, sub, site))
        codes = pd.concat([codes, pd.DataFrame([{"recording": rid, **r}])], ignore_index=True)
        print(f"code check {rid}: {r}")
    codes.to_csv(cpath, sep="\t", index=False)
    inc = an.inclusion(scores, codes)
    inc.to_csv(res / f"inclusion_{a.pipeline}.tsv", sep="\t", index=False)
    keep = set(inc.loc[inc["included"], "recording"])
    per = an.per_recording(scores[scores["recording"].isin(keep)])
    per.to_csv(res / f"per_recording_{a.pipeline}.tsv", sep="\t", index=False)
    t2, t3 = an.tests(per)
    t2.to_csv(res / f"t2_tests_{a.pipeline}.tsv", sep="\t", index=False)
    t3.to_csv(res / f"t3_controls_{a.pipeline}.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(inc[["recording", "kept", "seen_share", "included", "reason"]].to_string(index=False))
        print(t2.to_string(index=False))
        print(t3.to_string(index=False))
