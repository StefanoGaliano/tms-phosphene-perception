"""Phase 2 stages, standalone (PHASE2 D3; build step 11).

Why D3 and not eegunify stages (D1): Phase 1 runs one eegunify project per task
(``runs/<task>/``, one config each) and the eegunify Runner works inside one project, while
``sv_decode`` pools a subject's three tasks (pooled A1, cross-task A8) and the group stage pools
all subjects. So these stages read eegunify's ``epochs`` outputs and provenance directly and key
themselves with ``eegunify.erp.hashing.stage_key`` (same payload: stage, schema_version, config,
upstream keys, software versions). Layout and provenance follow eegunify:

    derivatives/sv_decode/sub-XX/eeg/sub-XX_desc-<analysis>_scores.npz
    derivatives/sv_decode/sub-XX/eeg/sub-XX_desc-decoding_provenance.json
    derivatives/sv_decode_group/<analysis>_*.tsv|npz, provenance.json

Upstream of ``sv_decode`` for a subject: for each task, the eegunify epochs ``stage_key`` and
the sha256 of the staged events.tsv (the metadata source). Upstream of the group stage: the
subjects' ``sv_decode`` keys. Provenance is written last, so a crash never leaves a valid entry.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eegunify.erp.hashing import file_sha256, stage_key
from start_vision.decoding import labels
from start_vision.decoding.analyses import ANALYSES, SubjectData
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import epochs_path

SUBJECT_STAGE, GROUP_STAGE = "sv_decode", "sv_decode_group"
SCHEMA_VERSION = 1

SUBJECT_ANALYSES = ANALYSES
"""Registry of subject analyses (``start_vision.decoding.analyses.ANALYSES``)."""


@dataclass(frozen=True)
class Project:
    """Where Phase 1 outputs live and where Phase 2 writes."""

    root: Path  # Start_Vision/
    runs: Path  # root/runs
    staged: Path  # root/sourcedata/staged/ds006171

    @classmethod
    def default(cls, root: Path) -> Project:
        return cls(root, root / "runs", root / "sourcedata" / "staged" / "ds006171")

    def sub_dir(self, subject: str) -> Path:
        return self.root / "derivatives" / SUBJECT_STAGE / f"sub-{subject}" / "eeg"

    def group_dir(self) -> Path:
        return self.root / "derivatives" / GROUP_STAGE


def _events(project: Project, subject: str, task: str) -> Path:
    return project.staged / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_events.tsv"


def subject_upstream(project: Project, subject: str, cfg: DecodingConfig) -> dict[str, str]:
    """Upstream keys of the subject's usable tasks. If ``results/phase2/inclusion.tsv`` exists
    (Step 0, ``step0.run``), only its included recordings count."""
    inc_file = project.root / "results" / "phase2" / "inclusion.tsv"
    allowed: set[str] | None = None
    if inc_file.exists():
        from start_vision.decoding.step0 import included_tasks

        allowed = set(included_tasks(project.root).get(subject, []))
    up: dict[str, str] = {}
    for task in cfg.tasks:
        if allowed is not None and task not in allowed:
            continue
        ep = epochs_path(project.runs, subject, task, cfg.epoch_set)
        prov = ep.parent / f"sub-{subject}_task-{task}_desc-epochs_provenance.json"
        if not ep.exists() or not prov.exists():
            continue
        rec = json.loads(prov.read_text())
        if rec.get("status") != "ok":
            continue
        up[f"{task}:epochs"] = str(rec["stage_key"])
        up[f"{task}:events"] = file_sha256(_events(project, subject, task))
    return up


def _hash_config(cfg: DecodingConfig, analyses: Sequence[str]) -> dict[str, Any]:
    p = cfg.payload()
    p.pop("n_jobs")  # parallelism does not change results
    return {"decoding": p, "analyses": sorted(analyses)}


def _cached(prov: Path, key: str, root: Path) -> bool:
    """Cache hit: same key, status ok, and every recorded output still on disk."""
    if not prov.exists():
        return False
    rec = json.loads(prov.read_text())
    return bool(rec.get("stage_key") == key and rec.get("status") == "ok"
                and all((root / o).exists() for o in rec.get("outputs", [])))


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **dict(arrays))  # type: ignore[arg-type]
    tmp.replace(path)


@dataclass(frozen=True)
class Outcome:
    stage: str
    subject: str | None
    status: str  # "ran", "cached", "skipped", "failed"
    key: str
    message: str = ""


def run_subject(project: Project, subject: str, cfg: DecodingConfig,
                analyses: Sequence[str] = tuple(SUBJECT_ANALYSES), force: bool = False
                ) -> Outcome:
    up = subject_upstream(project, subject, cfg)
    if not up:
        return Outcome(SUBJECT_STAGE, subject, "skipped", "", "no ok epochs upstream")
    key = stage_key(stage=SUBJECT_STAGE, schema_version=SCHEMA_VERSION,
                    config=_hash_config(cfg, analyses), upstream=up)
    out = project.sub_dir(subject)
    prov = out / f"sub-{subject}_desc-decoding_provenance.json"
    if not force and _cached(prov, key, project.root):
        return Outcome(SUBJECT_STAGE, subject, "cached", key)
    t0 = time.time()
    tasks = sorted({k.split(":")[0] for k in up})
    d = SubjectData(project.runs, project.staged, subject, tasks, cfg)
    ok, why = labels.enough_trials(d.main, cfg.min_trials_per_class)
    out.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    timing: dict[str, float] = {}
    if ok:
        for name in analyses:
            t1 = time.time()
            res = SUBJECT_ANALYSES[name](d)
            timing[name] = round(time.time() - t1, 1)
            path = out / f"sub-{subject}_desc-{name}_scores.npz"
            _save_npz(path, {**res, "times": d.main.times})
            outputs.append(str(path.relative_to(project.root)))
    ts = d.main
    rec = {"stage": SUBJECT_STAGE, "stage_key": key, "schema_version": SCHEMA_VERSION,
           "status": "ok", "subject": subject, "upstream_keys": up,
           "config": _hash_config(cfg, analyses), "outputs": outputs,
           "excluded": "" if ok else why, "n_trials": len(ts), "tasks": tasks,
           "extra": {"wall_time_s": round(time.time() - t0, 2), "per_analysis_s": timing}}
    _write_json(prov, rec)
    return Outcome(SUBJECT_STAGE, subject, "ran", key, "" if ok else why)


def run_group(project: Project, subjects: Sequence[str], cfg: DecodingConfig,
              force: bool = False) -> Outcome:
    up, included, excluded = {}, [], {}
    for s in subjects:
        prov = project.sub_dir(s) / f"sub-{s}_desc-decoding_provenance.json"
        if not prov.exists():
            excluded[s] = "sv_decode not run"
            continue
        rec = json.loads(prov.read_text())
        if rec.get("excluded"):
            excluded[s] = rec["excluded"]
            continue
        up[f"sub-{s}:{SUBJECT_STAGE}"] = rec["stage_key"]
        included.append(s)
    if len(included) < 2:
        return Outcome(GROUP_STAGE, None, "failed", "", "fewer than 2 subjects")
    key = stage_key(stage=GROUP_STAGE, schema_version=SCHEMA_VERSION,
                    config=_hash_config(cfg, ()), upstream=up)
    gdir = project.group_dir()
    prov = gdir / "provenance.json"
    if not force and _cached(prov, key, project.root):
        return Outcome(GROUP_STAGE, None, "cached", key)
    t0 = time.time()
    gdir.mkdir(parents=True, exist_ok=True)
    from start_vision.decoding import group

    figs = group.run(project, included, cfg)
    outputs = [str(f.relative_to(project.root)) for f in figs]
    _write_json(prov, {"stage": GROUP_STAGE, "stage_key": key, "status": "ok",
                       "schema_version": SCHEMA_VERSION, "upstream_keys": up,
                       "included": included, "excluded": excluded, "outputs": outputs,
                       "config": _hash_config(cfg, ()),
                       "extra": {"wall_time_s": round(time.time() - t0, 2)}})
    return Outcome(GROUP_STAGE, None, "ran", key)


def run(project: Project, subjects: Sequence[str], cfg: DecodingConfig,
        analyses: Sequence[str] = tuple(SUBJECT_ANALYSES), force: bool = False,
        group: bool = True) -> list[Outcome]:
    res = [run_subject(project, s, cfg, analyses, force) for s in subjects]
    if group:
        res.append(run_group(project, subjects, cfg, force))
    return res
