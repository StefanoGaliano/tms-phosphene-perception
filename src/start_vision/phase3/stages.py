"""Phase 3 step 9: cached cleaning stage (blueprint D1), one recording at a time.

Output per recording and pipeline in ``derivatives/sv_p3_clean/<pipeline>/``:
``sub-NN_site-SS_desc-detector_epo.fif`` (detector-ready epochs + label metadata),
``..._desc-qa.json`` (cleaning QA + stage key), ``..._desc-qaerp.npz`` (pulse-locked averages
after each step, label-blind, for figure F1). A rerun with the same key is a no-op.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mne
import numpy as np

from start_vision.phase3 import clean
from start_vision.phase3.config import Phase3Config

SCHEMA_VERSION = 1


def rec_id(sub: str, site: str) -> str:
    return f"sub-{sub}_site-{site}"


def set_path(cfg: Phase3Config, root: Path, sub: str, site: str) -> Path:
    hits = sorted((root / cfg.data_dir / site).glob(f"{sub}_*_{site}_SICL.set"))
    if not hits:
        raise FileNotFoundError(f"no .set for sub-{sub} site-{site}")
    return hits[0]


def key(cfg: Phase3Config, pipeline: str, set_file: Path) -> str:
    from eegunify.erp.hashing import stage_key

    st = set_file.stat()
    return stage_key(stage="sv_p3_clean", schema_version=SCHEMA_VERSION,
                     config={"cfg": cfg.model_dump(mode="json"), "pipeline": pipeline},
                     upstream={set_file.name: f"{st.st_size}:{int(st.st_mtime)}"})


def out_paths(cfg: Phase3Config, root: Path, pipeline: str, rid: str) -> dict[str, Path]:
    d = root / cfg.out_dir / pipeline
    return {"epo": d / f"{rid}_desc-detector_epo.fif", "qa": d / f"{rid}_desc-qa.json",
            "erp": d / f"{rid}_desc-qaerp.npz"}


class _QaErp:
    """Pulse-locked average (all pulses, label-blind) after each chain step."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def __call__(self, name: str, raw: mne.io.BaseRaw, pulses: np.ndarray) -> None:
        sf = raw.info["sfreq"]
        x = raw.get_data(picks="eeg")
        a, b = int(-0.1 * sf), int(0.6 * sf)
        ok = [p for p in pulses if p + a >= 0 and p + b < x.shape[1]]
        avg = np.mean([x[:, p + a:p + b] for p in ok], axis=0)
        avg -= avg[:, : int(0.098 * sf)].mean(axis=1, keepdims=True)
        self.store[f"{name}_avg"] = avg.astype(np.float32)
        self.store[f"{name}_times"] = np.arange(a, b) / sf
        self.store[f"{name}_ch"] = np.array([raw.ch_names[i]
                                             for i in mne.pick_types(raw.info, eeg=True)])


def run(cfg: Phase3Config, root: Path, sub: str, site: str, pipeline: str = "primary",
        force: bool = False) -> tuple[Path, bool]:
    """Clean one recording; returns (epochs path, ran?)."""
    sp = set_path(cfg, root, sub, site)
    rid = rec_id(sub, site)
    out = out_paths(cfg, root, pipeline, rid)
    k = key(cfg, pipeline, sp)
    if not force and out["qa"].exists() and out["epo"].exists():
        if json.loads(out["qa"].read_text(encoding="utf-8")).get("stage_key") == k:
            return out["epo"], False
    out["epo"].parent.mkdir(parents=True, exist_ok=True)
    rec = clean.read_recording(sp, cfg)
    cb = _QaErp()
    ep, qa = clean.clean(rec, cfg, root, pipeline, qa_cb=cb)
    qa |= {"stage_key": k, "recording": rid, "set_file": sp.name,
           "n_pulses": len(rec.pulses), "n_kept": len(ep)}
    tmp = out["epo"].with_name(out["epo"].stem + ".tmp-epo.fif")
    ep.save(tmp, overwrite=True, verbose="error")
    os.replace(tmp, out["epo"])
    np.savez_compressed(out["erp"], **cb.store)
    out["qa"].write_text(json.dumps(qa, indent=1, default=str), encoding="utf-8")
    return out["epo"], True
