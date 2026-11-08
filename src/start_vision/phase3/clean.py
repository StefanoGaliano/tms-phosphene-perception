"""Phase 3 step 8: one GIN recording -> detector-ready epochs (blueprint §7, C0-C12).

Label-blind: the chain gets pulse samples only; S100/S200 are attached at the very end as
metadata. Training steps call eegunify v0.9.3 as a library (``erp.preprocess.preprocess``,
``erp.ica.run_ica``, ``erp.epochs`` rejection helpers) with the training sections of
``configs/template_ds006171.yaml``; nothing in eegunify is edited.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mne
import numpy as np
import pandas as pd
import yaml

from start_vision import tms
from start_vision.phase3.config import Phase3Config
from start_vision.phosphenes import complete_contract_channels


@dataclass
class Recording:
    raw: mne.io.BaseRaw
    pulses: np.ndarray          # sample indices
    codes: list[str]            # S100/S200 per pulse (kept away from the chain)


def read_recording(set_path: Path, cfg: Phase3Config) -> Recording:
    """C0: read, set channel types, keep pulse samples and codes."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="error")
    types = {c: "eog" for c in cfg.eog_bipolar["VEOG"] + cfg.eog_bipolar["HEOG"]
             if c in raw.ch_names}
    raw.set_channel_types(types, verbose="error")
    raw.drop_channels([c for c in cfg.drop_channels if c in raw.ch_names])
    ev, eid = mne.events_from_annotations(raw, verbose="error")
    keep = {c: eid.get(c) for c in (cfg.codes.present, cfg.codes.absent)}
    other = set(eid) - set(keep) - {"boundary"}
    if other:
        raise ValueError(f"{set_path.name}: unexpected event codes {sorted(other)}")
    inv = {v: k for k, v in keep.items() if v is not None}
    sel = np.isin(ev[:, 2], list(inv))
    raw.set_annotations(None)
    return Recording(raw, ev[sel, 0] - raw.first_samp, [inv[c] for c in ev[sel, 2]])


def set_positions(raw: mne.io.BaseRaw) -> None:
    """C0: canonical names (FP1 -> Fp1) and 10-05 template positions for the EEG channels."""
    from start_vision.decoding.channels import positions

    raw.rename_channels({c: c.capitalize() for c in raw.ch_names if c in ("FP1", "FP2")})
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=positions(eeg), coord_frame="head"),
                    on_missing="ignore", verbose="error")


def _training_sections(cfg: Phase3Config, root: Path) -> Any:
    from eegunify.config.erp_sections import ArtifactsSection, PreprocSection
    from eegunify.config.schema import ReferenceSection

    tpl = yaml.safe_load((root / cfg.eegunify_template).read_text(encoding="utf-8"))
    art = dict(tpl["artifacts"])
    art["eog_bipolar"] = {k: list(v) for k, v in cfg.eog_bipolar.items()}
    ref = {k: v for k, v in tpl["reference"].items() if k in ("scheme", "channels")}
    return SimpleNamespace(preproc=PreprocSection.model_validate(tpl["preproc"]),
                           artifacts=ArtifactsSection.model_validate(art),
                           reference=ReferenceSection.model_validate(ref))


def reject(ep: mne.BaseEpochs, cfg: Phase3Config) -> tuple[np.ndarray, dict[str, int]]:
    """§6: ptp on EEG in the detector's windows + HEOG step (training thresholds)."""
    from eegunify.erp.epochs import moving_window_ptp, step_like

    r = cfg.rejection
    sf = ep.info["sfreq"]
    t = ep.times
    x = ep.get_data(picks="eeg") * 1e6
    bad_ptp = np.zeros(len(ep), bool)
    for a, b in r.ptp_windows_s:
        m = (t >= a) & (t <= b)
        ptp = moving_window_ptp(x[:, :, m], int(r.ptp_win_s * sf), int(r.ptp_step_s * sf))
        bad_ptp |= (ptp > r.ptp_uv).any(axis=1)
    bad_heog = np.zeros(len(ep), bool)
    if "HEOG" in ep.ch_names:
        h = ep.get_data(picks=["HEOG"]) * 1e6
        m = (t >= r.heog_window_s[0]) & (t <= r.heog_window_s[1])
        st = step_like(h[:, :, m], int(r.heog_win_s * sf), max(1, int(r.heog_step_s * sf)))
        bad_heog = (st > r.heog_step_uv).any(axis=1)
    keep = ~(bad_ptp | bad_heog)
    return keep, {"ptp": int(bad_ptp.sum()), "heog_step": int(bad_heog.sum()),
                  "kept": int(keep.sum()), "total": len(ep)}


def clean(rec: Recording, cfg: Phase3Config, root: Path, pipeline: str = "primary",
          qa_cb: Any = None) -> tuple[mne.Epochs, dict[str, Any]]:
    """C1-C12 on one recording. Returns detector-ready epochs (64 contract channels + VEOG/HEOG
    + C3/C4 copies as misc, 128 Hz, -1..1 s, kept trials only, label metadata) and a QA dict."""
    import json

    from eegunify.erp.ica import run_ica
    from eegunify.erp.preprocess import preprocess

    pl = cfg.pipelines[pipeline]
    qa: dict[str, Any] = {"pipeline": pipeline, "timings_s": {}}
    tic = time.perf_counter()
    raw, pulses = rec.raw.copy(), rec.pulses.copy()
    set_positions(raw)
    if qa_cb:
        qa_cb("raw", raw, pulses)
    p = cfg.pulse
    # C1-C2
    end = tms.artifact_end_ms(raw, pulses, p.artifact_k, p.artifact_hold_ms)
    cut = tms.cut_window_ms(p.cut_ms, end, p.extend_cut, p.max_cut_ms)
    tms.interpolate_pulses(raw, pulses, cut)
    qa |= {"artifact_end_ms": end, "cut_ms": list(cut)}
    if qa_cb:
        qa_cb("cut", raw, pulses)
    # C3
    raw, pulses = tms.resample_with_pulses(raw, pulses, cfg.ica_sfreq)
    qa["timings_s"]["c1_c3"] = time.perf_counter() - tic
    # C4-C5
    if cfg.decay_ica.enabled:
        d = cfg.decay_ica
        res = tms.decay_ica(raw, pulses, cut, d.epoch_s, d.fit_highpass_hz, d.window_ms,
                            d.ratio_threshold, d.method, d.seed)
        qa["decay_ica"] = {"excluded": res.excluded, "n_components": res.n_components,
                           "ratios_top5": sorted(res.ratios)[-5:]}
        qa["artifact_end_after_ica_ms"] = tms.artifact_end_ms(
            raw, pulses, p.artifact_k, p.artifact_hold_ms)
    qa["timings_s"]["c4_c5"] = time.perf_counter() - tic
    if qa_cb:
        qa_cb("decay_ica", raw, pulses)
    # C6-C8 via eegunify (training sections). Pulses as annotations so ICA fits skip them.
    sec = _training_sections(cfg, root)
    if pl.muscle_ica:
        icl = sec.artifacts.ica.iclabel
        icl = icl.model_copy(update={"exclude_classes": sorted(
            set(icl.exclude_classes) | {"muscle artifact"})})
        sec.artifacts = sec.artifacts.model_copy(
            update={"ica": sec.artifacts.ica.model_copy(update={"iclabel": icl})})
    sf = raw.info["sfreq"]
    # "stimulus/pulse" marks the trial blocks for eegunify's break exclusion (training default
    # fit_exclude: {} -> break_selector "stimulus/*"); BAD_pulse keeps the cut out of ICA fits.
    on = pulses / sf
    raw.set_annotations(mne.Annotations(
        np.r_[on, on + cut[0] / 1000 - 0.001],
        np.r_[np.zeros(len(on)), np.full(len(on), (cut[1] - cut[0]) / 1000 + 0.002)],
        ["stimulus/pulse"] * len(on) + ["BAD_pulse"] * len(on)))
    pre = preprocess(raw, cast(Any, sec), reference_free=False)
    if pre.failed:
        raise RuntimeError(f"eegunify preprocess QA failed: {pre.qa}")
    raw = pre.raw
    qa["bad_channels"] = pre.bads
    qa["interpolated"] = pre.interpolated
    ica = run_ica(raw, sec.artifacts, sec.reference)
    raw = ica.raw
    qa["ocular_ica"] = json.loads(json.dumps(ica.steps, default=str))
    qa["timings_s"]["c6_c8"] = time.perf_counter() - tic
    if qa_cb:
        qa_cb("ocular_ica", raw, (pulses * raw.info["sfreq"] / sf).astype(int))
    # C9
    sf2 = raw.info["sfreq"]
    pl2 = np.round(pulses * sf2 / sf).astype(int)
    ev = np.c_[pl2 + raw.first_samp, np.zeros_like(pl2), np.arange(1, len(pl2) + 1)]
    ep = mne.Epochs(raw, ev, tmin=cfg.epochs.window_s[0], tmax=cfg.epochs.window_s[1],
                    baseline=None, preload=True, reject_by_annotation=False, verbose="error",
                    event_repeated="drop")
    idx = ep.events[:, 2] - 1
    keep, counts = reject(ep, cfg)
    qa["rejection"] = counts
    ep = ep[np.where(keep)[0]]
    idx = idx[keep]
    # C10: low-pass + decimate exactly as decoding.data.prepare
    ep.filter(None, cfg.lowpass_hz, method="iir", iir_params={"order": 4, "ftype": "butter"},
              phase="zero", picks=["eeg", "eog"], verbose="error")
    factor = round(ep.info["sfreq"] / 128.0)
    ep.decimate(factor, verbose="error")
    # C11: average reference over recorded channels (preprocess did it), then the 5 channels
    contract = _contract_channels(root / cfg.detector)
    misc = ep.copy().pick(["C3", "C4"] + [c for c in ("VEOG", "HEOG") if c in ep.ch_names])
    misc.rename_channels({c: f"{c}_ctl" for c in misc.ch_names})
    misc.set_channel_types({c: "misc" for c in misc.ch_names}, verbose="error")
    full, missing = complete_contract_channels(ep, contract)
    full.add_channels([misc], force_update_info=True)
    qa["interpolated_contract"] = missing
    full.metadata = pd.DataFrame({"pulse": idx, "code": [rec.codes[i] for i in idx],
                                  "seen": [int(rec.codes[i] == cfg.codes.present) for i in idx]})
    qa["timings_s"]["total"] = time.perf_counter() - tic
    return full, qa


def _contract_channels(det: Path) -> list[str]:
    import json

    return list(json.loads((det / "contract.json").read_text())["channels"])
