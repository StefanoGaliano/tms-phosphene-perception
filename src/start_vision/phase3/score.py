"""Phase 3 step 10: frozen detector on each cleaned recording (blueprint D4, D5).

One ``Detector.from_epochs`` call per recording (the per-recording normalization). Output
``results/phase3/scores.tsv``: recording, subject, site, pipeline, pulse, code, seen, p_seen.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from start_vision.decoding.detector import Detector
from start_vision.phase3.config import Phase3Config

MOTOR = ["FC5", "FC3", "FC1", "C5", "C3", "C1", "CP5", "CP3", "CP1",
         "FC6", "FC4", "FC2", "C6", "C4", "C2", "CP6", "CP4", "CP2"]
"""R1: the 18 fronto-central/central/centro-parietal channels of the go/no-go."""


def detector_input(ep: mne.BaseEpochs, det: Detector) -> mne.BaseEpochs:
    c = det.contract
    e = ep.copy().pick(c["channels"])
    return e.crop(c["epoch_window_s"][0], c["epoch_window_s"][1])


def without_motor(ep: mne.BaseEpochs) -> mne.BaseEpochs:
    """R1: the motor channels replaced by spherical-spline interpolation from the others."""
    e = ep.copy()
    e.info["bads"] = [c for c in MOTOR if c in e.ch_names]
    e.interpolate_bads(reset_bads=True, verbose="error")
    return e


def score_epochs(ep: mne.BaseEpochs, det: Detector,
                 transform: Callable[[mne.BaseEpochs], mne.BaseEpochs] | None = None
                 ) -> np.ndarray:
    e = detector_input(ep, det)
    if transform is not None:
        e = transform(e)
    return det.from_epochs(e)


POSTERIOR = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"]
ALPHA_BAND = (8.0, 13.0)        # Phase 2 config alpha_band_hz (fixed band, no individual peak)
EARLY_CUT_S = 0.30              # R4a, user decision 11 / Q6


def early_lesion_scores(ep: mne.BaseEpochs, det: Detector) -> np.ndarray:
    """R4a: frozen model with feature bins starting at or after 0.30 s set to the training mean
    (0 after the scaler); no refit. Calibrated with the detector's own calibrator."""
    c = det.contract
    e = detector_input(ep, det)
    x = (e.get_data() * 1e6).astype(np.float32)
    lo, hi = c["epoch_window_s"]
    t = e.times
    m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    from start_vision.decoding.detector import prepare_subject

    xp = prepare_subject(x[..., m], t[m], tuple(c["baseline_window_s"]),
                         tuple(c["normalization"]["window_s"]))
    wf, sc, lm = (det.pipeline.steps[i][1] for i in range(3))
    f = sc.transform(wf.transform(xp))
    edges = np.arange(c["feature_window_s"][0], c["feature_window_s"][1] - 1e-9,
                      c["feature_bin_s"])
    nb = len(edges)
    late = np.tile(edges >= EARLY_CUT_S - 1e-9, f.shape[1] // nb)
    f[:, late] = 0.0
    s = lm.decision_function(f)
    return np.asarray(det.calibrator.predict_proba(np.asarray(s)[:, None])[:, 1])


def trial_features(ep: mne.BaseEpochs) -> pd.DataFrame:
    """Per-trial control features (label-free): C3-C4 (R2, R5), EOG (E1), prestim alpha (A1)."""
    from mne.time_frequency import psd_array_multitaper

    t = ep.times
    w = (t >= 0.15) & (t <= 0.6)
    out: dict[str, np.ndarray] = {}
    ch = ep.ch_names
    if "C3_ctl" in ch and "C4_ctl" in ch:
        x = ep.get_data(picks=["C3_ctl", "C4_ctl"]) * 1e6
        out["c3c4"] = (x[:, 0] - x[:, 1])[:, w].mean(1)
    for e in ("HEOG", "VEOG"):
        if f"{e}_ctl" in ch:
            x = ep.get_data(picks=[f"{e}_ctl"])[:, 0] * 1e6
            pre = x[:, (t >= -0.2) & (t < -0.01)].mean(1)
            out[e.lower()] = x[:, w].mean(1) - pre
    pre = (t >= -0.5) & (t <= -0.1)
    seg = ep.get_data(picks=[c for c in POSTERIOR if c in ch])[:, :, pre].astype(np.float64)
    psd, _ = psd_array_multitaper(seg, ep.info["sfreq"], fmin=ALPHA_BAND[0],
                                  fmax=ALPHA_BAND[1], verbose="error")
    out["alpha_log10"] = np.log10(psd.mean(-1)).mean(-1)
    return pd.DataFrame(out)


def parse_rid(path: Path) -> tuple[str, str]:
    parts = path.name.split("_")
    return parts[0].removeprefix("sub-"), parts[1].removeprefix("site-")


def score_all(cfg: Phase3Config, root: Path, pipeline: str = "primary") -> pd.DataFrame:
    det = Detector.load(root / cfg.detector)
    rows = []
    for f in sorted((root / cfg.out_dir / pipeline).glob("*_desc-detector_epo.fif")):
        ep = mne.read_epochs(f, verbose="error")
        sub, site = parse_rid(f)
        p = score_epochs(ep, det)
        p_nm = score_epochs(ep, det, without_motor)
        p_early = early_lesion_scores(ep, det)
        m = ep.metadata.reset_index(drop=True)
        rows.append(pd.concat([pd.DataFrame({
            "recording": f"sub-{sub}_site-{site}", "subject": sub, "site": site,
            "pipeline": pipeline, "pulse": m["pulse"], "code": m["code"], "seen": m["seen"],
            "p_seen": p, "p_seen_no_motor": p_nm, "p_seen_early": p_early}),
            trial_features(ep)], axis=1))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
