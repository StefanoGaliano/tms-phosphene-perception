"""Q41 secondaries not covered by scripts/q41_ips.py (QUESTIONS_PREREG §12, amendment 5),
on the IPS primary pipeline, included recordings only (results/ips/inclusion_primary.tsv):

  (d) Q38  H1 with P(seen) residualized on the hand index (mirrored-pair left - right,
           start_vision.questions.q38.hand_index); also all lateral channels interpolated.
  (d) Q37a shared pulse response removed: mean_sub and ssp3 (as scripts/questions_q37ab_phase3.py).
  (e) R4b  the early-only (0.15-0.30 s) detector already fitted for Phase 3
           (results/phase3/s7_r4b/r4b_detector), applied unchanged; no refit.
  (g) lesion of the parietal channels around the coil, interpolated from the others.

All uncorrected secondaries; one-sided sign flip on subject AUC (seed 20261109).
Run from Start_Vision/: ~/miniconda3/envs/eegunify/bin/python -I scripts/q41_secondary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from questions_q37ab_phase3 import variant  # noqa: E402

from start_vision.decoding.detector import Detector  # noqa: E402
from start_vision.phase3.analyses import (  # noqa: E402
    group_one_sided, rec_auc, residualized, subject_values)
from start_vision.phase3.score import detector_input  # noqa: E402
from start_vision.questions.q38 import hand_index, mirror_pairs  # noqa: E402

RES = ROOT / "results" / "ips"
OUT = RES / "secondary"
EPO = ROOT / "derivatives" / "sv_ips_clean" / "primary"
SEED = 20261109
LESION = ["P1", "P3", "P5", "CP1", "CP3", "CP5", "PO3", "PO7",
          "P2", "P4", "P6", "CP2", "CP4", "CP6", "PO4", "PO8"]   # amendment 5 (g)


def interp(e: mne.BaseEpochs, chs: list[str]) -> mne.BaseEpochs:
    e = e.copy()
    e.info["bads"] = [c for c in chs if c in e.ch_names]
    e.interpolate_bads(reset_bads=True, verbose="error")
    return e


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    det = Detector.load(ROOT / "detectors" / "sv-perception-tms-robust-v1.0.0")
    r4b = Detector.load(ROOT / "results" / "phase3" / "s7_r4b" / "r4b_detector")
    lateral = sorted({c for p in mirror_pairs(list(det.contract["channels"])) for c in p})
    inc = pd.read_csv(RES / "inclusion_primary.tsv", sep="\t", dtype={"subject": str})
    inc = inc[inc["included"]]
    rows = []
    for rid, sub, site in inc[["recording", "subject", "site"]].itertuples(index=False):
        ep = mne.read_epochs(EPO / f"{rid}_desc-detector_epo.fif", verbose="error")
        e = detector_input(ep, det)
        g = pd.DataFrame({"seen": ep.metadata["seen"].to_numpy().astype(int),
                          "p_seen": det.from_epochs(e)})
        g["hand"] = hand_index(e.get_data() * 1e6, list(e.ch_names), e.times)
        g["p_res"] = residualized(g, "p_seen", "hand")
        g["p_wide"] = det.from_epochs(interp(e, lateral))
        g["p_mean_sub"] = det.from_epochs(variant(e, "mean_sub"))
        g["p_ssp3"] = det.from_epochs(variant(e, "ssp3"))
        g["p_lesion"] = det.from_epochs(interp(e, LESION))
        g["p_r4b"] = r4b.from_epochs(detector_input(ep, r4b))
        r = {"recording": rid, "subject": sub, "site": site}
        for c in ("p_seen", "p_res", "hand", "p_wide", "p_mean_sub", "p_ssp3", "p_lesion", "p_r4b"):
            r["auc_" + c.removeprefix("p_")] = rec_auc(g, c)
        rows.append(r)
        print(rid, {k: round(v, 3) for k, v in r.items() if k.startswith("auc")}, flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_recording.tsv", sep="\t", index=False, float_format="%.5f")
    names = {"auc_seen": "H1 frozen (check = q41 primary)",
             "auc_res": "Q38 residualized on hand index",
             "auc_wide": f"Q38 (a) all {len(lateral)} lateral channels interpolated",
             "auc_hand": "Q38 (b) hand index alone",
             "auc_mean_sub": "Q37a mean_sub", "auc_ssp3": "Q37a ssp3",
             "auc_r4b": "R4b early-only detector (0.15-0.30 s)",
             "auc_lesion": "(g) parietal coil channels interpolated"}
    res = {v: group_one_sided(subject_values(per, k), seed=SEED) for k, v in names.items()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    t = pd.DataFrame([{"test": k, **v} for k, v in res.items()])
    t.to_csv(OUT / "group.tsv", sep="\t", index=False, float_format="%.4f")
    print(t[["test", "n", "mean", "p", "p_two_sided", "n_above"]].to_string(index=False))


if __name__ == "__main__":
    main()
