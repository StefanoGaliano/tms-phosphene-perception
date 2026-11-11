"""EXPLORATORY (not pre-registered), 2026-11-11, after reading Pigorini et al. 2026 and Knight et al. 2024.

A) Per-subject fixed threshold (Pigorini-style): condition-average P(seen) for phosphene and
   no-phosphene pulses per subject (mean over included sites), classified with the frozen
   detector threshold theta (0.521, from natural vision). Also the relative rule
   (phosphene average > none average).
B) Channel-group virtual lesions (R1 logic): H1 with each group interpolated from the others.
   Groups fixed before running: medial occipital (stimulated site), lateral occipito-temporal,
   frontal (size-matched control).
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from start_vision.decoding.detector import Detector
from start_vision.phase3.analyses import group_one_sided, per_recording, subject_values
from start_vision.phase3.score import score_epochs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "questions_to_solve" / "explore_lesion_threshold"
DET = Detector.load(ROOT / "detectors" / "sv-perception-tms-robust-v1.0.0")
THETA = 0.521148143854642
GROUPS = {"medial_occipital": ["O1", "Oz", "O2", "POz", "PO3", "PO4"],
          "lateral_occipitotemporal": ["P7", "P8", "PO7", "PO8", "TP7", "TP8", "T7", "T8"],
          "frontal_control": ["Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "F7", "F8"]}


def lesion(chs: list[str]):
    def f(e: mne.BaseEpochs) -> mne.BaseEpochs:
        e = e.copy()
        e.info["bads"] = [c for c in chs if c in e.ch_names]
        e.interpolate_bads(reset_bads=True, verbose="error")
        return e
    return f


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    sc = pd.read_csv(ROOT / "results" / "phase3" / "scores_primary.tsv", sep="\t", dtype={"subject": str})
    inc = pd.read_csv(ROOT / "results" / "phase3" / "inclusion_primary.tsv", sep="\t", dtype={"subject": str})
    inc = inc[inc["included"]]
    sc = sc[sc["recording"].isin(inc["recording"])]
    rec = sc.groupby(["subject", "recording", "seen"])["p_seen"].mean().unstack()
    sub = rec.groupby(level="subject").mean()
    a = pd.DataFrame({"phos": sub[1], "none": sub[0]})
    a["fixed_theta_both_correct"] = (a["phos"] >= THETA) & (a["none"] < THETA)
    a["relative_correct"] = a["phos"] > a["none"]
    a.to_csv(OUT / "A_per_subject_threshold.tsv", sep="\t", float_format="%.4f")
    res["A"] = {"theta": THETA, "n_subjects": len(a),
                "fixed_theta_both_correct": int(a["fixed_theta_both_correct"].sum()),
                "phos_avg_above_theta": int((a["phos"] >= THETA).sum()),
                "none_avg_below_theta": int((a["none"] < THETA).sum()),
                "relative_correct": int(a["relative_correct"].sum()),
                "mean_phos": float(a["phos"].mean()), "mean_none": float(a["none"].mean())}
    print(res["A"], flush=True)

    for name, chs in GROUPS.items():
        rows = []
        for rid, s, site in inc[["recording", "subject", "site"]].itertuples(index=False):
            ep = mne.read_epochs(ROOT / "derivatives" / "sv_p3_clean" / "primary"
                                 / f"{rid}_desc-detector_epo.fif", verbose="error")
            rows.append(pd.DataFrame({"recording": rid, "subject": s, "site": site,
                                      "seen": ep.metadata["seen"].to_numpy().astype(int),
                                      "p_seen": score_epochs(ep, DET, transform=lesion(chs))}))
        pr = per_recording(pd.concat(rows, ignore_index=True), n_perm=200)
        sv = subject_values(pr, "auc")
        res[name] = group_one_sided(sv) | {"channels": chs, "subjects_above_chance": int((sv > 0.5).sum())}
        print(name, res[name], flush=True)
        (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
