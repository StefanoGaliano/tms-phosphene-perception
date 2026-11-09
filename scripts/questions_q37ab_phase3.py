"""Questions_to_solve Q37a and Q37b on the Phase 3 primary pipeline (exploratory, not pre-registered).

Run from Start_Vision/: ~/miniconda3/envs/eegunify/bin/python -I scripts/questions_q37ab_phase3.py
Inputs: derivatives/sv_p3_clean/primary/*_desc-detector_epo.fif, results/phase3/scores_primary.tsv,
results/phase3/inclusion_primary.tsv. Output: results/questions_to_solve/q37ab_phase3/.
Same recordings as H1 (included = True). Settings fixed here before any result was computed.

Q37a, shared pulse response. Label-free, per recording, on the detector input (64 contract channels,
contract window), then the frozen detector unchanged (Detector.from_epochs):
  mean_sub : subtract the recording's mean over all kept trials (the common TMS-evoked response)
  ssp3     : project out the top 3 spatial components (SVD over channels) of that mean in 0..0.6 s
H1 rerun per variant (per-recording AUC, subject = mean of sites, one-sided sign flip).
Note: removing the mean of all trials leaves each class's deviation from it, so the class
difference survives; what is removed is the response both classes share.

Q37b, pulses needed. From the primary P(seen): per recording, 1000 mini-blocks per class of k pulses
drawn without replacement within the class, block score = mean P(seen); AUC(block class) for
k = 1, 5, 10, 20 (k limited by the smaller class). Subject = mean over sites; group mean, bootstrap CI.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from start_vision.decoding.detector import Detector
from start_vision.phase3 import analyses as an
from start_vision.phase3.score import detector_input

ROOT = Path.cwd()
OUT = ROOT / "results" / "questions_to_solve" / "q37ab_phase3"
DET = ROOT / "detectors" / "sv-perception-tms-robust-v1.0.0"
SEED = 20261109
N_SSP = 3
SSP_WINDOW_S = (0.0, 0.6)
KS = (1, 5, 10, 20)
N_BLOCKS = 1000


def variant(e: mne.BaseEpochs, how: str) -> mne.BaseEpochs:
    x = e.get_data()
    mean = x.mean(0, keepdims=True)
    if how == "mean_sub":
        y = x - mean
    elif how == "ssp3":
        tm = (e.times >= SSP_WINDOW_S[0]) & (e.times <= SSP_WINDOW_S[1])
        u, _, _ = np.linalg.svd(mean[0][:, tm], full_matrices=False)
        p = np.eye(x.shape[1]) - u[:, :N_SSP] @ u[:, :N_SSP].T
        y = np.einsum("ij,njt->nit", p, x)
    else:
        raise ValueError(how)
    return mne.EpochsArray(y, e.info, tmin=e.tmin, verbose="error")


def q37a(inc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    det = Detector.load(DET)
    rows = []
    for rid in inc.loc[inc.included, "recording"]:
        f = ROOT / "derivatives" / "sv_p3_clean" / "primary" / f"{rid}_desc-detector_epo.fif"
        ep = mne.read_epochs(f, verbose="error")
        y = ep.metadata["seen"].to_numpy()
        e = detector_input(ep, det)
        r = {"recording": rid, "subject": rid.split("_")[0][4:], "site": rid.split("-")[-1]}
        r["auc_frozen"] = roc_auc_score(y, det.from_epochs(e))
        for how in ("mean_sub", "ssp3"):
            r[f"auc_{how}"] = roc_auc_score(y, det.from_epochs(variant(e, how)))
        rows.append(r)
        print(rid, {k: round(v, 3) for k, v in r.items() if k.startswith("auc")}, flush=True)
    per = pd.DataFrame(rows)
    grp = [{"variant": c.removeprefix("auc_"), **an.group_one_sided(an.subject_values(per, c))}
           for c in ("auc_frozen", "auc_mean_sub", "auc_ssp3")]
    return per, pd.DataFrame(grp)


def q37b(inc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    s = pd.read_csv(ROOT / "results" / "phase3" / "scores_primary.tsv", sep="\t", dtype={"subject": str})
    s = s[s.recording.isin(inc.loc[inc.included, "recording"])]
    rows = []
    for rid, g in s.groupby("recording"):
        p1 = g.loc[g.seen == 1, "p_seen"].to_numpy()
        p0 = g.loc[g.seen == 0, "p_seen"].to_numpy()
        r = {"recording": rid, "subject": g.subject.iloc[0], "n1": len(p1), "n0": len(p0)}
        for k in KS:
            if k > min(len(p1), len(p0)):
                r[f"auc_k{k}"] = np.nan
                continue
            b1 = np.array([rng.choice(p1, k, replace=False).mean() for _ in range(N_BLOCKS)])
            b0 = np.array([rng.choice(p0, k, replace=False).mean() for _ in range(N_BLOCKS)])
            lab = np.r_[np.ones(N_BLOCKS), np.zeros(N_BLOCKS)]
            r[f"auc_k{k}"] = roc_auc_score(lab, np.r_[b1, b0])
        rows.append(r)
    per = pd.DataFrame(rows)
    grp = []
    for k in KS:
        v = an.subject_values(per, f"auc_k{k}").dropna().to_numpy()
        boot = rng.choice(v, (10_000, len(v))).mean(1)
        grp.append({"k_pulses": k, "n_subjects": len(v), "mean_auc": v.mean(),
                    "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                    "min_subject": v.min(), "max_subject": v.max(),
                    "n_subjects_auc_ge_0.75": int((v >= 0.75).sum())})
    return per, pd.DataFrame(grp)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inc = pd.read_csv(ROOT / "results" / "phase3" / "inclusion_primary.tsv", sep="\t")
    per_b, grp_b = q37b(inc)
    per_b.to_csv(OUT / "q37b_per_recording.tsv", sep="\t", index=False, float_format="%.4f")
    grp_b.to_csv(OUT / "q37b_group.tsv", sep="\t", index=False, float_format="%.4f")
    print(grp_b.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    per_a, grp_a = q37a(inc)
    per_a.to_csv(OUT / "q37a_per_recording.tsv", sep="\t", index=False, float_format="%.4f")
    grp_a.to_csv(OUT / "q37a_group.tsv", sep="\t", index=False, float_format="%.4f")
    print(grp_a.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
