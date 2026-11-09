"""Q37a trial-level TMS-evoked-response amplitude (P§6)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from start_vision.decoding.detector import Detector
from start_vision.phase3.analyses import group_one_sided, rec_auc, residualized, subject_values
from start_vision.phase3.score import detector_input
from start_vision.questions import figures as fg
from start_vision.questions import io
from start_vision.questions.params import SEED, TEMPLATE_WINDOW_S
from start_vision.questions.report import group_row, write_readme
from start_vision.questions.stats import one_sample

DETECTOR = Path("detectors") / "sv-perception-tms-robust-v1.0.0"


def tep_amplitude(x: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Least-squares projection of each trial on the label-free mean template, 0-0.6 s (P§6)."""
    m = (times >= TEMPLATE_WINDOW_S[0] - 1e-9) & (times <= TEMPLATE_WINDOW_S[1] + 1e-9)
    xs = x[..., m].reshape(len(x), -1)
    tpl = xs.mean(0)
    return np.asarray(xs @ tpl / (tpl @ tpl))


def recording_values(g: pd.DataFrame) -> dict[str, float]:
    gg = g.assign(p_res=residualized(g, "p_seen", "amp"))
    return {"auc_frozen": rec_auc(g, "p_seen"), "auc_res": rec_auc(gg, "p_res"),
            "auc_amp": rec_auc(g, "amp"),
            "amp_mean_phosphene": float(g.loc[g["seen"] == 1, "amp"].mean()),
            "amp_mean_none": float(g.loc[g["seen"] == 0, "amp"].mean())}


def run(root: Path, force: bool = False) -> Path:
    d = io.out_dir(root, "q37a", force)
    det = Detector.load(root / DETECTOR)
    scores = io.gin_scores(root)
    rows, trials = [], []
    recs = sorted(scores["recording"].unique())
    for rid, ep in io.gin_epochs_iter(root, recs):
        e = detector_input(ep, det)
        amp = tep_amplitude(e.get_data() * 1e6, e.times)
        g = scores[scores["recording"] == rid].reset_index(drop=True)
        if len(g) != len(amp) or not np.array_equal(g["seen"].to_numpy(),
                                                    ep.metadata["seen"].to_numpy()):
            raise ValueError(f"{rid}: scores and epochs disagree (pulse order)")
        g = g.assign(amp=amp)
        r: dict[str, Any] = {"recording": rid, "subject": g["subject"].iloc[0],
                             "site": g["site"].iloc[0], **recording_values(g)}
        rows.append(r)
        trials.append(g[["recording", "pulse", "seen", "amp"]])
    per = pd.DataFrame(rows)
    per.to_csv(d / "per_recording.tsv", sep="\t", index=False, float_format="%.5f")
    pd.concat(trials).to_csv(d / "trial_amplitude.tsv.gz", sep="\t", index=False,
                             float_format="%.5f")
    prim = group_one_sided(subject_values(per, "auc_res"), seed=SEED)
    frozen = group_one_sided(subject_values(per, "auc_frozen"), seed=SEED)
    amp_alone = one_sample(subject_values(per, "auc_amp"), 0.5, tail=2)
    ssp = pd.read_csv(root / io.QROOT / "q37ab_phase3" / "q37a_group.tsv", sep="\t")
    ssp_row = ssp[ssp["variant"] == "ssp3"].iloc[0]
    tab = pd.DataFrame([
        {"test": "H1 with P(seen) residualized on TEP amplitude > 0.5", "role": "[P]",
         "n": prim["n"], "mean": prim["mean"], "ci_low": prim["ci_low"],
         "ci_high": prim["ci_high"], "d_z": prim["d_z"], "p": prim["p"], "tail": 1.0,
         "n_positive": prim["n_above"]},
        {"test": "H1 frozen (reproduces Phase 3)", "role": "check", "n": frozen["n"],
         "mean": frozen["mean"], "p": frozen["p"], "tail": 1.0},
        group_row("TEP amplitude alone, AUC vs 0.5", "secondary", amp_alone),
        {"test": "SSP-3 (first look, copied as context)", "role": "context", "n": ssp_row["n"],
         "mean": ssp_row["mean"], "p": ssp_row["p"], "tail": 1.0}])
    tab.to_csv(d / "group.tsv", sep="\t", index=False, float_format="%.5f")
    io.write_primary(d, "q37a", "H1 residualized on TEP amplitude", "one-sided",
                     {"n": float(prim["n"]), "mean": float(prim["mean"]),
                      "ci_low": float(prim["ci_low"]), "ci_high": float(prim["ci_high"]),
                      "p": float(prim["p"])})
    fig, (a, b) = fg.axes(2)
    sf, sr = subject_values(per, "auc_frozen"), subject_values(per, "auc_res")
    a.scatter(sf, sr, color=fg.SLOTS[0], s=20, label=f"subjects (N {len(sf)})")
    a.plot([0.44, 0.66], [0.44, 0.66], color=fg.GRID, lw=1)
    fg.style(a, "subject AUC, frozen", "subject AUC, residualized on amplitude")
    a.legend(fontsize=7, frameon=False)
    b.scatter(per["amp_mean_none"], per["amp_mean_phosphene"], color=fg.SLOTS[1], s=20,
              label=f"recordings (N {len(per)})")
    lo, hi = per[["amp_mean_none", "amp_mean_phosphene"]].min().min(), per[
        ["amp_mean_none", "amp_mean_phosphene"]].max().max()
    b.plot([lo, hi], [lo, hi], color=fg.GRID, lw=1)
    fg.style(b, "mean amplitude, no phosphene", "mean amplitude, phosphene")
    b.legend(fontsize=7, frameon=False)
    fg.save(fig, d)
    if prim["p"] < 0.05:
        txt = "H1 survives: the effect is not the trial-to-trial size of the pulse response."
    else:
        txt = "H1 does not survive residualizing on the pulse-response amplitude."
    if amp_alone["p"] < 0.05:
        txt += (" TEP amplitude alone separates the classes: the phosphene modulates the evoked "
                "response (Taylor 2010-like), so residualizing removes part of a real effect.")
    write_readme(d, root, "Q37a trial-level TMS-evoked-response amplitude (GIN)", "§6",
                 ["derivatives/sv_p3_clean/primary (33 included recordings)",
                  "results/phase3/scores_primary.tsv"], {"Group tests": tab},
                 txt + " (P§6 rule)",
                 ["Same recordings as H1 (P§0 first look: mean subtraction, SSP-3)."])
    return d
