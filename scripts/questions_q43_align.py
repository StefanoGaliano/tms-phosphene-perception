"""Q43 (docs/QUESTIONS_PREREG.md §14, tag questions-prereg-amend7): label-free domain alignment (37h).

Q42 models M0, M1, M3; each unit (training subject, TMS recording) aligned on its own trials without
labels: M0/M1 centred + Ledoit-Wolf whitened; M3 re-centred (minus unit mean tangent vector).
[P] subject AUC(aligned M0) - AUC(M0) > 0, one-sided.
Usage: python scripts/questions_q43_align.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_s7_r4b as s7mod  # noqa: E402
import questions_q42_latency as q42  # noqa: E402

from start_vision.decoding import across, channels, config  # noqa: E402
from start_vision.decoding.detector import Detector  # noqa: E402

ROOT = s7mod.ROOT
OUT = ROOT / "results" / "questions_to_solve" / "prereg_q43"


def align(m: str, F: np.ndarray) -> np.ndarray:
    Z = F - F.mean(0, keepdims=True)
    if m == "M3":
        return Z
    ev, V = np.linalg.eigh(LedoitWolf().fit(Z).covariance_)
    return Z @ (V / np.sqrt(ev)) @ V.T


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load(ROOT / "configs" / "decoding.yaml")
    sets, _, _ = s7mod.training_sets(cfg)
    det = Detector.load(s7mod.FROZEN)
    res: dict = {"prereg": "docs/QUESTIONS_PREREG.md §14, tag questions-prereg-amend7",
                 "n_train_subjects": len(sets)}
    for pipe in ("primary", "paperlike"):
        inc = s7mod.included(pipe)
        test = [s7mod.tms_trialset(s7mod.read_rec(pipe, r), det, r) for r in inc["recording"]]
        h = channels.harmonize(sets + test)
        tr = [across.normalize(s, cfg) for s in h[:len(sets)]]
        te = [across.normalize(s, cfg) for s in h[len(sets):]]
        rec_sub = dict(zip(inc["recording"], inc["subject"], strict=True))
        subs = sorted(set(rec_sub.values()))
        ys = [s.labels("seen") for s in tr]
        out: dict = {}
        sub_auc: dict[str, np.ndarray] = {}
        for m in ("M0", "M1", "M3"):
            Ftr = [q42.feats(m, s.X, s.times, False) for s in tr]
            Fte = [q42.feats(m, s.X, s.times, True) for s in te]
            for tag, A, B in (("raw", Ftr, Fte),
                              ("aligned", [align(m, f) for f in Ftr], [align(m, f) for f in Fte])):
                est = q42.clf(cfg).fit(np.concatenate(A), np.concatenate(ys))
                rec = {str(s.meta["subject"].iloc[0]):
                       roc_auc_score(s.labels("seen"), est.decision_function(f))
                       for s, f in zip(te, B, strict=True)}
                k = f"{m}_{tag}"
                sub_auc[k] = np.array([np.mean([rec[r] for r in rec if rec_sub[r] == su])
                                       for su in subs])
                out[f"{k}_vs_chance"] = q42.summ(sub_auc[k] - 0.5, 1) | {
                    "auc_mean": float(sub_auc[k].mean())}
                if pipe == "primary" and tag == "aligned":
                    loso = []
                    for i in range(len(A)):
                        tri = [j for j in range(len(A)) if j != i]
                        e = q42.clf(cfg).fit(np.concatenate([A[j] for j in tri]),
                                             np.concatenate([ys[j] for j in tri]))
                        loso.append(roc_auc_score(ys[i], e.decision_function(A[i])))
                    out[f"{k}_natural_loso_auc"] = float(np.mean(loso))
                print(pipe, k, json.dumps(out[f"{k}_vs_chance"]), flush=True)
        out["P_M0_aligned_minus_raw"] = q42.summ(sub_auc["M0_aligned"] - sub_auc["M0_raw"], 1)
        out["a_M1_aligned_minus_raw"] = q42.summ(sub_auc["M1_aligned"] - sub_auc["M1_raw"], 2)
        out["a_M3_aligned_minus_raw"] = q42.summ(sub_auc["M3_aligned"] - sub_auc["M3_raw"], 2)
        out["subject_auc"] = {k: v.tolist() for k, v in sub_auc.items()}
        res[pipe] = out
        (OUT / "q43.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
        print(pipe, "P", json.dumps(out["P_M0_aligned_minus_raw"]), flush=True)


if __name__ == "__main__":
    main()
