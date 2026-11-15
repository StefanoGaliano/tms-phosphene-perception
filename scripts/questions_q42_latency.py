"""Q42 (docs/QUESTIONS_PREREG.md §13, tag questions-prereg-amend6): latency-tolerant features (37e).

M0 0.15-0.60 s 50 ms bins; M1 same window one bin; M2 M0 applied to TMS at +0.20 s; M3 log-Euclidean
covariance. Trained on all included ds006171 subjects (C = 0.01, frozen detector's C), tested on GIN.
[P] subject AUC(M1) - AUC(M0) > 0, one-sided.
Usage: python scripts/questions_q42_latency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_s7_r4b as s7mod  # noqa: E402

from start_vision.decoding import across, channels, config, models, stats  # noqa: E402
from start_vision.decoding.detector import Detector  # noqa: E402
from start_vision.decoding.features import window_means  # noqa: E402

ROOT = s7mod.ROOT
OUT = ROOT / "results" / "questions_to_solve" / "prereg_q42"
WIN, SHIFT, C, SEED = (0.15, 0.60), 0.20, 0.01, 20261109


def cov_tangent(X: np.ndarray, times: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    m = (times >= win[0] - 1e-9) & (times <= win[1] + 1e-9)
    seg = X[..., m].astype(np.float64)
    seg = seg - seg.mean(-1, keepdims=True)
    n_ch = seg.shape[1]
    iu = np.triu_indices(n_ch)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    out = np.empty((len(seg), len(iu[0])))
    for i, s in enumerate(seg):
        c = s @ s.T / s.shape[1]
        c = 0.9 * c + 0.1 * np.trace(c) / n_ch * np.eye(n_ch)
        ev, V = np.linalg.eigh(c)
        out[i] = ((V * np.log(ev)) @ V.T)[iu] * w
    return out


def feats(model: str, X: np.ndarray, times: np.ndarray, test: bool) -> np.ndarray:
    if model == "M0":
        return window_means(X, times, WIN, 0.05)
    if model == "M1":
        return window_means(X, times, WIN, WIN[1] - WIN[0])
    if model == "M2":
        w = (WIN[0] + SHIFT, WIN[1] + SHIFT) if test else WIN
        return window_means(X, times, w, 0.05)
    return cov_tangent(X, times, WIN)


def clf(cfg: config.DecodingConfig):  # type: ignore[no-untyped-def]
    return make_pipeline(StandardScaler(), models.logistic(cfg, C))


def signflip(x: np.ndarray, tail: int, n: int = 10_000) -> float:
    rng = np.random.default_rng(SEED)
    null = (rng.choice([-1, 1], size=(n, len(x))) * x).mean(1)
    obs = x.mean()
    if tail == 1:
        return float((np.sum(null >= obs) + 1) / (n + 1))
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def summ(x: np.ndarray, tail: int) -> dict:
    lo, hi = stats.bootstrap_ci(x, seed=SEED)
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)], "p": signflip(x, tail),
            "tail": "one" if tail == 1 else "two", "d_z": float(x.mean() / x.std(ddof=1)),
            "n_positive": int((x > 0).sum()), "n": len(x)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load(ROOT / "configs" / "decoding.yaml")
    sets, _, _ = s7mod.training_sets(cfg)
    det = Detector.load(s7mod.FROZEN)
    res: dict = {"prereg": "docs/QUESTIONS_PREREG.md §13, tag questions-prereg-amend6",
                 "n_train_subjects": len(sets), "C": C, "window_s": list(WIN), "shift_s": SHIFT}
    for pipe in ("primary", "paperlike"):
        inc = s7mod.included(pipe)
        test = [s7mod.tms_trialset(s7mod.read_rec(pipe, r), det, r) for r in inc["recording"]]
        h = channels.harmonize(sets + test)
        tr = [across.normalize(s, cfg) for s in h[:len(sets)]]
        te = [across.normalize(s, cfg) for s in h[len(sets):]]
        rec_sub = dict(zip(inc["recording"], inc["subject"], strict=True))
        out: dict = {}
        sub_auc: dict[str, np.ndarray] = {}
        ys = [s.labels("seen") for s in tr]
        for m in ("M0", "M1", "M2", "M3"):
            F = [feats(m, s.X, s.times, False) for s in tr]  # cached once per subject
            Xtr = np.concatenate(F)
            ytr = np.concatenate(ys)
            est = clf(cfg).fit(Xtr, ytr)
            rec = {}
            for s in te:
                rid = str(s.meta["subject"].iloc[0])
                rec[rid] = roc_auc_score(s.labels("seen"),
                                         est.decision_function(feats(m, s.X, s.times, True)))
            subs = sorted(set(rec_sub.values()))
            sub_auc[m] = np.array([np.mean([rec[r] for r in rec if rec_sub[r] == su])
                                   for su in subs])
            out[f"{m}_vs_chance"] = summ(sub_auc[m] - 0.5, 1) | {
                "auc_mean": float(sub_auc[m].mean())}
            if pipe == "primary" and m != "M2":
                loso = []
                for i in range(len(tr)):
                    tri = [j for j in range(len(tr)) if j != i]
                    e = clf(cfg).fit(np.concatenate([F[j] for j in tri]),
                                     np.concatenate([ys[j] for j in tri]))
                    loso.append(roc_auc_score(ys[i], e.decision_function(F[i])))
                out[f"{m}_natural_loso_auc"] = float(np.mean(loso))
            print(pipe, m, json.dumps(out[f"{m}_vs_chance"]), flush=True)
        out["P_M1_minus_M0"] = summ(sub_auc["M1"] - sub_auc["M0"], 1)
        out["b_M2_minus_M0"] = summ(sub_auc["M2"] - sub_auc["M0"], 2)
        out["b_M3_minus_M0"] = summ(sub_auc["M3"] - sub_auc["M0"], 2)
        out["subject_auc"] = {k: v.tolist() for k, v in sub_auc.items()}
        res[pipe] = out
        (OUT / "q42.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
        print(pipe, "P", json.dumps(out["P_M1_minus_M0"]), flush=True)


if __name__ == "__main__":
    main()
