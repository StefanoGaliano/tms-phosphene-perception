"""Q39 (docs/QUESTIONS_PREREG.md §10, tag questions-prereg-amend3): time-shifted transfer.

The S7 decoder (Phase 2 per-time-point recipe, tms-robust channels) trained on all included
ds006171 subjects as a GeneralizingEstimator; each training-time classifier tested at every time
point of each included TMS recording. Subject = mean of included sites.

[P] mean AUC, train 0.18-0.28 s x test 0.08-0.18 s, > 0.5 one-sided (sign-flip, 10 000).
Usage: python scripts/questions_q39_tg.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_s7_r4b as s7mod  # noqa: E402

from start_vision.decoding import across, channels, config, models, stats  # noqa: E402
from start_vision.decoding.data import TrialSet  # noqa: E402
from start_vision.decoding.detector import Detector  # noqa: E402

ROOT = s7mod.ROOT
OUT = ROOT / "results" / "questions_to_solve" / "prereg_q39"
T0, T1 = 0.0, 1.0
TRAIN_VAN, TEST_EARLY, TEST_LATE = (0.18, 0.28), (0.08, 0.18), (0.28, 0.38)
TRAIN_LP, TEST_LP_EARLY = (0.35, 0.55), (0.25, 0.35)


def crop(ts: TrialSet) -> TrialSet:
    m = (ts.times >= T0 - 1e-9) & (ts.times <= T1 + 1e-9)
    return TrialSet(ts.X[:, :, m], ts.times[m], ts.ch_names, ts.sfreq, ts.meta, ts.info)


def mask(times: np.ndarray, w: tuple[float, float]) -> np.ndarray:
    return (times >= w[0] - 1e-9) & (times <= w[1] + 1e-9)


def run_pipeline(pipe: str, sets: list[TrialSet], cfg: config.DecodingConfig) -> dict:
    det = Detector.load(s7mod.FROZEN)
    inc = s7mod.included(pipe)
    test = [s7mod.tms_trialset(s7mod.read_rec(pipe, r), det, r) for r in inc["recording"]]
    h = channels.harmonize(sets + test)
    tr = TrialSet.concat([across.normalize(s, cfg) for s in h[:len(sets)]])
    tr = crop(tr)
    est = models.generalizing(cfg).fit(tr.X, tr.labels("seen"))
    mats = {}
    for s in h[len(sets):]:
        n = crop(across.normalize(s, cfg))
        rid = str(n.meta["subject"].iloc[0])
        mats[rid] = np.asarray(est.score(n.X, n.labels("seen")), dtype=float)
        print(f"{pipe} {rid}: diag mean {np.diag(mats[rid]).mean():.3f}", flush=True)
    times = tr.times
    rec_sub = dict(zip(inc["recording"], inc["subject"], strict=True))
    subs = sorted(set(rec_sub.values()))
    sub = np.stack([np.mean([mats[r] for r in mats if rec_sub[r] == s], axis=0) for s in subs])
    np.savez(OUT / f"q39_tg_{pipe}.npz", sub_auc=sub, times=times, subjects=np.array(subs))

    def cell(trw: tuple[float, float], tew: tuple[float, float]) -> np.ndarray:
        return sub[:, mask(times, trw)][:, :, mask(times, tew)].mean(axis=(1, 2))

    seed = cfg.seed
    prim = cell(TRAIN_VAN, TEST_EARLY)
    late = cell(TRAIN_VAN, TEST_LATE)
    diag_van = np.array([np.diag(m)[mask(times, TRAIN_VAN)].mean() for m in sub])
    van_rows = sub[:, mask(times, TRAIN_VAN)].mean(axis=1)  # (n_sub, n_test)
    peak_t = times[van_rows.argmax(axis=1)]
    res = {
        "n_subjects": len(subs), "n_recordings": len(mats),
        "P_van_train_early_test": stats.window_test(prim, seed=seed)
        | {"one_sided_p": float(group_p(prim - 0.5, seed)), "n_above": int((prim > 0.5).sum())},
        "a_shift_early_minus_late": paired(prim - late, seed),
        "van_diagonal_mean": float(diag_van.mean()),
        "van_train_late_test_mean": float(late.mean()),
        "b_lp_train_earlier_test": stats.window_test(cell(TRAIN_LP, TEST_LP_EARLY), seed=seed),
        "c_peak_test_time_group_s": float(times[van_rows.mean(0).argmax()]),
        "c_peak_test_time_subjects_s": [float(x) for x in peak_t],
        "c_peak_test_time_median_s": float(np.median(peak_t)),
    }
    cl = stats.cluster_auc(sub, seed=seed)
    res["d_clusters"] = cl[cl["p"] < 0.05].to_dict("records")
    return res


def group_p(x: np.ndarray, seed: int, n: int = 10_000) -> float:
    rng = np.random.default_rng(seed)
    obs = x.mean()
    flips = rng.choice([-1, 1], size=(n, len(x)))
    null = (flips * x).mean(axis=1)
    return float((np.sum(null >= obs) + 1) / (n + 1))


def paired(d: np.ndarray, seed: int, n: int = 10_000) -> dict:
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1, 1], size=(n, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return {"mean": float(d.mean()), "p_two_sided": float((np.sum(null >= abs(d.mean())) + 1) / (n + 1)),
            "n_positive": int((d > 0).sum()), "d_z": float(d.mean() / d.std(ddof=1))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load(ROOT / "configs" / "decoding.yaml")
    sets, _, _ = s7mod.training_sets(cfg)
    res = {"prereg": "docs/QUESTIONS_PREREG.md §10, tag questions-prereg-amend3",
           "n_train_subjects": len(sets), "times_s": [T0, T1]}
    for pipe in ("primary", "paperlike"):
        res[pipe] = run_pipeline(pipe, sets, cfg)
        (OUT / "q39.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
        print(json.dumps(res[pipe]["P_van_train_early_test"], default=float), flush=True)


if __name__ == "__main__":
    main()
