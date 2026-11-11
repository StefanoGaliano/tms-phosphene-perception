"""Q40 (docs/QUESTIONS_PREREG.md §11, tag questions-prereg-amend4): reverse transfer.

Per-time-point decoder trained on all included TMS recordings (phosphene vs none), tested on each
of the 34 included ds006171 subjects (seen vs unseen). [P]: mean AUC 0.15-0.60 s > 0.5.
Usage: python scripts/questions_q40_reverse.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_s7_r4b as s7mod  # noqa: E402

from start_vision.decoding import across, channels, config, stats  # noqa: E402
from start_vision.decoding.detector import Detector  # noqa: E402

ROOT = s7mod.ROOT
OUT = ROOT / "results" / "questions_to_solve" / "prereg_q40"
WINDOWS = {"P detector 0.15-0.60": (0.15, 0.60), "VAN 0.18-0.28": (0.18, 0.28),
           "LP 0.35-0.55": (0.35, 0.55), "prestim -0.5-0": (-0.5, 0.0)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load(ROOT / "configs" / "decoding.yaml")
    nat, _, _ = s7mod.training_sets(cfg)
    det = Detector.load(s7mod.FROZEN)
    res: dict = {"prereg": "docs/QUESTIONS_PREREG.md §11, tag questions-prereg-amend4",
                 "n_test_subjects": len(nat)}
    for pipe in ("primary", "paperlike"):
        inc = s7mod.included(pipe)
        tms = [s7mod.tms_trialset(s7mod.read_rec(pipe, r), det, r) for r in inc["recording"]]
        h = channels.harmonize(tms + nat)
        df = across.transfer(h[:len(tms)], h[len(tms):], cfg)
        times = h[0].times
        auc = np.vstack([np.asarray(a, dtype=float) for a in df["auc"]])
        np.savez(OUT / f"q40_auc_{pipe}.npz", auc=auc, times=times,
                 subjects=df["subject"].to_numpy())
        r: dict = {"n_train_recordings": len(tms), "n_test_subjects": len(auc)}
        for k, (lo, hi) in WINDOWS.items():
            m = (times >= lo - 1e-9) & (times <= hi + 1e-9)
            v = auc[:, m].mean(1)
            r[k] = stats.window_test(v, seed=cfg.seed) | {"n_above": int((v > 0.5).sum())}
        post = times >= -1e-9
        cl = stats.cluster_auc(auc[:, post], times[post], seed=cfg.seed)
        r["clusters"] = cl[cl["p"] < 0.05].to_dict("records")
        r["peak_auc"] = float(auc.mean(0).max())
        r["peak_time_s"] = float(times[int(auc.mean(0).argmax())])
        res[pipe] = r
        print(pipe, json.dumps(r["P detector 0.15-0.60"], default=float), flush=True)
        (OUT / "q40.json").write_text(json.dumps(res, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
