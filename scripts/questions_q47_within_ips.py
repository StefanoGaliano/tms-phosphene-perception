"""Q47 (docs/QUESTIONS_PREREG.md §18, tag questions-prereg-amend11): within-recording decodability,
parietal (IPS) vs occipital phosphenes. S8 code unchanged on IPS; occipital from saved S8 curves.
[P] IPS subject mean AUC (0.15-0.60 s) < occipital, one-sided permutation of group means.
Usage: python scripts/questions_q47_within_ips.py
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from start_vision.decoding import stats
from start_vision.decoding.detector import Detector
from start_vision.phase3.secondary import s8_sliding

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "questions_to_solve" / "prereg_q47"
FROZEN = ROOT / "detectors" / "sv-perception-tms-robust-v1.0.0"
SEED = 20261109
W, WB = (0.15, 0.60), (0.07, 0.98)


def wmean(a: np.ndarray, t: np.ndarray, w: tuple[float, float]) -> np.ndarray:
    m = (t >= w[0] - 1e-9) & (t <= w[1] + 1e-9)
    return a[..., m].mean(-1)


def perm_less(a: np.ndarray, b: np.ndarray, n: int = 10_000) -> float:
    """One-sided p for mean(a) < mean(b), relabelling units between groups."""
    rng = np.random.default_rng(SEED)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    null = np.empty(n)
    for i in range(n):
        p = rng.permutation(pool)
        null[i] = p[:len(a)].mean() - p[len(a):].mean()
    return float((np.sum(null <= obs) + 1) / (n + 1))


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 10_000) -> list[float]:
    rng = np.random.default_rng(SEED)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ch = list(Detector.load(FROZEN).contract["channels"])
    inc = pd.read_csv(ROOT / "results" / "ips" / "inclusion_primary.tsv", sep="\t",
                      dtype={"subject": str})
    inc = inc[inc["included"]]
    curves, t = {}, None
    for rid in inc["recording"]:
        ep = mne.read_epochs(ROOT / "derivatives" / "sv_ips_clean" / "primary"
                             / f"{rid}_desc-detector_epo.fif", verbose="error")
        t, curves[rid] = s8_sliding(ep, ch, seed=SEED)
        print(rid, f"{wmean(curves[rid], t, W):.3f}", flush=True)
    rec_sub = dict(zip(inc["recording"], inc["subject"], strict=True))
    rec_site = dict(zip(inc["recording"], inc["site"], strict=True))
    subs = sorted(set(rec_sub.values()))
    ips = np.stack([np.mean([curves[r] for r in curves if rec_sub[r] == s], 0) for s in subs])
    z = np.load(ROOT / "results" / "phase3" / "secondary_curves_primary.npz", allow_pickle=True)
    occ, to = z["s8_auc"], z["s8_times"]
    assert np.allclose(to, t)
    np.savez(OUT / "q47_ips_curves.npz", sub_auc=ips, times=t, subjects=np.array(subs))
    res: dict = {"prereg": "docs/QUESTIONS_PREREG.md §18, tag questions-prereg-amend11",
                 "n_ips_subjects": len(subs), "n_ips_recordings": len(curves),
                 "n_occ_subjects": len(occ)}
    for tag, w in (("P_0.15-0.60", W), ("b_0.07-0.98", WB)):
        a, b = wmean(ips, t, w), wmean(occ, t, w)
        res[tag] = {"ips_mean": float(a.mean()), "occ_mean": float(b.mean()),
                    "diff": float(a.mean() - b.mean()), "ci95": boot_diff(a, b),
                    "p_one_sided_ips_lower": perm_less(a, b)}
    cl = stats.cluster_auc(ips, times=t, seed=SEED)
    res["a_ips_clusters"] = [{k: float(v) for k, v in r.items()} for r in
                             cl[cl["p"] < 0.05].to_dict("records")]
    a, b = wmean(ips, t, W).mean(), wmean(occ, t, W).mean()
    # Frozen-detector transfer read from the current H1 results (was hard-coded; the occipital
    # value 0.5399 was the 17-subject run, stale after the sub-18 O1 re-run of 2026-11-16).
    f_ips = float(json.loads((ROOT / "results" / "ips" / "summary.json").read_text())
                  ["primary"]["mean"])
    t2 = pd.read_csv(ROOT / "results" / "phase3" / "t2_tests_primary.tsv", sep="\t")
    f_occ = float(t2.loc[t2["test"] == "H1 primary", "mean"].iloc[0])
    res["c_share_carried"] = {"ips": (f_ips - 0.5) / (a - 0.5), "occ": (f_occ - 0.5) / (b - 0.5),
                              "frozen_transfer": {"ips": f_ips, "occ": f_occ}}
    for site in ("P3", "P4"):
        v = [wmean(curves[r], t, W) for r in curves if rec_site[r] == site]
        res[f"d_{site}"] = {"mean": float(np.mean(v)), "n": len(v)}
    (OUT / "q47.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
