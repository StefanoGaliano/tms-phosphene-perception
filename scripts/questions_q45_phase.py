"""Q45 (docs/QUESTIONS_PREREG.md §16, tag questions-prereg-amend9): prestimulus alpha phase (Q09).

[P] POS z at -0.02 s (posterior ROI, 8-13 Hz, segment -0.60..-0.02 s cut before filtering,
mirror-padded), per recording vs 1000 label permutations; subject = mean z; one-sided sign-flip.
Usage: python scripts/questions_q45_phase.py
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy.signal import hilbert
from sklearn.metrics import roc_auc_score

from start_vision.decoding import stats
from start_vision.phase3.score import ALPHA_BAND, POSTERIOR

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "questions_to_solve" / "prereg_q45"
SEED, NPERM = 20261109, 1000
SEG = (-0.60, -0.02)
TR = (-0.40, -0.05)


def phases(ep: mne.BaseEpochs) -> tuple[np.ndarray, np.ndarray]:
    """(n_trials, n_times_in_segment) circular-mean ROI phase; times."""
    t = ep.times
    m = (t >= SEG[0] - 1e-9) & (t <= SEG[1] + 1e-9)
    x = ep.get_data(picks=[c for c in POSTERIOR if c in ep.ch_names])[:, :, m].astype(np.float64)
    n = x.shape[-1]
    pad = np.concatenate([x[..., ::-1], x, x[..., ::-1]], axis=-1)
    f = filter_data(pad, ep.info["sfreq"], *ALPHA_BAND, method="fir", phase="zero",
                    verbose="error")[..., n:2 * n]
    ph = np.angle(hilbert(f, axis=-1))
    return np.angle(np.exp(1j * ph).mean(1)), t[m]


def itc(ph: np.ndarray) -> np.ndarray:
    return np.abs(np.exp(1j * ph).mean(0))


def pos(ph: np.ndarray, y: np.ndarray) -> np.ndarray:
    return itc(ph[y == 1]) + itc(ph[y == 0]) - 2 * itc(ph)


def pos_z(ph: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    obs = pos(ph, y)
    null = np.stack([pos(ph, rng.permutation(y)) for _ in range(NPERM)])
    return (obs - null.mean(0)) / null.std(0)


def run(pipe: str) -> dict:
    inc = pd.read_csv(ROOT / "results" / "phase3" / f"inclusion_{pipe}.tsv", sep="\t",
                      dtype={"subject": str})
    inc = inc[inc["included"]]
    sc = pd.read_csv(ROOT / "results" / "phase3" / f"scores_{pipe}.tsv", sep="\t",
                     dtype={"subject": str})
    rng = np.random.default_rng(SEED)
    rows, zt, rates = [], {}, []
    edges = np.linspace(-np.pi, np.pi, 7)
    for rid, sub in inc[["recording", "subject"]].itertuples(index=False):
        ep = mne.read_epochs(ROOT / "derivatives" / "sv_p3_clean" / pipe
                             / f"{rid}_desc-detector_epo.fif", verbose="error")
        g = sc[sc["recording"] == rid]
        y = ep.metadata["seen"].to_numpy().astype(int)
        assert len(g) == len(y) and (g["seen"].to_numpy() == y).all()
        ph, tt = phases(ep)
        z = pos_z(ph, y, rng)
        zt[rid] = z
        p0 = ph[:, -1]
        rate = [y[(p0 >= a) & (p0 < b)].mean() if ((p0 >= a) & (p0 < b)).any() else np.nan
                for a, b in zip(edges[:-1], edges[1:], strict=True)]
        rates.append(rate)
        peak = np.abs(p0) < np.pi / 2
        ps = g["p_seen"].to_numpy()
        ok = np.isfinite(ps)
        hi = (ps > np.nanmedian(ps)).astype(int)

        def auc(mk: np.ndarray) -> float:
            mk = mk & ok
            return float(roc_auc_score(y[mk], ps[mk])) if len(set(y[mk])) == 2 else np.nan

        rows.append({"recording": rid, "subject": sub, "pos_z": float(z[-1]),
                     "auc_peak_half": auc(peak), "auc_trough_half": auc(~peak),
                     "pos_z_pseen": float(pos_z(ph[ok], hi[ok], rng)[-1]),
                     "n": len(y), "seen_share": float(y.mean())})
        print(pipe, rid, f"POS z {z[-1]:.2f}", flush=True)
    pr = pd.DataFrame(rows)
    pr.to_csv(OUT / f"q45_per_recording_{pipe}.tsv", sep="\t", index=False, float_format="%.5f")
    sv = pr.groupby("subject")[["pos_z", "auc_peak_half", "auc_trough_half", "pos_z_pseen"]].mean()
    subs = sv.index.tolist()
    rec_sub = dict(zip(pr["recording"], pr["subject"], strict=True))
    zsub = np.stack([np.mean([zt[r] for r in zt if rec_sub[r] == s], axis=0) for s in subs])
    tm = (tt >= TR[0] - 1e-9) & (tt <= TR[1] + 1e-9)
    t_obs, clusters, pv, _ = mne.stats.permutation_cluster_1samp_test(
        zsub[:, tm], n_permutations=10_000, tail=1, seed=SEED, out_type="mask", verbose="error")
    cl = [{"t_start": float(tt[tm][c].min()), "t_end": float(tt[tm][c].max()), "p": float(p)}
          for c, p in zip(clusters, pv, strict=True)]
    x = sv["pos_z"].to_numpy()
    lo, hi_ = stats.bootstrap_ci(x, seed=SEED)
    d = (sv["auc_peak_half"] - sv["auc_trough_half"]).dropna().to_numpy()
    rate = np.nanmean(np.array(rates), 0)
    np.savez(OUT / f"q45_pos_time_{pipe}.npz", zsub=zsub, times=tt, subjects=np.array(subs))
    return {
        "n_subjects": len(subs), "n_recordings": len(pr),
        "P_pos_z_at_-0.02": {"mean": float(x.mean()), "ci95": [lo, hi_],
                             "p_one_sided": float(stats._signflip_p(x, 10_000, SEED, tail=1)),
                             "d_z": float(x.mean() / x.std(ddof=1)), "n_positive": int((x > 0).sum())},
        "a_time_clusters": cl,
        "b_phosphene_rate_by_phase_bin": {"bin_centres_rad": list((edges[:-1] + edges[1:]) / 2),
                                          "rate": rate.tolist(),
                                          "modulation_depth": float(np.nanmax(rate) - np.nanmin(rate))},
        "c_auc_peak_minus_trough": {"peak": float(sv["auc_peak_half"].mean()),
                                    "trough": float(sv["auc_trough_half"].mean()),
                                    "diff": float(d.mean()),
                                    "p_two_sided": float(stats._signflip_p(d, 10_000, SEED, tail=0))},
        "d_pos_z_pseen_median_split": {
            "mean": float(sv["pos_z_pseen"].mean()),
            "p_one_sided": float(stats._signflip_p(sv["pos_z_pseen"].to_numpy(), 10_000, SEED, tail=1))},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"prereg": "docs/QUESTIONS_PREREG.md §16, tag questions-prereg-amend9"}
    for pipe in ("primary", "paperlike"):
        res[pipe] = run(pipe)
        (OUT / "q45.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
        print(pipe, json.dumps(res[pipe]["P_pos_z_at_-0.02"]), flush=True)


if __name__ == "__main__":
    main()
