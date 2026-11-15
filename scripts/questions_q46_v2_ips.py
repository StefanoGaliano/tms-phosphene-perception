"""Q46 (docs/QUESTIONS_PREREG.md §17, tag questions-prereg-amend10): v2 recipe tested once on IPS.

v2 = ds006171 + Eklund, equal dataset weight, window mean 0.15-0.60 s, C = 0.01.
[P] IPS subject AUC(v2) - AUC(frozen, Q41) > 0, one-sided.
Usage: python scripts/questions_q46_v2_ips.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_s7_r4b as s7mod  # noqa: E402
import questions_q42_latency as q42  # noqa: E402
import questions_q44_pooled as q44  # noqa: E402

from start_vision.decoding import across, channels, config  # noqa: E402
from start_vision.decoding.detector import Detector  # noqa: E402
from start_vision.decoding.sources import Eklund2018Adapter  # noqa: E402

ROOT = s7mod.ROOT
OUT = ROOT / "results" / "questions_to_solve" / "prereg_q46"
IPS = ROOT / "results" / "ips"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load(ROOT / "configs" / "decoding.yaml")
    ds_sets, _, _ = s7mod.training_sets(cfg)
    ad = Eklund2018Adapter(cfg)
    ek_sets = [ad.load(s, ROOT) for s in ad.subjects(ROOT)]
    det = Detector.load(s7mod.FROZEN)
    hand = pd.read_csv(IPS / "hand_groups.tsv", sep="\t", dtype=str, index_col=0)["hand"]
    res: dict = {"prereg": "docs/QUESTIONS_PREREG.md §17, tag questions-prereg-amend10",
                 "n_ds006171": len(ds_sets), "n_eklund_units": len(ek_sets)}
    for pipe in ("primary", "paperlike"):
        inc = pd.read_csv(IPS / f"inclusion_{pipe}.tsv", sep="\t", dtype={"subject": str})
        inc = inc[inc["included"]]
        frozen = pd.read_csv(IPS / f"per_recording_{pipe}.tsv", sep="\t",
                             dtype={"subject": str}).set_index("recording")["auc"]
        test = []
        for rid in inc["recording"]:
            ep = mne.read_epochs(ROOT / "derivatives" / "sv_ips_clean" / pipe
                                 / f"{rid}_desc-detector_epo.fif", verbose="error")
            test.append(s7mod.tms_trialset(ep, det, rid))
        h = channels.harmonize(ds_sets + ek_sets + test)
        nd, ne = len(ds_sets), len(ek_sets)
        units = [across.normalize(s, cfg) for s in h]
        tr, te = units[:nd + ne], units[nd + ne:]
        F = [q42.feats("M1", u.X, u.times, False) for u in tr]
        ys = [u.labels("seen") for u in tr]
        ds = ["ds006171"] * nd + ["eklund"] * ne
        Fte = [q42.feats("M1", u.X, u.times, True) for u in te]
        models = {"v2": q44.fit(cfg, F, ys, ds), "eklund_only": q44.fit(cfg, F[nd:], ys[nd:], ds[nd:])}
        rows = []
        for u, f, (rid, sub, site) in zip(te, Fte, inc[["recording", "subject", "site"]].itertuples(
                index=False), strict=True):
            r = {"recording": rid, "subject": sub, "site": site, "frozen": float(frozen[rid])}
            for k, m in models.items():
                r[k] = float(roc_auc_score(u.labels("seen"), m.decision_function(f)))
            rows.append(r)
            print(pipe, rid, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
        pr = pd.DataFrame(rows)
        pr.to_csv(OUT / f"q46_per_recording_{pipe}.tsv", sep="\t", index=False, float_format="%.5f")
        sv = pr.groupby("subject")[["frozen", "v2", "eklund_only"]].mean()
        out: dict = {"n_subjects": len(sv), "n_recordings": len(pr), "n_channels": len(te[0].ch_names),
                     "P_v2_minus_frozen": q42.summ((sv["v2"] - sv["frozen"]).to_numpy(), 1),
                     "frozen_mean": float(sv["frozen"].mean()),
                     "a_v2_vs_chance": q42.summ(sv["v2"].to_numpy() - 0.5, 1)
                     | {"auc_mean": float(sv["v2"].mean())},
                     "d_eklund_only_vs_chance": q42.summ(sv["eklund_only"].to_numpy() - 0.5, 1)
                     | {"auc_mean": float(sv["eklund_only"].mean())},
                     "d_eklund_only_minus_frozen": q42.summ((sv["eklund_only"] - sv["frozen"]).to_numpy(), 2)}
        for site in ("P3", "P4"):
            g = pr[pr["site"] == site]
            out[f"b_{site}"] = {"v2": float(g["v2"].mean()), "frozen": float(g["frozen"].mean()),
                                "diff": q42.summ((g["v2"] - g["frozen"]).to_numpy(), 2)}
        grp = sv.join(hand, how="left")
        a = grp.loc[grp["hand"] == "right", "v2"].to_numpy()
        b = grp.loc[grp["hand"] == "left", "v2"].to_numpy()
        if len(a) and len(b):
            out["e_hand_groups"] = {"right_mean": float(a.mean()), "left_mean": float(b.mean()),
                                    "n_right": len(a), "n_left": len(b),
                                    "p_mannwhitney": float(mannwhitneyu(a, b).pvalue)}
        out["subject_auc"] = sv.reset_index().to_dict("records")
        res[pipe] = out
        (OUT / "q46.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
        print(pipe, "P", json.dumps(out["P_v2_minus_frozen"]), flush=True)


if __name__ == "__main__":
    main()
