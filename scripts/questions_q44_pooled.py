"""Q44 (docs/QUESTIONS_PREREG.md §15, tag questions-prereg-amend8): ds006171 + Eklund training (37g).

[P] M0: TMS subject AUC(pooled) - AUC(ds006171 only) > 0, one-sided. Equal total weight per dataset.
Usage: python scripts/questions_q44_pooled.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_s7_r4b as s7mod  # noqa: E402
import questions_q42_latency as q42  # noqa: E402

from start_vision.decoding import across, channels, config  # noqa: E402
from start_vision.decoding.data import TrialSet  # noqa: E402
from start_vision.decoding.detector import Detector  # noqa: E402
from start_vision.decoding.sources import Eklund2018Adapter  # noqa: E402

ROOT = s7mod.ROOT
OUT = ROOT / "results" / "questions_to_solve" / "prereg_q44"


def symmetric(ts: TrialSet) -> TrialSet:
    """Homologous pairs (odd/even suffix) -> their mean; midline and unpaired kept."""
    names = list(ts.ch_names)
    idx = {n: i for i, n in enumerate(names)}
    cols, out_names, used = [], [], set()
    for n in names:
        if n in used:
            continue
        m = re.fullmatch(r"([A-Za-z]+)(\d+)", n)
        partner = f"{m.group(1)}{int(m.group(2)) + 1}" if m and int(m.group(2)) % 2 else None
        if partner in idx:
            cols.append((ts.X[:, idx[n]] + ts.X[:, idx[partner]]) / 2)
            out_names.append(f"{n}/{partner}")
            used |= {n, partner}
        elif not (m and int(m.group(2)) % 2 == 0 and f"{m.group(1)}{int(m.group(2)) - 1}" in idx):
            cols.append(ts.X[:, idx[n]])
            out_names.append(n)
            used.add(n)
    return TrialSet(np.stack(cols, 1), ts.times, tuple(out_names), ts.sfreq, ts.meta, None)


def fit(cfg, F: list[np.ndarray], ys: list[np.ndarray], ds: list[str]):  # type: ignore[no-untyped-def]
    n = {d: sum(len(f) for f, dd in zip(F, ds, strict=True) if dd == d) for d in set(ds)}
    w = np.concatenate([np.full(len(f), 1.0 / n[d]) for f, d in zip(F, ds, strict=True)])
    w *= len(w) / w.sum()
    return q42.clf(cfg).fit(np.concatenate(F), np.concatenate(ys), logisticregression__sample_weight=w)


def tms_auc(est, te: list[TrialSet], Fte: list[np.ndarray], rec_sub: dict, subs: list) -> np.ndarray:  # type: ignore[no-untyped-def]
    rec = {str(s.meta["subject"].iloc[0]): roc_auc_score(s.labels("seen"), est.decision_function(f))
           for s, f in zip(te, Fte, strict=True)}
    return np.array([np.mean([rec[r] for r in rec if rec_sub[r] == su]) for su in subs])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load(ROOT / "configs" / "decoding.yaml")
    ds_sets, _, _ = s7mod.training_sets(cfg)
    ad = Eklund2018Adapter(cfg)
    ek_sets = [ad.load(s, ROOT) for s in ad.subjects(ROOT)]
    print(f"Eklund units: {len(ek_sets)}", flush=True)
    det = Detector.load(s7mod.FROZEN)
    res: dict = {"prereg": "docs/QUESTIONS_PREREG.md §15, tag questions-prereg-amend8",
                 "n_ds006171": len(ds_sets), "n_eklund_units": len(ek_sets)}
    for pipe in ("primary", "paperlike"):
        inc = s7mod.included(pipe)
        test = [s7mod.tms_trialset(s7mod.read_rec(pipe, r), det, r) for r in inc["recording"]]
        h = channels.harmonize(ds_sets + ek_sets + test)
        nd, ne = len(ds_sets), len(ek_sets)
        units = [across.normalize(s, cfg) for s in h]
        dsn, ekn, te = units[:nd], units[nd:nd + ne], units[nd + ne:]
        rec_sub = dict(zip(inc["recording"], inc["subject"], strict=True))
        subs = sorted(set(rec_sub.values()))
        out: dict = {"n_channels": len(units[0].ch_names)}
        auc: dict[str, np.ndarray] = {}
        variants = [("M0", False), ("M1", False), ("M0", True)]
        for m, sym in variants:
            tr = [symmetric(u) if sym else u for u in dsn + ekn]
            tt = [symmetric(u) if sym else u for u in te]
            F = [q42.feats(m, u.X, u.times, False) for u in tr]
            ys = [u.labels("seen") for u in tr]
            ds = ["ds006171"] * nd + ["eklund"] * ne
            Fte = [q42.feats(m, u.X, u.times, True) for u in tt]
            tag = f"{m}{'_sym' if sym else ''}"
            auc[f"{tag}_pooled"] = tms_auc(fit(cfg, F, ys, ds), tt, Fte, rec_sub, subs)
            if not sym:
                auc[f"{tag}_ds_only"] = tms_auc(fit(cfg, F[:nd], ys[:nd], ds[:nd]), tt, Fte,
                                                rec_sub, subs)
                auc[f"{tag}_eklund_only"] = tms_auc(fit(cfg, F[nd:], ys[nd:], ds[nd:]), tt, Fte,
                                                    rec_sub, subs)
                if pipe == "primary":
                    e1 = fit(cfg, F[:nd], ys[:nd], ds[:nd])
                    e2 = fit(cfg, F[nd:], ys[nd:], ds[nd:])
                    out[f"c_{tag}_ds_to_eklund"] = float(np.mean(
                        [roc_auc_score(y, e1.decision_function(f)) for f, y in zip(F[nd:], ys[nd:], strict=True)]))
                    out[f"c_{tag}_eklund_to_ds"] = float(np.mean(
                        [roc_auc_score(y, e2.decision_function(f)) for f, y in zip(F[:nd], ys[:nd], strict=True)]))
            for k in [k for k in auc if k.startswith(tag + "_")]:
                out[f"{k}_vs_chance"] = q42.summ(auc[k] - 0.5, 1) | {"auc_mean": float(auc[k].mean())}
                print(pipe, k, round(float(auc[k].mean()), 4), flush=True)
        out["P_M0_pooled_minus_ds_only"] = q42.summ(auc["M0_pooled"] - auc["M0_ds_only"], 1)
        out["a_M1_pooled_minus_ds_only"] = q42.summ(auc["M1_pooled"] - auc["M1_ds_only"], 2)
        out["b_M0_sym_minus_M0_pooled"] = q42.summ(auc["M0_sym_pooled"] - auc["M0_pooled"], 2)
        out["subject_auc"] = {k: v.tolist() for k, v in auc.items()}
        res[pipe] = out
        (OUT / "q44.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
        print(pipe, "P", json.dumps(out["P_M0_pooled_minus_ds_only"]), flush=True)


if __name__ == "__main__":
    main()
