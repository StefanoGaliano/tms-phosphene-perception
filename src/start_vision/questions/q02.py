"""Q02 seen with wrong content (P§3): Eklund correctness slope; ds006171 residualized AUC."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from start_vision.questions import figures as fg
from start_vision.questions import io
from start_vision.questions.params import MIN_CELL
from start_vision.questions.q01 import _ols_slope
from start_vision.questions.report import group_row, write_readme
from start_vision.questions.stats import one_sample


def eklund_slopes(trials: pd.DataFrame) -> pd.DataFrame:
    """Rating >= 2; per session OLS p_seen ~ correct + centred duration; >= 3 wrong and correct."""
    rows = []
    for (g, fp), s in trials[trials["rating"] >= 2].groupby(["gabor", "fp"], sort=True):
        c = s["obj_corr"].to_numpy(float)
        r: dict[str, Any] = {"gabor": g, "fp": fp, "n_correct": int((c == 1).sum()),
                             "n_wrong": int((c == 0).sum())}
        if min(r["n_correct"], r["n_wrong"]) >= MIN_CELL:
            dur = s["duration_ms"].to_numpy(float)
            y = s["p_seen"].to_numpy(float)
            r["slope_cov"] = _ols_slope(y, c, dur - dur.mean())
            r["mean_correct"] = float(y[c == 1].mean())
            r["mean_wrong"] = float(y[c == 0].mean())
        else:
            r["slope_cov"] = r["mean_correct"] = r["mean_wrong"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def eklund_group(per: pd.DataFrame, gabor: str = "large") -> dict[str, float]:
    return one_sample(per.loc[per["gabor"] == gabor, "slope_cov"], 0.0, tail=1)


def _resid(score: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    b = np.polyfit(contrast, score, 1)
    return np.asarray(score - np.polyval(b, contrast))


def ds006171_auc(trials: pd.DataFrame) -> pd.DataFrame:
    t = trials.copy()
    t["score_resid"] = np.nan
    for _, g in t.groupby(["subject", "task"]):
        t.loc[g.index, "score_resid"] = _resid(g["score"].to_numpy(float),
                                               g["gabor_contrast"].to_numpy(float))
    rows = []
    for s, g in t[t["seen"] & t["correct"].notna()].groupby("subject", sort=True):
        y = g["correct"].to_numpy() == 1
        r: dict[str, Any] = {"subject": s, "n_correct": int(y.sum()), "n_wrong": int((~y).sum())}
        ok = min(r["n_correct"], r["n_wrong"]) >= MIN_CELL
        r["auc_raw"] = roc_auc_score(y, g["score"]) if ok else np.nan
        r["auc_resid"] = roc_auc_score(y, g["score_resid"]) if ok else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def run(root: Path, force: bool = False) -> Path:
    d = io.out_dir(root, "q02", force)
    ek = eklund_slopes(io.eklund_trials(root))
    ek.to_csv(d / "per_session_eklund.tsv", sep="\t", index=False, float_format="%.5f")
    ds = ds006171_auc(io.ds006171_trials(root))
    ds.to_csv(d / "per_subject_ds006171.tsv", sep="\t", index=False, float_format="%.5f")
    res = {"large": eklund_group(ek, "large"), "small": eklund_group(ek, "small"),
           "ds": one_sample(ds["auc_resid"], 0.5, tail=2)}
    tab = pd.DataFrame([group_row("Eklund large: correctness slope (centred duration) > 0", "[P]",
                                  res["large"]),
                        group_row("Eklund small: same", "replication", res["small"]),
                        group_row("ds006171: AUC correct vs wrong, contrast-residualized, vs 0.5",
                                  "secondary", res["ds"])])
    tab.to_csv(d / "group.tsv", sep="\t", index=False, float_format="%.5f")
    io.write_primary(d, "q02", "Eklund large: correctness slope > 0", "one-sided", res["large"])
    fig, (a, b) = fg.axes(2)
    for i, g in enumerate(("large", "small")):
        v = ek.loc[ek["gabor"] == g, "slope_cov"].dropna()
        a.scatter(np.full(len(v), i), v, color=fg.SLOTS[i], s=18, label=f"{g} (N {len(v)})")
    a.axhline(0, color=fg.GRID, lw=1)
    a.set_xticks([0, 1], ["large", "small"])
    fg.style(a, "Eklund, rating >= 2", "P(seen) slope: correct - wrong")
    a.legend(fontsize=7, frameon=False)
    v = ds["auc_resid"].dropna()
    b.scatter(np.zeros(len(v)), v, color=fg.SLOTS[0], s=18, label=f"subjects (N {len(v)})")
    b.axhline(0.5, color=fg.GRID, lw=1)
    b.set_xticks([])
    fg.style(b, "ds006171, seen trials", "AUC correct vs wrong (resid.)")
    b.legend(fontsize=7, frameon=False)
    fg.save(fig, d)
    ok = res["large"]["p"] < 0.05 and res["ds"]["p"] < 0.05 and res["ds"]["mean"] > 0.5
    write_readme(d, root, "Q02 seen with wrong content (Eklund; ds006171)", "§3",
                 ["results/eklund2018/trial_scores.tsv.gz", "loso_predictions.tsv + staged events"],
                 {"Group tests": tab},
                 ("both datasets positive: the read-out needs content-bearing evidence (P§3 rule)."
                  if ok else "not positive in both datasets (P§3 rule): see the table."),
                 ["Wrong-content 'seen' trials can be false reports (P§3).",
                  "Same data as the first look (P§0)."])
    return d
