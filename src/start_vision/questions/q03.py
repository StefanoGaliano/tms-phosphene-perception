"""Q03 'unseen but correct': guessing-mixture test (P§4)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from start_vision.decoding import stats as dstats
from start_vision.questions import figures as fg
from start_vision.questions import io
from start_vision.questions.params import CHANCE_DS006171, CHANCE_EKLUND, MIN_CELL, TOST_BOUND
from start_vision.questions.report import group_row, write_readme
from start_vision.questions.stats import one_sample


def mixture(a: float, c: float, m_i: float, m_g: float) -> tuple[float, float]:
    """f = max(0, (a - c)/(1 - c)); D_pred = f / (f + (1 - f) c) * (m_I - m_G) (P§4 verbatim)."""
    f = max(0.0, (a - c) / (1 - c))
    if f == 0.0:
        return 0.0, 0.0
    return f, f / (f + (1 - f) * c) * (m_i - m_g)


def session_values(trials: pd.DataFrame, c: float = CHANCE_EKLUND) -> pd.DataFrame:
    rows = []
    for (g, fp), s in trials.groupby(["gabor", "fp"], sort=True):
        r1 = s[s["rating"] == 1]
        cor, wro = r1[r1["obj_corr"] == 1], r1[r1["obj_corr"] == 0]
        ref = s[(s["rating"] == 2) & (s["obj_corr"] == 1)]
        r: dict[str, Any] = {"gabor": g, "fp": fp, "n_r1": len(r1), "n_r1_correct": len(cor),
                             "n_r1_wrong": len(wro), "n_r2_correct": len(ref)}
        if min(len(cor), len(wro), len(ref)) >= MIN_CELL:
            a = len(cor) / len(r1)
            m_g, m_i = float(wro["p_seen"].mean()), float(ref["p_seen"].mean())
            f, dp = mixture(a, c, m_i, m_g)
            d_obs = float(cor["p_seen"].mean()) - m_g
            r |= {"a": a, "f": f, "m_G": m_g, "m_I": m_i, "D_pred": dp, "D_obs": d_obs,
                  "D_obs_minus_pred": d_obs - dp}
        rows.append(r)
    return pd.DataFrame(rows)


def group(per: pd.DataFrame, gabor: str = "large") -> dict[str, Any]:
    s = per[per["gabor"] == gabor]
    v = s["D_obs_minus_pred"].dropna().to_numpy() if "D_obs_minus_pred" in s else np.array([])
    return {"primary": one_sample(v, 0.0, tail=2),
            "tost": dstats.tost(v, TOST_BOUND) if len(v) > 1 else {"p": np.nan},
            "D_obs": one_sample(s.get("D_obs", pd.Series(dtype=float)), 0.0, tail=2),
            "D_pred_mean": float(s["D_pred"].mean()) if "D_pred" in s else np.nan}


def interpretation(r: dict[str, Any]) -> str:
    p, m, eq = r["primary"]["p"], r["primary"]["mean"], r["tost"]["p"]
    if p < 0.05 and m < 0:
        return ("below the prediction: informed unseen trials carry less of the signature than "
                "weak seen ones (evidence for processing without awareness; P§4 rule).")
    if p < 0.05 and m > 0:
        return "above the prediction: not predicted by either account (P§4 rule)."
    if eq < 0.05:
        return ("consistent with the criterion account: D_obs - D_pred not different from 0 and "
                "equivalent within ±0.03 (P§4 rule).")
    return ("D_obs - D_pred not different from 0 but not shown equivalent: inconclusive "
            "(P§4 rule).")


def ds006171_check(trials: pd.DataFrame) -> pd.DataFrame:
    """Unseen trials with the orientation question; a, f, D_obs on the LOSO score (answer 5)."""
    rows = []
    for s, g in trials[~trials["seen"] & trials["correct"].notna()].groupby("subject"):
        cor, wro = g[g["correct"] == 1], g[g["correct"] == 0]
        r: dict[str, Any] = {"subject": s, "n": len(g)}
        if min(len(cor), len(wro)) >= MIN_CELL:
            a = len(cor) / len(g)
            r |= {"a": a, "f": max(0.0, (a - CHANCE_DS006171) / (1 - CHANCE_DS006171)),
                  "D_obs": float(cor["score"].mean() - wro["score"].mean())}
        rows.append(r)
    return pd.DataFrame(rows)


def run(root: Path, force: bool = False) -> Path:
    d = io.out_dir(root, "q03", force)
    per = session_values(io.eklund_trials(root))
    per.to_csv(d / "per_session.tsv", sep="\t", index=False, float_format="%.5f")
    ds = ds006171_check(io.ds006171_trials(root))
    ds.to_csv(d / "per_subject_ds006171.tsv", sep="\t", index=False, float_format="%.5f")
    res = {g: group(per, g) for g in ("large", "small")}
    rows = []
    for g, r in res.items():
        role = "[P]" if g == "large" else "replication"
        rows.append(group_row(f"{g}: D_obs - D_pred (two-sided)", role, r["primary"]))
        rows.append({"test": f"{g}: TOST ±{TOST_BOUND} on D_obs - D_pred", "role": "secondary",
                     "p": r["tost"]["p"]})
        rows.append(group_row(f"{g}: D_obs", "secondary", r["D_obs"]))
        rows.append({"test": f"{g}: mean D_pred", "role": "secondary", "mean": r["D_pred_mean"]})
    rows.append(group_row("ds006171: a vs 0.5", "check", one_sample(ds.get("a", []), 0.5, 2)))
    rows.append(group_row("ds006171: D_obs (LOSO score)", "check",
                          one_sample(ds.get("D_obs", []), 0.0, 2)))
    tab = pd.DataFrame(rows)
    tab.to_csv(d / "group.tsv", sep="\t", index=False, float_format="%.5f")
    io.write_primary(d, "q03", "Eklund large: D_obs - D_pred", "two-sided", res["large"]["primary"])
    fig, (ax,) = fg.axes(1, 4.0, 3.6)
    for i, g in enumerate(("large", "small")):
        s = per[per["gabor"] == g].dropna(subset=["D_obs"])
        ax.scatter(s["D_pred"], s["D_obs"], color=fg.SLOTS[i], s=20, label=f"{g} (N {len(s)})")
    lim = [-0.05, 0.2]
    ax.plot(lim, lim, color=fg.GRID, lw=1)
    fg.style(ax, "D_pred (mixture model)", "D_obs (unseen correct - wrong)")
    ax.legend(fontsize=7, frameon=False)
    fg.save(fig, d)
    write_readme(d, root, "Q03 'unseen but correct': guessing-mixture test", "§4",
                 ["results/eklund2018/trial_scores.tsv.gz", "loso_predictions.tsv + staged events"],
                 {"Group tests": tab}, interpretation(res["large"]),
                 ["Mixture assumptions: informed trials always correct; guesses uniform.",
                  "Rating-2 correct trials as the reference class for weak percepts (PROJECT "
                  "CHOICE).", "ds006171: no rating 2, so a, f and D_obs only (answer 5)."])
    return d
