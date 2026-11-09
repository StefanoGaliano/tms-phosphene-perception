"""Q37b pulses needed (P§7, amendment 1): independence model, k* on O1, test on O2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from start_vision.phase3.analyses import group_one_sided, rec_auc, subject_values
from start_vision.questions import figures as fg
from start_vision.questions import io
from start_vision.questions.params import K_STAR_AUC, N_BLOCKS, Q37B_KS, SEED
from start_vision.questions.report import group_row, write_readme
from start_vision.questions.stats import one_sample


def predicted_auc(auc1: float, k: int) -> float:
    """d' = sqrt(2) Phi^-1(AUC); AUC_k = Phi(d' sqrt(k) / sqrt(2)) (P§7)."""
    dp = np.sqrt(2) * norm.ppf(auc1)
    return float(norm.cdf(dp * np.sqrt(k) / np.sqrt(2)))


def block_auc(p1: np.ndarray, p0: np.ndarray, k: int, n_blocks: int,
              rng: np.random.Generator) -> float:
    if k > min(len(p1), len(p0)):
        return np.nan
    b1 = np.array([rng.choice(p1, k, replace=False).mean() for _ in range(n_blocks)])
    b0 = np.array([rng.choice(p0, k, replace=False).mean() for _ in range(n_blocks)])
    return float(roc_auc_score(np.r_[np.ones(n_blocks), np.zeros(n_blocks)], np.r_[b1, b0]))


def recording_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, (rid, g) in enumerate(scores.groupby("recording", sort=True)):
        rng = np.random.default_rng([SEED, i])           # one generator per recording (answer 10)
        p1 = g.loc[g["seen"] == 1, "p_seen"].to_numpy()
        p0 = g.loc[g["seen"] == 0, "p_seen"].to_numpy()
        a1 = rec_auc(g, "p_seen")
        r: dict[str, Any] = {"recording": rid, "subject": g["subject"].iloc[0],
                             "site": g["site"].iloc[0], "auc1": a1}
        for k in Q37B_KS:
            r[f"pred_{k}"] = predicted_auc(a1, k)
            r[f"obs_{k}"] = block_auc(p1, p0, k, N_BLOCKS, rng)
            r[f"obs_minus_pred_{k}"] = r[f"obs_{k}"] - r[f"pred_{k}"]
        rows.append(r)
    return pd.DataFrame(rows)


def choose_k_star(per_o1: pd.DataFrame) -> int | None:
    """Smallest k whose group mean (over subjects) of O1 observed AUC reaches 0.75 (P§7)."""
    for k in Q37B_KS:
        if subject_values(per_o1, f"obs_{k}").mean() >= K_STAR_AUC:
            return k
    return None


def o2_test(per: pd.DataFrame, k_star: int | None, d: Path) -> dict[str, Any]:
    """Writes k_star.json first, then computes the O2 statistic (P§7)."""
    (d / "k_star.json").write_text(json.dumps({"k_star": k_star, "chosen_on": "O1 recordings",
                                               "rule": f"smallest k with mean AUC >= {K_STAR_AUC}"
                                               }) + "\n")
    k = k_star if k_star is not None else max(Q37B_KS)
    o2 = per[per["site"] == "O2"]
    return {"k_used": k, **group_one_sided(subject_values(o2, f"obs_{k}"), seed=SEED)}


def run(root: Path, force: bool = False) -> Path:
    d = io.out_dir(root, "q37b", force)
    per = recording_table(io.gin_scores(root))
    per.to_csv(d / "per_recording.tsv", sep="\t", index=False, float_format="%.5f")
    k_star = choose_k_star(per[per["site"] == "O1"])
    prim = o2_test(per, k_star, d)
    rows = [{"test": f"O2 block AUC at k = {prim['k_used']} > 0.5 (k* "
                     f"{'not reached' if k_star is None else k_star} on O1)", "role": "[P]",
             "n": prim["n"], "mean": prim["mean"], "ci_low": prim["ci_low"],
             "ci_high": prim["ci_high"], "d_z": prim["d_z"], "p": prim["p"], "tail": 1.0,
             "n_positive": prim["n_above"]}]
    for k in Q37B_KS:
        rows.append(group_row(f"observed - predicted, k = {k} (pooled sites)", "secondary",
                              one_sample(subject_values(per, f"obs_minus_pred_{k}"), 0.0, 2)))
        for site in ("O1", "O2"):
            s = per[per["site"] == site]
            rows.append({"test": f"{site} mean observed / predicted AUC, k = {k}",
                         "role": "descriptive", "mean": subject_values(s, f"obs_{k}").mean(),
                         "ci_low": subject_values(s, f"pred_{k}").mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(d / "group.tsv", sep="\t", index=False, float_format="%.5f")
    io.write_primary(d, "q37b", f"O2 block AUC at k = {prim['k_used']}", "one-sided",
                     {k: float(prim[k]) for k in ("n", "mean", "ci_low", "ci_high", "p")})
    fig, axs = fg.axes(2)
    for ax, site in zip(axs, ("O1", "O2"), strict=True):
        s = per[per["site"] == site]
        obs = [subject_values(s, f"obs_{k}").mean() for k in Q37B_KS]
        pred = [subject_values(s, f"pred_{k}").mean() for k in Q37B_KS]
        ax.plot(Q37B_KS, pred, "o--", color=fg.MUTED, lw=1.5, label="independence model")
        ax.plot(Q37B_KS, obs, "o-", color=fg.SLOTS[0], lw=2, label="observed blocks")
        ax.axhline(K_STAR_AUC, color=fg.GRID, lw=1)
        if k_star is not None:
            ax.axvline(k_star, color=fg.INK, lw=1, ls=":")
        fg.style(ax, "pulses per block (k)", "mean AUC over subjects", f"site {site}")
        ax.legend(fontsize=7, frameon=False)
    fg.save(fig, d)
    write_readme(d, root, "Q37b pulses needed for a block read-out (GIN)", "§7",
                 ["results/phase3/scores_primary.tsv (included recordings)"], {"Tests": tab},
                 f"k* on O1: {'not reached' if k_star is None else k_star}; O2 block AUC at "
                 f"k = {prim['k_used']}: {prim['mean']:.3f} (planning estimate, P§7).",
                 ["Subjects with single-pulse AUC < 0.5 are reported, not excluded.",
                  "O1 and O2 are mostly the same subjects: O2 holds out recordings, not people.",
                  "One generator per recording (answer 10): first-look numbers not reproduced "
                  "exactly.",
                  "Blocks are random draws within class, so dependence between neighbouring "
                  "pulses cannot enter them: observed - predicted tests the equal-variance "
                  "Gaussian shape of the single-pulse scores, not independence (P§8 note)."])
    return d
