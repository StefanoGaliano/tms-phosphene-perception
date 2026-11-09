"""Questions_to_solve Q01-Q03 on Eklund & Wiens 2018 (frozen tms-robust detector scores).

Run: ~/miniconda3/envs/eegunify/bin/python -I scripts/questions_q01_q03_eklund.py  (from Start_Vision/)
Input: results/eklund2018/trial_scores.tsv.gz (P(seen) per trial, already computed; nothing refitted).
Output: results/questions_to_solve/q01_q03_eklund/{per_session.tsv, group.tsv, README.md}

Trials: critical near-threshold Gabors only (classes aware, unaware, unaware_incorrect,
aware_incorrect = condition 1-4), duration exclusion applied. Controls (high contrast) and catch
trials are left out. obj_corr: 1 correct localization, 0 wrong (3 = catch, not used).
Unit of inference: session (= subject within a Gabor size). Large Gabor primary, small replication.

Q01 graded: per-session Spearman rho(rating 1/2/3, P(seen)) on correct-localization trials; also
    mean P(seen) per rating and the step differences 2-1 and 3-2 (sessions with >= MIN trials per level).
Q02 identification: rating >= 2 trials, correct minus wrong localization (sessions with >= MIN wrong).
Q03 unconscious: rating 1 trials, correct minus wrong localization; compared with the awareness
    effect (aware correct minus unaware correct) in the same sessions.
Group test: two-sided sign-flip permutation (10 000) on the per-session values, bootstrap 95 % CI
(10 000), d_z. All thresholds below were fixed before looking at group statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN = 3  # minimum trials per cell for a session to enter a contrast
N_PERM = 10_000
SEED = 20261109
ROOT = Path.cwd()
OUT = ROOT / "results" / "questions_to_solve" / "q01_q03_eklund"
CRIT = {"aware", "unaware", "unaware_incorrect", "aware_incorrect"}


def group_test(x: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    x = x[np.isfinite(x)]
    n = len(x)
    m = x.mean()
    sd = x.std(ddof=1)
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, n))
    null = (flips * x).mean(1)
    p = (np.sum(np.abs(null) >= abs(m)) + 1) / (N_PERM + 1)
    boot = rng.choice(x, size=(N_PERM, n), replace=True).mean(1)
    return {
        "n": n,
        "mean": m,
        "ci_low": np.percentile(boot, 2.5),
        "ci_high": np.percentile(boot, 97.5),
        "d_z": m / sd if sd > 0 else np.nan,
        "p_two_sided": p,
        "n_positive": int((x > 0).sum()),
    }


def per_session(g: pd.DataFrame) -> dict[str, float]:
    r: dict[str, float] = {}
    c = g[g.obj_corr == 1]
    for k in (1, 2, 3):
        s = c.loc[c.rating == k, "p_seen"]
        r[f"n_r{k}_correct"] = len(s)
        r[f"p_r{k}_correct"] = s.mean() if len(s) >= MIN else np.nan
    r["q01_rho"] = spearmanr(c.rating, c.p_seen)[0] if c.rating.nunique() > 1 else np.nan
    r["q01_step_2_minus_1"] = r["p_r2_correct"] - r["p_r1_correct"]
    r["q01_step_3_minus_2"] = r["p_r3_correct"] - r["p_r2_correct"]
    aw = g[g.rating >= 2]
    a1, a0 = aw.loc[aw.obj_corr == 1, "p_seen"], aw.loc[aw.obj_corr == 0, "p_seen"]
    r["n_aware_correct"], r["n_aware_wrong"] = len(a1), len(a0)
    r["p_aware_wrong"] = a0.mean() if len(a0) >= MIN else np.nan
    r["q02_correct_minus_wrong"] = (
        a1.mean() - a0.mean() if len(a0) >= MIN and len(a1) >= MIN else np.nan
    )
    un = g[g.rating == 1]
    u1, u0 = un.loc[un.obj_corr == 1, "p_seen"], un.loc[un.obj_corr == 0, "p_seen"]
    r["n_unaware_correct"], r["n_unaware_wrong"] = len(u1), len(u0)
    r["p_unaware_wrong"] = u0.mean() if len(u0) >= MIN else np.nan
    r["q03_correct_minus_wrong"] = (
        u1.mean() - u0.mean() if len(u0) >= MIN and len(u1) >= MIN else np.nan
    )
    r["awareness_effect"] = a1.mean() - u1.mean() if len(a1) >= MIN and len(u1) >= MIN else np.nan
    r["q02_wrong_minus_unaware_correct"] = r["p_aware_wrong"] - (
        u1.mean() if len(u1) >= MIN else np.nan
    )
    return r


def main() -> None:
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(ROOT / "results" / "eklund2018" / "trial_scores.tsv.gz", sep="\t")
    d = d[d.cls.isin(CRIT) & d.exclusion.isna()]
    rows = [{"gabor": gb, "fp": fp, **per_session(g)} for (gb, fp), g in d.groupby(["gabor", "fp"])]
    ps = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    ps.to_csv(OUT / "per_session.tsv", sep="\t", index=False, float_format="%.4f")
    measures = [
        "q01_rho",
        "q01_step_2_minus_1",
        "q01_step_3_minus_2",
        "q02_correct_minus_wrong",
        "q02_wrong_minus_unaware_correct",
        "q03_correct_minus_wrong",
        "awareness_effect",
    ]
    out = []
    for gb in ("large", "small"):
        s = ps[ps.gabor == gb]
        for m in measures:
            out.append({"gabor": gb, "measure": m, **group_test(s[m].to_numpy(float), rng)})
        for col in (
            "p_r1_correct",
            "p_r2_correct",
            "p_r3_correct",
            "p_aware_wrong",
            "p_unaware_wrong",
        ):
            v = s[col].dropna()
            out.append({"gabor": gb, "measure": f"mean_{col}", "n": len(v), "mean": v.mean()})
        # Q03 relative size, sessions with both values
        both = s.dropna(subset=["q03_correct_minus_wrong", "awareness_effect"])
        ratio = both.q03_correct_minus_wrong.mean() / both.awareness_effect.mean()
        out.append(
            {"gabor": gb, "measure": "q03_over_awareness_effect", "n": len(both), "mean": ratio}
        )
    grp = pd.DataFrame(out)
    grp.to_csv(OUT / "group.tsv", sep="\t", index=False, float_format="%.4f")
    print(grp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
