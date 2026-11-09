"""Questions_to_solve from saved results only (no epochs; runs/ was removed on purpose).

Run from Start_Vision/: ~/miniconda3/envs/eegunify/bin/python -I scripts/questions_saved_results.py
Output: results/questions_to_solve/saved_results/{q02_q03_ds006171.tsv, q04_attention.tsv,
        q17_size.tsv, q27_below_chance.tsv, group.tsv}

Q02/Q03 ds006171: held-out LOSO decision scores of the Phase 2 window detector
  (derivatives/sv_decode_group/loso_predictions.tsv; each subject scored by a model that never saw
  them) joined to the staged events (objective_correct, gabor_contrast). Trials with the
  orientation question only. Per subject, tasks pooled: AUC(orientation correct vs wrong) within
  seen trials (Q02) and within unseen trials (Q03). Sensitivity: the same on scores with contrast
  regressed out within subject x task (linear, label-free). Minimum 3 trials per cell.
Q04: Phase 1 VAN per task (results/phase1/per_subject.tsv) and per-task LOSO AUC (seen vs unseen):
  paired nocue vs informative (sign flip, two-sided) and TOST +-0.03 AUC (as Phase 2 A8).
Q17: Eklund frozen-detector AUC large vs small, sessions with both (results/eklund2018/session_aucs.tsv),
  paired difference, TOST +-0.03.
Q27: Phase 2 temporal generalization (a2_generalization.npz): clusters of AUC < 0.5, post-stimulus.
All settings fixed here before the results were computed. Exploratory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from start_vision.decoding import stats

ROOT = Path.cwd()
OUT = ROOT / "results" / "questions_to_solve" / "saved_results"
SEED = 20261109
MIN = 3
N = 10_000


def signflip(x: np.ndarray, rng: np.random.Generator, center: float = 0.0) -> dict[str, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] - center
    flips = rng.choice([-1.0, 1.0], size=(N, len(x)))
    null = (flips * x).mean(1)
    p = (np.sum(np.abs(null) >= abs(x.mean())) + 1) / (N + 1)
    boot = rng.choice(x, size=(N, len(x)), replace=True).mean(1)
    return {
        "n": len(x),
        "mean": x.mean() + center,
        "ci_low": np.percentile(boot, 2.5) + center,
        "ci_high": np.percentile(boot, 97.5) + center,
        "d_z": x.mean() / x.std(ddof=1),
        "p_two_sided": p,
        "n_positive": int((x > 0).sum()),
    }


def staged(sub: str, task: str) -> pd.DataFrame:
    f = (
        ROOT
        / "sourcedata"
        / "staged"
        / "ds006171"
        / f"sub-{sub}"
        / "eeg"
        / f"sub-{sub}_task-{task}_events.tsv"
    )
    e = pd.read_csv(f, sep="\t", dtype={"trial_number": float})
    e = e[e.trial_type.str.startswith("stimulus/")]
    return e[["trial_number", "objective_correct", "gabor_contrast"]]


def q02_q03(rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    p = pd.read_csv(
        ROOT / "derivatives" / "sv_decode_group" / "loso_predictions.tsv",
        sep="\t",
        dtype={"subject": str},
    )
    parts = []
    for (s, t), g in p.groupby(["subject", "task"]):
        m = g.merge(staged(s, t), on="trial_number", how="left", validate="one_to_one")
        c = m.gabor_contrast.astype(float)
        b = np.polyfit(c, m.score, 1) if c.nunique() > 1 else (0.0, m.score.mean())
        m["score_resid"] = m.score - np.polyval(b, c)
        parts.append(m)
    d = pd.concat(parts)
    d = d[d.objective_correct.astype(str).isin(["True", "False", "1", "0", "1.0", "0.0"])]
    d["correct"] = d.objective_correct.astype(str).isin(["True", "1", "1.0"])
    rows = []
    for s, g in d.groupby("subject"):
        r = {"subject": s}
        for lab, sel in (("seen", g.seen), ("unseen", ~g.seen)):
            h = g[sel]
            r[f"n_{lab}_correct"], r[f"n_{lab}_wrong"] = (
                int(h.correct.sum()),
                int((~h.correct).sum()),
            )
            r[f"acc_{lab}"] = h.correct.mean()
            ok = min(r[f"n_{lab}_correct"], r[f"n_{lab}_wrong"]) >= MIN
            for col in ("score", "score_resid"):
                r[f"auc_{lab}_{col}"] = roc_auc_score(h.correct, h[col]) if ok else np.nan
            r[f"contrast_diff_{lab}"] = (
                (
                    h.loc[h.correct, "gabor_contrast"].astype(float).mean()
                    - h.loc[~h.correct, "gabor_contrast"].astype(float).mean()
                )
                if ok
                else np.nan
            )
        rows.append(r)
    per = pd.DataFrame(rows)
    grp = []
    for lab, q in (("seen", "Q02"), ("unseen", "Q03")):
        for col in ("score", "score_resid"):
            grp.append(
                {
                    "question": q,
                    "measure": f"auc_{lab}_{col}",
                    **signflip(per[f"auc_{lab}_{col}"].to_numpy(), rng, 0.5),
                }
            )
        grp.append(
            {
                "question": q,
                "measure": f"orientation_acc_{lab}",
                **signflip(per[f"acc_{lab}"].to_numpy(), rng, 0.5),
            }
        )
        grp.append(
            {
                "question": q,
                "measure": f"contrast_diff_{lab}",
                **signflip(per[f"contrast_diff_{lab}"].to_numpy(), rng),
            }
        )
    return per, grp


def q04(rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    ps = pd.read_csv(
        ROOT / "results" / "phase1" / "per_subject.tsv", sep="\t", dtype={"subject": str}
    )
    v = ps[ps.component == "van"].pivot_table(
        index="subject", columns=["scope", "form"], values="diff"
    )
    p = pd.read_csv(
        ROOT / "derivatives" / "sv_decode_group" / "loso_predictions.tsv",
        sep="\t",
        dtype={"subject": str},
    )
    auc = (
        p.groupby(["subject", "task"])
        .apply(lambda g: roc_auc_score(g.seen, g.score) if g.seen.nunique() == 2 else np.nan)
        .unstack()
    )
    grp = []
    for name, a, b in (
        (
            "van_contralateral_informative_minus_nocue",
            v[("informative", "contralateral")],
            v[("nocue", "contralateral")],
        ),
        (
            "van_bilateral_informative_minus_nocue",
            v[("informative", "bilateral_matched")],
            v[("nocue", "bilateral_matched")],
        ),
        (
            "van_bilateral_noninformative_minus_nocue",
            v[("noninformative", "bilateral_matched")],
            v[("nocue", "bilateral_matched")],
        ),
        ("loso_auc_informative_minus_nocue", auc["informative"], auc["nocue"]),
        ("loso_auc_noninformative_minus_nocue", auc["noninformative"], auc["nocue"]),
    ):
        dlt = (a - b).dropna().to_numpy()
        row = {"question": "Q04", "measure": name, **signflip(dlt, rng)}
        if name.startswith("loso"):
            row["tost_p_0.03"] = stats.tost(dlt, 0.03)["p"]
        grp.append(row)
    for t in ("nocue", "noninformative", "informative"):
        grp.append(
            {
                "question": "Q04",
                "measure": f"loso_auc_{t}",
                **signflip(auc[t].dropna().to_numpy(), rng, 0.5),
            }
        )
    out = auc.add_prefix("loso_auc_").join(
        v.set_axis([f"van_{a}_{b}" for a, b in v.columns], axis=1)
    )
    return out.reset_index(), grp


def q17(rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    s = pd.read_csv(ROOT / "results" / "eklund2018" / "session_aucs.tsv", sep="\t")
    w = s.pivot_table(index="fp", columns="gabor", values="auc_primary").dropna()
    dlt = (w["large"] - w["small"]).to_numpy()
    return w.reset_index(), [
        {
            "question": "Q17",
            "measure": "auc_large_minus_small",
            **signflip(dlt, rng),
            "tost_p_0.03": stats.tost(dlt, 0.03)["p"],
        }
    ]


def q27() -> pd.DataFrame:
    d = np.load(ROOT / "derivatives" / "sv_decode_group" / "a2_generalization.npz")
    t = d["times"]
    post = t >= 0
    auc = d["auc"][:, post][:, :, post]
    cl = stats.cluster_auc(1.0 - auc, n_permutations=N, seed=SEED)  # clusters of AUC < 0.5
    cl["mean_auc"] = 1.0 - cl["mean_auc"]
    gm = auc.mean(0)
    i, j = np.unravel_index(np.argmin(gm), gm.shape)
    cl["group_min_auc"], cl["min_train_s"], cl["min_test_s"] = gm[i, j], t[post][i], t[post][j]
    return cl


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    per23, g23 = q02_q03(rng)
    per23.to_csv(OUT / "q02_q03_ds006171.tsv", sep="\t", index=False, float_format="%.4f")
    per4, g4 = q04(rng)
    per4.to_csv(OUT / "q04_attention.tsv", sep="\t", index=False, float_format="%.4f")
    per17, g17 = q17(rng)
    per17.to_csv(OUT / "q17_size.tsv", sep="\t", index=False, float_format="%.4f")
    c27 = q27()
    c27.to_csv(OUT / "q27_below_chance.tsv", sep="\t", index=False, float_format="%.4f")
    grp = pd.DataFrame(g23 + g4 + g17)
    grp.to_csv(OUT / "group.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(grp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(c27.head().to_string(index=False))


if __name__ == "__main__":
    main()
