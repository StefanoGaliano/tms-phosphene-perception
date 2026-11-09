"""Q01 graded awareness (P§2): per-session OLS of P(seen) on rating with centred duration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from start_vision.questions import figures as fg
from start_vision.questions import io
from start_vision.questions.params import MIN_CELL
from start_vision.questions.report import group_row, write_readme
from start_vision.questions.stats import one_sample


def _ols_slope(y: np.ndarray, *cols: np.ndarray) -> float:
    x = np.column_stack([np.ones_like(y), *cols])
    return float(np.linalg.lstsq(x, y, rcond=None)[0][1])


def session_slopes(trials: pd.DataFrame) -> pd.DataFrame:
    """Correct-localization trials; a session enters with >= MIN_CELL trials at >= 2 levels."""
    rows = []
    for (g, fp), s in trials[trials["obj_corr"] == 1].groupby(["gabor", "fp"], sort=True):
        n = {k: int((s["rating"] == k).sum()) for k in (1, 2, 3)}
        r: dict[str, Any] = {"gabor": g, "fp": fp, **{f"n_r{k}": v for k, v in n.items()}}
        m = {k: (s.loc[s["rating"] == k, "p_seen"].mean() if n[k] >= MIN_CELL else np.nan)
             for k in (1, 2, 3)}
        r |= {f"mean_r{k}": m[k] for k in (1, 2, 3)}
        r["step_2_minus_1"], r["step_3_minus_2"] = m[2] - m[1], m[3] - m[2]
        r["eligible"] = sum(v >= MIN_CELL for v in n.values()) >= 2
        if r["eligible"]:
            y = s["p_seen"].to_numpy(float)
            rat = s["rating"].to_numpy(float)
            dur = s["duration_ms"].to_numpy(float)
            r["slope_cov"] = _ols_slope(y, rat, dur - dur.mean())
            r["slope_nocov"] = _ols_slope(y, rat)
        else:
            r["slope_cov"] = r["slope_nocov"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def group(per: pd.DataFrame, gabor: str = "large") -> dict[str, dict[str, float]]:
    s = per[per["gabor"] == gabor]
    return {"primary": one_sample(s["slope_cov"], 0.0, tail=1),
            "no_covariate": one_sample(s["slope_nocov"], 0.0, tail=1),
            "step_2_minus_1": one_sample(s["step_2_minus_1"], 0.0, tail=2),
            "step_3_minus_2": one_sample(s["step_3_minus_2"], 0.0, tail=2)}


def interpretation(r: dict[str, dict[str, float]]) -> str:
    if r["primary"]["p"] < 0.05:
        return ("graded read-out: the rating slope is positive with the duration covariate "
                "(P§2 rule).")
    if r["no_covariate"]["p"] < 0.05:
        return ("gradation attributed to duration: the slope is positive without the covariate "
                "but not with it (P§2 rule).")
    return "no graded read-out shown: the rating slope is not positive (P§2 rule)."


def run(root: Path, force: bool = False) -> Path:
    d = io.out_dir(root, "q01", force)
    trials = io.eklund_trials(root)
    per = session_slopes(trials)
    per.to_csv(d / "per_session.tsv", sep="\t", index=False, float_format="%.5f")
    res = {g: group(per, g) for g in ("large", "small")}
    rows = [group_row(f"{g}: {k}", "[P]" if (g == "large" and k == "primary") else
                      ("replication" if k == "primary" else "secondary"), v)
            for g, rr in res.items() for k, v in rr.items()]
    tab = pd.DataFrame(rows)
    tab.to_csv(d / "group.tsv", sep="\t", index=False, float_format="%.5f")
    io.write_primary(d, "q01", "large Gabors: rating slope with centred duration > 0", "one-sided",
                     res["large"]["primary"])
    fig, axs = fg.axes(2)
    for ax, g in zip(axs, ("large", "small"), strict=True):
        s = per[per["gabor"] == g]
        for row in s[["mean_r1", "mean_r2", "mean_r3"]].to_numpy():
            ax.plot([1, 2, 3], row, color=fg.GRID, lw=1)
        ax.plot([1, 2, 3], s[["mean_r1", "mean_r2", "mean_r3"]].mean().to_numpy(), "o-",
                color=fg.SLOTS[0], lw=2, label="mean over sessions")
        ax.set_xticks([1, 2, 3], ["did not see", "saw something", "saw clearly"], fontsize=7)
        fg.style(ax, "rating (correct localization)", "mean P(seen)", f"{g} Gabors (N {len(s)})")
        ax.legend(fontsize=7, frameon=False)
    fg.save(fig, d)
    write_readme(d, root, "Q01 graded awareness (Eklund & Wiens 2018)", "§2",
                 [f"results/eklund2018/trial_scores.tsv.gz ({len(trials)} critical trials)"],
                 {"Group tests": tab}, interpretation(res["large"]),
                 ["Rating 3 is sparse (sessions without >= 3 'saw clearly' trials have no "
                  "step 3-2).",
                  "Same data as the first look (P§0): specified before the rerun, not "
                  "confirmatory."])
    return d
