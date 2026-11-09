"""Q04 attention, contrast-matched (P§5): per-task LOSO AUC after matching, equivalence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from start_vision import codes
from start_vision.decoding import stats as dstats
from start_vision.matching import balance_report, match
from start_vision.questions import figures as fg
from start_vision.questions import io
from start_vision.questions.params import MATCH_BINS, MATCH_SEED, MIN_CELL, TOST_BOUND
from start_vision.questions.report import group_row, write_readme
from start_vision.questions.stats import one_sample

TASKS = ("nocue", "noninformative", "informative")


def subject_task_auc(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, t), g in trials.groupby(["subject", "task"], sort=True):
        m_in = pd.DataFrame({"cls": np.where(g["seen"], codes.SEEN, codes.UNSEEN),
                             "side": g["side"].to_numpy(),
                             "gabor_contrast": g["gabor_contrast"].to_numpy(float)},
                            index=g.index)
        keep = match(m_in, n_bins=MATCH_BINS, seed=MATCH_SEED)
        y, sc = g["seen"].to_numpy(), g["score"].to_numpy()
        k = keep.to_numpy()
        r: dict[str, Any] = {"subject": s, "task": t, "n": len(g), "n_matched": int(k.sum()),
                             "mean_contrast": float(g["gabor_contrast"].mean())}
        r["auc_unmatched"] = roc_auc_score(y, sc) if 0 < y.sum() < len(y) else np.nan
        ym = y[k]
        ok = min(int(ym.sum()), int((~ym).sum())) >= MIN_CELL
        r["auc_matched"] = roc_auc_score(ym, sc[k]) if ok else np.nan
        r |= {f"balance_{a}": b for a, b in balance_report(m_in, keep).items()}
        rows.append(r)
    return pd.DataFrame(rows)


def group(per: pd.DataFrame) -> dict[str, Any]:
    w = per.pivot_table(index="subject", columns="task", values="auc_matched")
    c = per.pivot_table(index="subject", columns="task", values="mean_contrast")
    out: dict[str, Any] = {}
    for t in ("informative", "noninformative"):
        if t not in w or "nocue" not in w:
            continue
        diff = (w[t] - w["nocue"]).dropna().to_numpy()
        out[f"{t}_diff"] = one_sample(diff, 0.0, tail=2)
        out[f"{t}_tost"] = dstats.tost(diff, TOST_BOUND)
        out[f"{t}_contrast_diff"] = one_sample((c[t] - c["nocue"]).dropna(), 0.0, tail=2)
    out["primary_tost"] = out.get("informative_tost", {"p": np.nan, "n": 0})
    for t in TASKS:
        if t in w:
            out[f"auc_{t}"] = one_sample(w[t], 0.5, tail=2)
    return out


def run(root: Path, force: bool = False) -> Path:
    d = io.out_dir(root, "q04", force)
    per = subject_task_auc(io.ds006171_trials(root))
    per.to_csv(d / "per_subject_task.tsv", sep="\t", index=False, float_format="%.5f")
    r = group(per)
    tp = r["primary_tost"]
    rows = [{"test": f"informative - nocue: TOST ±{TOST_BOUND} (equivalence p)", "role": "[P]",
             "n": tp["n"], "mean": tp["mean"], "p": tp["p"]},
            {"test": f"noninformative - nocue: TOST ±{TOST_BOUND}", "role": "secondary",
             "n": r["noninformative_tost"]["n"], "mean": r["noninformative_tost"]["mean"],
             "p": r["noninformative_tost"]["p"]},
            group_row("informative - nocue (difference test)", "secondary", r["informative_diff"]),
            group_row("noninformative - nocue (difference test)", "secondary",
                      r["noninformative_diff"]),
            group_row("informative - nocue: mean contrast", "secondary",
                      r["informative_contrast_diff"]),
            group_row("noninformative - nocue: mean contrast", "secondary",
                      r["noninformative_contrast_diff"]),
            *[group_row(f"matched AUC {t} vs 0.5", "descriptive", r[f"auc_{t}"]) for t in TASKS]]
    tab = pd.DataFrame(rows)
    tab.to_csv(d / "group.tsv", sep="\t", index=False, float_format="%.5f")
    io.write_primary(d, "q04", "informative - nocue matched AUC: TOST equivalence ±0.03",
                     "equivalence", {"n": float(tp["n"]), "mean": float(tp["mean"]),
                                     "p": float(tp["p"])})
    w = per.pivot_table(index="subject", columns="task", values="auc_matched")
    fig, (ax,) = fg.axes(1, 4.4, 3.2)
    ax.axhspan(-TOST_BOUND, TOST_BOUND, color=fg.BAND, label=f"±{TOST_BOUND} equivalence band")
    for i, t in enumerate(("informative", "noninformative")):
        v = (w[t] - w["nocue"]).dropna()
        ax.scatter(np.full(len(v), i), v, color=fg.SLOTS[i], s=18,
                   label=f"{t} - nocue (N {len(v)})")
    ax.axhline(0, color=fg.GRID, lw=1)
    ax.set_xticks([0, 1], ["informative", "noninformative"])
    fg.style(ax, "cue task minus no cue", "matched AUC difference")
    ax.legend(fontsize=7, frameon=False)
    fg.save(fig, d)
    write_readme(d, root, "Q04 attention, contrast-matched (ds006171 detector)", "§5",
                 ["loso_predictions.tsv + staged events (34 subjects)"], {"Group tests": tab},
                 ("equivalence shown: the informative cue does not change the matched AUC beyond "
                  f"±{TOST_BOUND}." if tp["p"] < 0.05 else
                  f"equivalence within ±{TOST_BOUND} not shown for the informative cue."),
                 ["Deferred: the VAN part (needs the ds006171 epochs, runs/).",
                  "Matching bins contrast within hemifield (Phase 1 rule); noninformative trials "
                  "form one 'side unknown' group (answer 6)."])
    return d
