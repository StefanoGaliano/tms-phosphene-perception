"""Synthetic inputs with the real column layout (blueprint §11) for the Questions_to_solve tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def eklund_table(n_sessions: int = 16, n_trials: int = 300, gabors: tuple[str, ...] = ("large",),
                 rating_effect: float = 0.1, duration_effect: float = 0.0,
                 duration_follows_rating: bool = False, correct_effect: float = 0.0,
                 seed: int = 0) -> pd.DataFrame:
    """Rows as ``questions.io.eklund_trials`` returns them."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in gabors:
        for fp in range(1, n_sessions + 1):
            rating = rng.choice([1, 2, 3], n_trials, p=[0.5, 0.35, 0.15])
            dur = rng.normal(40, 5, n_trials) + (8 * (rating - 2) if duration_follows_rating else 0)
            corr = np.where(rating == 1, rng.random(n_trials) < 0.5,
                            rng.random(n_trials) < 0.85).astype(int)
            p = (0.5 + rating_effect * (rating - 1) + duration_effect * (dur - 40) / 5
                 + correct_effect * corr * (rating >= 2) + rng.normal(0, 0.08, n_trials))
            cls = np.where(rating >= 2, np.where(corr == 1, "aware", "aware_incorrect"),
                           np.where(corr == 1, "unaware", "unaware_incorrect"))
            rows.append(pd.DataFrame({"gabor": g, "fp": fp, "cls": cls, "condition": 1,
                                      "rating": rating, "obj_corr": corr, "duration_ms": dur,
                                      "p_seen": p}))
    return pd.concat(rows, ignore_index=True)


def ds006171_table(n_subjects: int = 20, n_trials: int = 300, contrast_auc_only: bool = False,
                   task_drop: float = 0.0, contrast_gain: float = 15.0,
                   seed: int = 0) -> pd.DataFrame:
    """Rows as ``questions.io.ds006171_trials`` returns them (three tasks). ``contrast_gain`` sets
    how strongly the score follows contrast."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for task, shift in (("nocue", 0.0), ("noninformative", 0.0), ("informative", -0.01)):
            c = rng.uniform(0.01, 0.05, n_trials) + shift
            seen = rng.random(n_trials) < 1 / (1 + np.exp(-(c - 0.03) * 150))
            gain = 0.0 if contrast_auc_only else (1.0 - (task_drop if task == "informative" else 0))
            score = contrast_gain * (c - 0.03) + gain * seen + rng.normal(0, 1, n_trials)
            correct = np.where(rng.random(n_trials) < 0.15,
                               (rng.random(n_trials) < np.where(seen, 0.85, 0.5)).astype(float),
                               np.nan)
            side = "sideunknown" if task == "noninformative" else None
            rows.append(pd.DataFrame({
                "subject": f"{s:02d}", "task": task, "trial_number": np.arange(n_trials) + 1.0,
                "seen": seen, "score": score, "gabor_contrast": c, "correct": correct,
                "side": side or rng.choice(["left", "right"], n_trials)}))
    return pd.concat(rows, ignore_index=True)


def gin_scores_table(n_subjects: int = 12, n_pulses: int = 300, auc_shift: float = 0.3,
                     correlated: bool = False, seed: int = 0) -> pd.DataFrame:
    """Rows as ``questions.io.gin_scores`` returns them (two sites per subject)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for site in ("O1", "O2"):
            seen = (rng.random(n_pulses) < 0.45).astype(int)
            base = rng.normal(0, 1, n_pulses)
            if correlated:  # slow drift shared by neighbouring pulses within each class
                for v in (0, 1):
                    idx = np.flatnonzero(seen == v)
                    drift = np.repeat(rng.normal(0, 1.5, len(idx) // 20 + 1), 20)[:len(idx)]
                    base[idx] += drift[rng.permutation(len(idx))]
            p = 1 / (1 + np.exp(-(base + auc_shift * seen)))
            rows.append(pd.DataFrame({"recording": f"sub-{s:02d}_site-{site}",
                                      "subject": f"{s:02d}", "site": site,
                                      "pulse": np.arange(n_pulses), "seen": seen, "p_seen": p}))
    return pd.concat(rows, ignore_index=True)
