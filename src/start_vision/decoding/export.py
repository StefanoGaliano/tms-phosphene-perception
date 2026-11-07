"""A11: choose, fit and export the frozen detectors (PHASE2 §9, D9; build step 15).

Variants (D9):
- ``full``: 128 cap channels.
- ``tms-robust``: the 64-channel 10-10 subset (``configs/channels/ds006171_to_biosemi64.tsv``),
  feature windows starting >= ``t_min_robust_s``.

Window (``candidate_windows_s``) and C (``C_grid``) are chosen by LOSO mean AUC over all
subjects; the unbiased estimate of that procedure is a **nested** LOSO (the outer subject never
takes part in the choice). Every model trained without subjects {i, j} serves two inner folds
(outer i / inner j and outer j / inner i), and the outer score of subject i under a chosen
config is its plain LOSO score with that config, so the nested estimate costs one pass over
subject pairs. Features are computed per subject once (label-free normalization, stateless
window means), so nothing is fitted outside a training fold.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from start_vision.decoding import channels, models, stats
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from start_vision.decoding.detector import DETECTOR_BASELINE_S, Detector, prepare_subject

VERSION = "1.0.0"
DATASET = {"dataset": "OpenNeuro ds006171", "doi": "10.18112/openneuro.ds006171.v1.0.0"}

Config = tuple[tuple[float, float], float]


def _features(sets: Sequence[TrialSet], catch: Sequence[TrialSet | None], cfg: DecodingConfig,
              window: tuple[float, float]) -> list[np.ndarray]:
    """Per subject window features of its main trials (normalized over all its trials)."""
    feat = models.WindowFeatures(sets[0].times, window, cfg.feature_bin_s)
    out = []
    for s, c in zip(sets, catch, strict=True):
        x = s.X if c is None or not len(c) else np.concatenate([s.X, c.X])
        f = feat.transform(prepare_subject(x, s.times, DETECTOR_BASELINE_S, cfg.norm_window_s))
        out.append(f[:len(s)])
    return out


def _fit_score(cfg: DecodingConfig, C: float, fs: list[np.ndarray], ys: list[np.ndarray],
               train: Sequence[int], test: Sequence[int]) -> dict[int, float]:
    pipe = models.window_detector(cfg, np.zeros(1), (0.0, 0.0), C)
    pipe.steps[0] = (pipe.steps[0][0], "passthrough")
    pipe.fit(np.concatenate([fs[k] for k in train]), np.concatenate([ys[k] for k in train]))
    return {k: float(roc_auc_score(ys[k], pipe.decision_function(fs[k]))) for k in test}


def nested_selection(sets: Sequence[TrialSet], catch: Sequence[TrialSet | None],
                     cfg: DecodingConfig, windows: Sequence[tuple[float, float]],
                     backend: str = "sklearn") -> dict[str, Any]:
    """LOSO choice of (window, C) on all subjects, and its nested-LOSO held-out AUCs.

    ``backend="torch"``: the same fits on the GPU (``gpu_select``), run one after another.
    """
    from joblib import Parallel, delayed

    n = len(sets)
    ys = [s.labels("seen") for s in sets]
    configs: list[Config] = [(w, C) for w in windows for C in cfg.C_grid]
    feats = {w: _features(sets, catch, cfg, w) for w in windows}
    everyone = range(n)
    caches: dict[tuple[float, float], dict[str, Any]] = {w: {} for w in windows}

    def fit_one(c: float, w: tuple[float, float], train: list[int], test: list[int]
              ) -> dict[int, float]:
        if backend == "torch":
            from start_vision.decoding import gpu_select
            return gpu_select.fit_score(c, feats[w], ys, train, test, caches[w])
        return _fit_score(cfg, c, feats[w], ys, train, test)

    def loso(cf: Config, i: int) -> tuple[Config, int, float]:
        w, c = cf
        return cf, i, fit_one(c, w, [k for k in everyone if k != i], [i])[i]

    def pair(cf: Config, i: int, j: int) -> tuple[Config, int, int, dict[int, float]]:
        w, c = cf
        return cf, i, j, fit_one(c, w, [k for k in everyone if k not in (i, j)], [i, j])

    par = Parallel(n_jobs=1 if backend == "torch" else cfg.n_jobs, prefer="threads")
    loso_auc = {cf: np.full(n, np.nan) for cf in configs}
    n_fits = len(configs) * (n + n * (n - 1) // 2)
    print(f"selection: {len(configs)} configs x {n} subjects = {n_fits} fits ({backend})",
          flush=True)
    for cf, i, a in par(delayed(loso)(cf, i) for cf in configs for i in everyone):
        loso_auc[cf][i] = a
    # inner[cf][i, j] = AUC of subject j, model trained without {i, j}
    inner = {cf: np.full((n, n), np.nan) for cf in configs}
    for cf, i, j, r in par(delayed(pair)(cf, i, j) for cf in configs
                           for i, j in itertools.combinations(everyone, 2)):
        inner[cf][i, j], inner[cf][j, i] = r[j], r[i]
    print("selection: done", flush=True)
    chosen_outer, nested = [], np.full(n, np.nan)
    for i in everyone:
        score = {cf: np.nanmean(inner[cf][i]) for cf in configs}
        best = max(configs, key=lambda cf: (score[cf], -configs.index(cf)))
        chosen_outer.append(best)
        nested[i] = loso_auc[best][i]
    mean_loso = {cf: float(np.mean(loso_auc[cf])) for cf in configs}
    best = max(configs, key=lambda cf: (mean_loso[cf], -configs.index(cf)))
    return {
        "window_s": best[0], "C": best[1],
        "grid": [{"window_s": list(w), "C": C, "loso_auc_mean": mean_loso[(w, C)]}
                 for w, C in configs],
        "nested_auc": nested,
        "nested_choices": [{"window_s": list(w), "C": C} for w, C in chosen_outer],
    }


def _variant_sets(sets: Sequence[TrialSet], catch: Sequence[TrialSet | None], variant: str,
                  root: Path, electrodes_tsv: Path) -> tuple[list[TrialSet],
                                                            list[TrialSet | None],
                                                            pd.DataFrame]:
    cap_pos, _ = channels.cap_positions(electrodes_tsv)
    if variant == "full":
        names = list(sets[0].ch_names)
        xyz = np.array([cap_pos[n] for n in names])
        table = pd.DataFrame({"target": names, "cap": names, "distance_mm": 0.0,
                              "exact": True, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
        return list(sets), list(catch), table
    table = pd.read_csv(root / "configs" / "channels" / "ds006171_to_biosemi64.tsv", sep="\t")
    return ([channels.subset(s, table) for s in sets],
            [channels.subset(c, table) if c is not None and len(c) else None for c in catch],
            table)


def export(sets: Sequence[TrialSet], catch: Sequence[TrialSet | None], cfg: DecodingConfig,
           variant: str, root: Path, electrodes_tsv: Path, out_dir: Path, fig_dir: Path,
           stage_key: str, tasks: Sequence[str], backend: str = "sklearn") -> Path:
    from start_vision.decoding.figures import f10_calibration

    vs, vc, table = _variant_sets(sets, catch, variant, root, electrodes_tsv)
    windows = [w for w in cfg.candidate_windows_s
               if variant != "tms-robust" or w[0] >= cfg.t_min_robust_s - 1e-9]
    sel = nested_selection(vs, vc, cfg, windows, backend)
    subjects = [str(s.meta["subject"].iloc[0]) for s in vs]
    det = Detector.fit(vs, cfg, tuple(sel["window_s"]), variant=variant, version=VERSION,
                       C=sel["C"], catch=vc, channel_table=table,
                       trained_on=DATASET | {"subjects": subjects, "tasks": list(tasks),
                                             "trials": "stimulus-present (main); catch "
                                                       "trials only in normalization"})
    nested = sel["nested_auc"]
    lo, hi = stats.bootstrap_ci(nested, cfg.n_bootstrap, cfg.seed)
    lo2, hi2 = stats.bootstrap_ci(np.array([r["auc"] for r in
                                            det.performance["loso_auc_per_subject"]]),
                                  cfg.n_bootstrap, cfg.seed)
    det.performance |= {
        "loso_auc_ci95": [lo2, hi2],
        "selection": {"rule": "max LOSO mean AUC over windows x C (ties: first)",
                      "backend": backend,
                      "window_s": list(sel["window_s"]), "C": sel["C"], "grid": sel["grid"]},
        "nested_loso_auc_per_subject": dict(zip(subjects, map(float, nested), strict=True)),
        "nested_loso_auc_mean": float(nested.mean()), "nested_loso_auc_ci95": [lo, hi],
        "nested_choices": sel["nested_choices"],
    }
    det.contract["stage_key"] = stage_key
    folder = det.save(out_dir / f"sv-perception-{variant}-v{VERSION}")
    f10_calibration(det.calibration, det.held_out, fig_dir, name=f"F10_{variant}",
                    title=f": {variant}")
    held: dict[str, Any] = {k.replace(" ", "_"): v for k, v in det.held_out.items()}
    np.savez(fig_dir.parent / f"detector_{variant}_held_out.npz", **held)
    print(f"export: {variant} window {sel['window_s']} C {sel['C']} "
          f"LOSO {det.performance['loso_auc_mean']:.3f} nested {nested.mean():.3f}",
          flush=True)
    (fig_dir.parent / f"detector_{variant}_summary.json").write_text(json.dumps(
        {k: det.performance[k] for k in ("loso_auc_mean", "loso_auc_ci95",
                                         "nested_loso_auc_mean", "nested_loso_auc_ci95",
                                         "threshold", "balanced_accuracy_at_threshold",
                                         "brier", "selection")}
        | {"catch": det.performance.get("catch")}, indent=2) + "\n")
    return folder


def run(project: Any, subjects: Sequence[str], cfg: DecodingConfig,
        variants: Sequence[str] = ("tms-robust", "full"), backend: str = "sklearn"
        ) -> list[Path]:
    """Load every included subject's main and catch trials, export each variant."""
    from eegunify.erp.hashing import stage_key
    from start_vision.decoding.analyses import SubjectData
    from start_vision.decoding.stages import _hash_config, subject_upstream
    from start_vision.decoding.step0 import included_tasks

    inc = included_tasks(project.root)
    sets, catch = [], []
    up: dict[str, str] = {}
    for s in subjects:
        d = SubjectData(project.runs, project.staged, s, inc[s], cfg)
        sets.append(d.main)
        catch.append(d.catch if len(d.catch) else None)
        up |= {f"sub-{s}:{k}": v for k, v in subject_upstream(project, s, cfg).items()}
        del d
    tasks = sorted({t for s in subjects for t in inc[s]})
    electrodes = (project.root / "sourcedata" / "ds006171" / "sub-01" / "eeg"
                  / "sub-01_electrodes.tsv")
    out = []
    for v in variants:
        key = stage_key(stage=f"sv_detector_{v}", schema_version=1,
                        config=_hash_config(cfg, []) | {"variant": v, "version": VERSION},
                        upstream=up)
        out.append(export(sets, catch, cfg, v, project.root, electrodes,
                          project.root / "detectors", project.root / "results" / "phase2"
                          / "figures", key, tasks, backend))
    return out
