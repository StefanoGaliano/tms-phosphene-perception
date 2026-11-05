"""Phase 2 group level (PHASE2 §7-8, §10; build step 14). Called by ``sv_decode_group``.

Subject-level arrays come from ``derivatives/sv_decode``; across-subject analyses (A7, A8 LOSO,
A10) load the included subjects' trials once. Windows for window-level summaries, fixed before
group results: VAN 180-280 ms and LP 350-550 ms (docs/PHASE1_REFERENCES.md §2) and the
post-stimulus window ``analyses.WIN_S`` (200-600 ms). A10 windows start at {0, 100, 150, 200} ms
and end at 600 ms (the end of ``WIN_S``).

Outputs: tables and figures in ``results/phase2/`` (committed); arrays and LOSO predictions in
``derivatives/sv_decode_group/`` (not committed).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy import stats as sps

from start_vision.decoding import across, channels, figures, stats
from start_vision.decoding.analyses import WIN_S, SubjectData
from start_vision.decoding.config import DecodingConfig

WINDOWS = {"VAN 180-280": (0.18, 0.28), "LP 350-550": (0.35, 0.55), "post 200-600": WIN_S}
A10_STARTS = (0.0, 0.1, 0.15, 0.2)
A10_END = WIN_S[1]
A10_CHANNELS = (128, 64, 32)

SERIES = {  # family -> list of (analysis file, key, label)
    "A1 within-subject": [("a1_sliding", "pooled", "pooled"), ("a1_sliding", "nocue", "nocue"),
                          ("a1_sliding", "noninformative", "noninformative"),
                          ("a1_sliding", "informative", "informative")],
    "A4/A5 contrast": [("a4_within_bin", "pooled", "within contrast bins"),
                       ("a5_contrast", "residualized", "contrast regressed out"),
                       ("a5_contrast", "within_seen", "contrast | seen"),
                       ("a5_contrast", "within_unseen", "contrast | unseen")],
    "C3 eye movements": [("c3_eog", "eog_only", "EOG channels only"),
                         ("c3_eog", "no_ica", "without ICA"),
                         ("c3_eog", "heog_clean", "HEOG-step trials removed")],
    "A9 cross-condition": [("a9_cross_condition", "hemifield", "cross-hemifield"),
                           ("a9_cross_condition", "hemifield_mirrored",
                            "cross-hemifield, mirrored"),
                           ("a9_cross_condition", "orientation", "cross-orientation")],
}


def _load(project: Any, subjects: Sequence[str], name: str) -> dict[str, Any]:
    return {s: np.load(project.sub_dir(s) / f"sub-{s}_desc-{name}_scores.npz")
            for s in subjects
            if (project.sub_dir(s) / f"sub-{s}_desc-{name}_scores.npz").exists()}


def _stack(d: dict[str, Any], key: str) -> tuple[np.ndarray, list[str]]:
    subs = [s for s, z in d.items() if key in z.files]
    return (np.array([d[s][key] for s in subs]) if subs else np.zeros((0, 0))), subs


def _win(auc: np.ndarray, times: np.ndarray, w: tuple[float, float]) -> np.ndarray:
    return np.asarray(auc[:, (times >= w[0] - 1e-9) & (times < w[1] - 1e-9)].mean(1))


def _row(family: str, test: str, r: dict[str, float], **kw: Any) -> dict[str, Any]:
    return {"family": family, "test": test, **{k: r.get(k) for k in
            ("n", "mean", "t", "p", "d_z", "ci_low", "ci_high")}, **kw}


def _a8_rows(a8m: np.ndarray, cfg: DecodingConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diag = np.array([np.nanmean(np.diag(m)) for m in a8m])
    off = np.array([np.nanmean(m[~np.eye(3, dtype=bool)]) for m in a8m])
    rows.append(_row("A8 cross-task", "off-diagonal window AUC > 0.5",
                     stats.window_test(off, cfg.n_permutations, cfg.n_bootstrap, cfg.seed)))
    rows.append(_row("A8 cross-task", "diagonal minus off-diagonal",
                     stats.paired_test(diag, off, cfg.n_permutations, cfg.n_bootstrap,
                                       cfg.seed)))
    tost = stats.tost(diag - off, cfg.tost_bound_auc)
    rows.append({"family": "A8 cross-task",
                 "test": f"TOST diag - off within ±{cfg.tost_bound_auc}",
                 "n": tost["n"], "mean": tost["mean"], "p": tost["p"],
                 "note": f"p_low {tost['p_low']:.3g}, p_high {tost['p_high']:.3g}"})
    return rows


def run(project: Any, subjects: Sequence[str], cfg: DecodingConfig) -> list[Path]:
    res_dir = project.root / "results" / "phase2"
    fig_dir = res_dir / "figures"
    arr_dir = project.group_dir()
    for d in (res_dir, fig_dir, arr_dir):
        d.mkdir(parents=True, exist_ok=True)
    kw: dict[str, Any] = {"n_permutations": cfg.n_permutations, "p": cfg.cluster_p,
                          "seed": cfg.seed}
    rows: list[dict[str, Any]] = []
    clusters: list[pd.DataFrame] = []
    a1 = _load(project, subjects, "a1_sliding")
    times = next(iter(a1.values()))["times"]

    # ---- subject-level series: clusters + window tests ----
    curves: dict[str, dict[str, np.ndarray]] = {}
    masks: dict[str, dict[str, np.ndarray]] = {}
    for family, items in SERIES.items():
        curves[family], masks[family] = {}, {}
        for name, key, label in items:
            auc, subs = _stack(_load(project, subjects, name), key)
            if len(subs) < 2:
                continue
            np.savez(arr_dir / f"{name}_{key}.npz", auc=auc, times=times,
                     subjects=np.array(subs))
            cl = stats.cluster_auc(auc, times, **kw)
            clusters.append(cl.assign(family=family, series=label, n=len(subs)))
            curves[family][label] = auc
            masks[family][label] = stats.cluster_mask(auc, **kw)
            for wname, w in WINDOWS.items():
                rows.append(_row(family, f"{label}: {wname}",
                                 stats.window_test(_win(auc, times, w), cfg.n_permutations,
                                                   cfg.n_bootstrap, cfg.seed)))
            rows.append(_row("prestimulus check", f"{label}: -500..0 ms",
                             stats.window_test(_win(auc, times, (-0.5, 0.0)),
                                               cfg.n_permutations, cfg.n_bootstrap, cfg.seed)))
    pooled = curves["A1 within-subject"]["pooled"]
    tfce_p = stats.tfce_auc(pooled, cfg.n_permutations, cfg.seed)
    pd.DataFrame({"time_s": times, "p_tfce": tfce_p,
                  "mean_auc": pooled.mean(0)}).to_csv(res_dir / "a1_tfce.tsv", sep="\t",
                                                      index=False)
    # paired: does a control lower the post-window AUC?
    a1_post = dict(zip(_stack(a1, "pooled")[1], _win(pooled, times, WIN_S), strict=True))
    for name, key, label in (("a4_within_bin", "pooled", "within contrast bins"),
                             ("a5_contrast", "residualized", "contrast regressed out"),
                             ("c3_eog", "no_ica", "without ICA"),
                             ("c3_eog", "heog_clean", "HEOG-step trials removed")):
        auc, subs = _stack(_load(project, subjects, name), key)
        if len(subs) >= 2:
            ctl = _win(auc, times, WIN_S)
            ref = np.array([a1_post[s] for s in subs])
            rows.append(_row("controls vs A1 (post 200-600)", f"A1 minus {label}",
                             stats.paired_test(ref, ctl, cfg.n_permutations, cfg.n_bootstrap,
                                               cfg.seed)))

    # ---- A2: 2D cluster ----
    a2, subs2 = _stack(_load(project, subjects, "a2_generalization"), "pooled")
    a2_mask = None
    if len(subs2) >= 2:
        a2_mask = stats.cluster_mask(a2, **kw)
        stats.cluster_auc(a2, None, **kw).assign(
            family="A2 generalization", series="pooled", n=len(subs2)).to_csv(
            res_dir / "a2_clusters.tsv", sep="\t", index=False)
        np.savez(arr_dir / "a2_generalization.npz", auc=a2, mask=a2_mask, times=times)

    # ---- C4 alpha, C5 balance + catch ----
    c4 = _load(project, subjects, "c4_alpha")
    c5 = _load(project, subjects, "c5_balance_catch")
    c4t = pd.DataFrame()
    catch_note = "C5 not run"
    if c4 and c5:
        c4t = pd.DataFrame([{"subject": s, **{k: float(z[k]) for k in z.files if k != "times"}}
                            for s, z in c4.items()])
        c4t.to_csv(res_dir / "c4_alpha_per_subject.tsv", sep="\t", index=False)
        for col in ("alpha_only", "post_only", "alpha_plus_post", "post_resid_alpha",
                    "low_alpha", "high_alpha"):
            rows.append(_row("C4 alpha", f"{col} > 0.5",
                             stats.window_test(c4t[col].to_numpy(), cfg.n_permutations,
                                               cfg.n_bootstrap, cfg.seed)))
        for a, b in (("alpha_plus_post", "post_only"), ("post_only", "post_resid_alpha"),
                     ("high_alpha", "low_alpha")):
            rows.append(_row("C4 alpha", f"{a} minus {b}",
                             stats.paired_test(c4t[a].to_numpy(), c4t[b].to_numpy(),
                                               cfg.n_permutations, cfg.n_bootstrap, cfg.seed)))
        und = np.array([float(np.mean(z["undersampled"])) for z in c5.values()])
        rows.append(_row("C5 balance", "undersampled window AUC > 0.5",
                         stats.window_test(und, cfg.n_permutations, cfg.n_bootstrap, cfg.seed)))
        catch_rows = [{"subject": s, "n_fa": int(z["catch_fa"].sum()),
                       "n_cr": int((1 - z["catch_fa"]).sum()),
                       "auc_fa_vs_cr": float(z["auc_fa_vs_cr"])}
                      for s, z in c5.items() if "catch_fa" in z.files]
        catch = pd.DataFrame(catch_rows)
        catch.to_csv(res_dir / "c5_catch_per_subject.tsv", sep="\t", index=False)
        enough = catch[catch.n_fa >= 3]
        catch_note = f"{len(enough)} subjects with >= 3 false alarms (blueprint needs >= 10)"
        if len(enough) >= 10:
            d = enough.auc_fa_vs_cr.to_numpy() - 0.5
            w = sps.wilcoxon(d)
            rows.append({"family": "C5 catch", "test": "FA vs CR detector score (AUC - 0.5)",
                         "n": len(d), "mean": float(enough.auc_fa_vs_cr.mean()),
                         "t": float(w.statistic), "p": float(w.pvalue),
                         "d_z": stats.rank_biserial(d), "ci_low": None, "ci_high": None,
                         "note": "Wilcoxon; d_z column = rank-biserial"})

    # ---- A8 within: 3x3 + TOST ----
    a8 = _load(project, subjects, "a8_cross_task")
    order = list(cfg.tasks)
    mats = []
    for z in a8.values():
        if "matrix" not in z.files:
            continue
        tk = [str(t) for t in z["tasks"]]
        m = np.full((3, 3), np.nan)
        wm = _win(z["matrix"].reshape(-1, len(times)), times, WIN_S).reshape(len(tk), len(tk))
        for i, ti in enumerate(tk):
            for j, tj in enumerate(tk):
                m[order.index(ti), order.index(tj)] = wm[i, j]
        mats.append(m)
    a8m = np.array(mats)
    if len(a8m) >= 2:
        rows += _a8_rows(a8m, cfg)

    # ---- across-subject: A7, A8 LOSO, A10 ----
    print("group: loading trials of", len(subjects), "subjects for LOSO", flush=True)
    from start_vision.decoding.data import epochs_path
    from start_vision.decoding.step0 import included_tasks

    inc_file = project.root / "results" / "phase2" / "inclusion.tsv"
    inc = included_tasks(project.root) if inc_file.exists() else {
        s: [t for t in cfg.tasks if epochs_path(project.runs, s, t, cfg.epoch_set).exists()]
        for s in subjects}
    sets, info = [], None
    for s in subjects:
        d = SubjectData(project.runs, project.staged, s, inc[s], cfg)
        sets.append(across.normalize(d.main, cfg))
        info = info or d.main.info
        del d
    loso_auc, loso_subs = across.loso_sliding(sets, cfg, normalized=True)
    np.savez(arr_dir / "a7_loso_sliding.npz", auc=loso_auc, times=times,
             subjects=np.array(loso_subs))
    loso_mask = stats.cluster_mask(loso_auc, **kw)
    clusters.append(stats.cluster_auc(loso_auc, times, **kw).assign(
        family="A7 LOSO", series="LOSO sliding", n=len(loso_subs)))
    table, preds = across.loso_window(sets, cfg, WIN_S, normalized=True)
    table.to_csv(res_dir / "a7_loso_window_per_subject.tsv", sep="\t", index=False)
    preds.to_csv(arr_dir / "loso_predictions.tsv", sep="\t", index=False)
    rows.append(_row("A7 LOSO", "window detector 200-600 ms > 0.5",
                     stats.window_test(table.auc.to_numpy(), cfg.n_permutations,
                                       cfg.n_bootstrap, cfg.seed)))
    for wname, w in WINDOWS.items():
        rows.append(_row("A7 LOSO", f"LOSO sliding: {wname}",
                         stats.window_test(_win(loso_auc, times, w), cfg.n_permutations,
                                           cfg.n_bootstrap, cfg.seed)))
    lct, _ = across.loso_cross_task(sets, cfg, WIN_S, order, normalized=True)
    np.savez(arr_dir / "a8_loso_cross_task.npz", auc=lct, tasks=np.array(order))
    a10: dict[tuple[int, float], list[float]] = {}
    cap = list(sets[0].ch_names)
    chan_dir = project.root / "configs" / "channels"
    tsv = {} if len(cap) != 128 else {
        n: pd.read_csv(chan_dir / f"ds006171_to_biosemi{n}.tsv", sep="\t") for n in (64, 32)}
    for n_ch in A10_CHANNELS if tsv else ():
        sub_sets = sets if n_ch == 128 else [channels.subset(s, tsv[n_ch]) for s in sets]
        for start in A10_STARTS:
            t, _ = across.loso_window(sub_sets, cfg, (start, A10_END), normalized=True)
            a10[(n_ch, start)] = list(t.auc)
            print(f"group: A10 {n_ch} ch from {start * 1000:.0f} ms: "
                  f"{np.mean(t.auc):.3f}", flush=True)
        del sub_sets
    if a10:
        figures.t3_channels_windows(a10).to_csv(res_dir / "t3_channels_windows.tsv", sep="\t",
                                                index=False)

    # ---- tables ----
    t2 = figures.t2_tests(rows)
    t2.to_csv(res_dir / "t2_tests.tsv", sep="\t", index=False)
    pd.concat(clusters, ignore_index=True).to_csv(res_dir / "clusters.tsv", sep="\t",
                                                  index=False)

    # ---- figures ----
    out = [res_dir / "t2_tests.tsv", res_dir / "clusters.tsv",
           figures.f1_sliding(times, curves["A1 within-subject"],
                              {"pooled": masks["A1 within-subject"]["pooled"]}, fig_dir)]
    if a2_mask is not None:
        out.append(figures.f2_generalization(times, a2, a2_mask, fig_dir))
    a3 = _load(project, subjects, "a3_patterns")
    if a3 and len(cap) == 128:
        pats = np.array([z["patterns"] for z in a3.values()])
        pinfo = mne.pick_info(info, list(range(len(cap))))
        if pinfo.get_montage() is None:
            pinfo.set_montage(channels._template(), on_missing="ignore", verbose="error")
        out.append(figures.f3_patterns(pats, times, pinfo, list(WINDOWS.values()), fig_dir))
    if curves["A4/A5 contrast"]:
        out.append(figures.f_compare(times, {"pooled": pooled, **curves["A4/A5 contrast"]},
                                     "F4  Contrast controls (vs all-trial A1, pooled)",
                                     "F4_contrast", fig_dir))
    if curves["C3 eye movements"]:
        out.append(figures.f_compare(times, {"pooled": pooled, **curves["C3 eye movements"]},
                                     "F6  Eye-movement controls", "F6_eog", fig_dir))
    if len(c4t):
        out.append(figures.f7_paired(c4t, ["alpha_only", "post_only", "alpha_plus_post",
                                           "post_resid_alpha"], fig_dir))
    out.append(figures.f8_loso(times, loso_auc, loso_mask, table, fig_dir))
    w9 = {k: _win(v, times, WIN_S) for k, v in curves["A9 cross-condition"].items()}
    if len(a8m) >= 2 and w9:
        out.append(figures.f9_cross(a8m, order, w9, fig_dir))
    (res_dir / "group_notes.txt").write_text(
        f"subjects: {len(subjects)}\nC5 catch: {catch_note}\n"
        "A6 response-key control: not run (yes/no key side not recorded; decision 3)\n")
    return out
