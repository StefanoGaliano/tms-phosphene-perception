"""Phase 3 S7 and R4b (PHASE3_PREREG §8, §9), run once the ds006171 epochs were rebuilt.

S7  Retrained transfer: a sliding decoder (Phase 2 estimator, ``across.transfer``) trained on all
    included ds006171 subjects (stimulus-present trials, the 64-channel ``tms-robust`` set),
    tested time point by time point on each included TMS recording. No TMS labels in training.
    Group: subject = mean of its included sites; cluster test over time and window tests.
R4b Early-only detector: Phase 2 recipe (``tms-robust`` channels, LOSO-calibrated), feature
    window fixed at 0.15-0.30 s, C chosen by LOSO over the Phase 2 grid; applied to the TMS data
    with the H1 test. Saved under results/phase3/r4b/, not exported to ``detectors/``.

Usage: python scripts/phase3_s7_r4b.py [--backend torch|sklearn] [--only s7|r4b]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from start_vision.decoding import across, channels, config, stats
from start_vision.decoding.analyses import SubjectData
from start_vision.decoding.data import META_COLUMNS, TrialSet
from start_vision.decoding.detector import Detector
from start_vision.decoding.export import nested_selection
from start_vision.decoding.stages import Project
from start_vision.decoding.step0 import included_tasks
from start_vision.phase3.analyses import group_one_sided, per_recording, subject_values
from start_vision.phase3.score import detector_input, score_epochs

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "detectors" / "sv-perception-tms-robust-v1.0.0"
R4B_WINDOW = (0.15, 0.30)
WINDOWS = {"VAN 180-280": (0.18, 0.28), "LP 350-550": (0.35, 0.55),
           "post 150-600": (0.15, 0.6), "prestim -500-0": (-0.5, 0.0)}
PIPELINES = ("primary", "paperlike")


def training_sets(cfg: config.DecodingConfig) -> tuple[list[TrialSet], list[TrialSet | None],
                                                         pd.DataFrame]:
    proj = Project.default(ROOT)
    table = pd.read_csv(ROOT / "configs" / "channels" / "ds006171_to_biosemi64.tsv", sep="\t")
    sets, catch = [], []
    for s, tasks in sorted(included_tasks(ROOT).items()):
        d = SubjectData(proj.runs, proj.staged, s, tasks, cfg)
        sets.append(channels.subset(d.main, table))
        catch.append(channels.subset(d.catch, table) if len(d.catch) else None)
        print(f"train sub-{s}: {len(sets[-1])} trials", flush=True)
        del d
    return sets, catch, table


def included(pipeline: str) -> pd.DataFrame:
    inc = pd.read_csv(ROOT / "results" / "phase3" / f"inclusion_{pipeline}.tsv", sep="\t",
                      dtype={"subject": str})
    return inc[inc["included"]]


def read_rec(pipeline: str, rid: str) -> mne.BaseEpochs:
    return mne.read_epochs(ROOT / "derivatives" / "sv_p3_clean" / pipeline
                           / f"{rid}_desc-detector_epo.fif", verbose="error")


def tms_trialset(ep: mne.BaseEpochs, det: Detector, rid: str) -> TrialSet:
    e = detector_input(ep, det)
    m = pd.DataFrame({c: np.nan for c in META_COLUMNS}, index=range(len(e)))
    m["subject"], m["dataset"], m["task"] = rid, "GIN_phosphenes", "tms"
    m["seen"] = ep.metadata["seen"].to_numpy().astype(int)
    m["stim_present"] = True
    return TrialSet((e.get_data() * 1e6).astype(np.float32), e.times, tuple(e.ch_names),
                    float(e.info["sfreq"]), m, e.info)


def s7(cfg: config.DecodingConfig, sets: list[TrialSet], out: Path) -> dict:
    det = Detector.load(FROZEN)
    res: dict = {"n_train_subjects": len(sets), "n_train_trials": int(sum(map(len, sets)))}
    for pipe in PIPELINES:
        inc = included(pipe)
        test = [tms_trialset(read_rec(pipe, r), det, r) for r in inc["recording"]]
        h = channels.harmonize(sets + test)
        tr, te = h[:len(sets)], h[len(sets):]
        df = across.transfer(tr, te, cfg)
        times = tr[0].times
        auc = np.vstack([np.asarray(a, dtype=float) for a in df["auc"]])
        rec_sub = dict(zip(inc["recording"], inc["subject"], strict=True))
        subs = sorted(set(rec_sub.values()))
        sub_auc = np.vstack([auc[[rec_sub[r] == s for r in df["subject"]]].mean(0)
                             for s in subs])
        np.savez(out / f"s7_auc_{pipe}.npz", rec_auc=auc, sub_auc=sub_auc, times=times,
                 recordings=df["subject"].to_numpy(), subjects=np.array(subs))
        win = {}
        for k, (lo, hi) in WINDOWS.items():
            msk = (times >= lo - 1e-9) & (times <= hi + 1e-9)
            win[k] = stats.window_test(sub_auc[:, msk].mean(1), seed=cfg.seed)
        cl = stats.cluster_auc(sub_auc, times, seed=cfg.seed)
        res[pipe] = {"n_subjects": len(subs), "n_recordings": len(auc), "windows": win,
                     "clusters": cl[cl["p"] < 0.05].to_dict("records"),
                     "all_clusters": cl.to_dict("records"),
                     "peak_auc": float(sub_auc.mean(0).max()),
                     "peak_time_s": float(times[int(sub_auc.mean(0).argmax())])}
        print(f"S7 {pipe}: peak {res[pipe]['peak_auc']:.3f} at {res[pipe]['peak_time_s']:.3f} s;"
              f" sig clusters {len(res[pipe]['clusters'])}", flush=True)
    (out / "s7.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res


def r4b(cfg: config.DecodingConfig, sets: list[TrialSet], catch: list[TrialSet | None],
        table: pd.DataFrame, out: Path, backend: str) -> dict:
    sel = nested_selection(sets, catch, cfg, [R4B_WINDOW], backend)
    subjects = [str(s.meta["subject"].iloc[0]) for s in sets]
    det = Detector.fit(sets, cfg, R4B_WINDOW, variant="tms-robust", version="0.0.0-r4b",
                       C=sel["C"], catch=catch, channel_table=table,
                       trained_on={"dataset": "OpenNeuro ds006171", "subjects": subjects,
                                   "note": "R4b analysis detector; not exported"})
    folder = det.save(out / "r4b_detector")
    nested = np.asarray(sel["nested_auc"], dtype=float)
    res: dict = {"window_s": list(R4B_WINDOW), "C": sel["C"], "backend": backend,
                 "loso_auc_mean": det.performance["loso_auc_mean"],
                 "nested_loso_auc_mean": float(nested.mean()), "detector": str(folder)}
    for pipe in PIPELINES:
        inc = included(pipe)
        rows = []
        for rid, sub, site in inc[["recording", "subject", "site"]].itertuples(index=False):
            ep = read_rec(pipe, rid)
            p = score_epochs(ep, det)
            rows.append(pd.DataFrame({"recording": rid, "subject": sub, "site": site,
                                      "seen": ep.metadata["seen"].to_numpy().astype(int),
                                      "p_seen": p}))
        scores = pd.concat(rows, ignore_index=True)
        scores.to_csv(out / f"r4b_scores_{pipe}.tsv", sep="\t", index=False)
        pr = per_recording(scores, n_perm=2000)
        pr.to_csv(out / f"r4b_per_recording_{pipe}.tsv", sep="\t", index=False,
                  float_format="%.5f")
        res[pipe] = group_one_sided(subject_values(pr, "auc")) | {
            "n_subjects": int(pr["subject"].nunique()), "n_recordings": len(pr),
            "subjects_above_chance": int((subject_values(pr, "auc") > 0.5).sum())}
        print(f"R4b {pipe}: {res[pipe]}", flush=True)
    (out / "r4b.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--backend", default="torch", choices=["torch", "sklearn"])
    a.add_argument("--only", choices=["s7", "r4b"])
    args = a.parse_args()
    cfg = config.load(ROOT / "configs" / "decoding.yaml")
    out = ROOT / "results" / "phase3" / "s7_r4b"
    out.mkdir(parents=True, exist_ok=True)
    sets, catch, table = training_sets(cfg)
    if args.only in (None, "r4b"):
        r4b(cfg, sets, catch, table, out, args.backend)
    if args.only in (None, "s7"):
        s7(cfg, sets, out)


if __name__ == "__main__":
    main()
