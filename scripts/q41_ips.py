"""Q41 (docs/QUESTIONS_PREREG.md §12, tag questions-prereg-amend5): the frozen detector on IPS
parietal phosphenes (GIN DB_123/IPS_parietal_phosphenes).

  python -I scripts/q41_ips.py clean      # one recording at a time: download .fdt, clean in the
                                          # three pipelines, C3-C4 code check, delete the .fdt
  python -I scripts/q41_ips.py analyze    # scoring, [P] and secondary tests

The Phase 3 code is reused unchanged; only the config paths change (data_dir, out_dir).
Sites P3/P4 are reached through symlinks P3 -> P3_left_stim, P4 -> P4_right_stim.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from start_vision.phase3 import analyses as an  # noqa: E402
from start_vision.phase3 import config, score, stages  # noqa: E402

DATA = ROOT / "sourcedata" / "ips_parietal_phosphenes"
URL = "https://gin.g-node.org/DB_123/IPS_parietal_phosphenes/raw/master"
FOLDER = {"P3": "P3_left_stim", "P4": "P4_right_stim"}
RES = ROOT / "results" / "ips"
PIPELINES = ("primary", "paperlike", "gonogo")
SEED = 20261109                      # amendment 5
MAX_S100_SHARE = 0.5                 # amendment 5 label rule


def cfg() -> config.Phase3Config:
    c = config.load(ROOT / "configs" / "phase3.yaml")
    return c.model_copy(update={"data_dir": Path("sourcedata/ips_parietal_phosphenes"),
                                "out_dir": Path("derivatives/sv_ips_clean")})


def recordings() -> pd.DataFrame:
    cc = pd.read_csv(RES / "code_count.tsv", sep="\t", dtype={"subject": str})
    cc["site"] = cc["site"].str[:2]
    cc["recording"] = "sub-" + cc["subject"] + "_site-" + cc["site"]
    cc["label_ok"] = cc["S100_share"] <= MAX_S100_SHARE
    return cc


def _size(url: str) -> int:
    h = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    return int([ln for ln in h.splitlines() if ln.lower().startswith("content-length")][-1]
               .split(":")[1])


def download(fdt: Path) -> None:
    url = f"{URL}/{fdt.parent.name}/{fdt.name}"
    n = _size(url)
    if fdt.exists() and fdt.stat().st_size == n:
        return
    rc = 0
    for _ in range(5):            # GIN refuses range resume (curl 33): retry from scratch
        fdt.unlink(missing_ok=True)
        rc = subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(fdt), url]).returncode
        if fdt.exists() and fdt.stat().st_size == n:
            break
        time.sleep(10)
    else:
        raise OSError(f"{fdt.name}: download failed (curl {rc})")
    if fdt.stat().st_size != n:
        raise OSError(f"{fdt.name}: {fdt.stat().st_size} B, expected {n}")


def run_clean(local_only: bool = False) -> None:
    for s, f in FOLDER.items():
        link = DATA / s
        if not link.exists():
            link.symlink_to(f)
    c = cfg()
    recs = recordings()
    cpath = RES / "code_check.tsv"
    codes = pd.read_csv(cpath, sep="\t") if cpath.exists() else pd.DataFrame(
        columns=["recording", "code_diff_uv", "code_p"])
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=1)
    prefetch: dict = {}
    done_ids: set = set()
    todo = []
    for q in recs.itertuples():
        if not q.label_ok or all(stages.out_paths(c, ROOT, p, q.recording)["epo"].exists()
                                 for p in PIPELINES):
            continue
        qs = stages.set_path(c, ROOT, q.subject, q.site)
        todo.append((q.recording, q.recording, DATA / FOLDER[q.site] / qs.name.replace(".set", ".fdt")))
    for r in recs.itertuples():
        if not r.label_ok:
            print(f"skip {r.recording}: S100 share {r.S100_share} > {MAX_S100_SHARE}", flush=True)
            continue
        done = all(stages.out_paths(c, ROOT, p, r.recording)["epo"].exists() for p in PIPELINES)
        if done and r.recording in set(codes["recording"]):
            print(f"done {r.recording}", flush=True)
            continue
        sp = stages.set_path(c, ROOT, r.subject, r.site)
        fdt = DATA / FOLDER[r.site] / sp.name.replace(".set", ".fdt")
        if local_only and not fdt.exists():
            continue
        t0 = time.time()
        if r.recording in prefetch:
            prefetch.pop(r.recording).result()
        download(fdt)
        if not local_only:        # fetch the next recording while this one is cleaned
            nxt = next((q for q in todo if q[0] not in prefetch and q[0] != r.recording
                        and q[1] not in done_ids), None)
            if nxt is not None:
                prefetch[nxt[0]] = pool.submit(download, nxt[2])
        done_ids.add(r.recording)
        t1 = time.time()
        for p in PIPELINES:
            stages.run(c, ROOT, r.subject, r.site, p)
        if r.recording not in set(codes["recording"]):
            cc = an.code_check(sp)
            codes = pd.concat([codes, pd.DataFrame([{"recording": r.recording, **cc}])],
                              ignore_index=True)
            codes.to_csv(cpath, sep="\t", index=False)
        fdt.unlink()
        print(f"cleaned {r.recording}: download {t1 - t0:.0f} s, clean {time.time() - t1:.0f} s",
              flush=True)


def hand_groups(per: pd.DataFrame, codes: pd.DataFrame) -> pd.Series:
    """Amendment 5: phosphene = right hand if C3-C4 (S100 - S200) negative with p < 0.05,
    left hand if positive with p < 0.05; a subject whose significant recordings disagree, or
    with none significant, is unassigned."""
    m = per.merge(codes, on="recording")
    out = {}
    for sub, g in m.groupby("subject"):
        sig = g[g["code_p"] < 0.05]
        signs = set(np.sign(sig["code_diff_uv"]))
        out[sub] = ("right" if signs == {-1.0} else "left" if signs == {1.0} else "unassigned")
    return pd.Series(out, name="hand")


def run_analyze() -> None:
    from scipy.stats import mannwhitneyu, wilcoxon

    from start_vision.decoding import stats

    c = cfg()
    codes = pd.read_csv(RES / "code_check.tsv", sep="\t")
    summary: dict[str, object] = {}
    for p in PIPELINES:
        sc = score.score_all(c, ROOT, p)
        sc.to_csv(RES / f"scores_{p}.tsv", sep="\t", index=False)
        inc = an.inclusion(sc)          # no code-direction exclusion here (amendment 5)
        inc = inc.merge(codes, on="recording", how="left")
        inc.to_csv(RES / f"inclusion_{p}.tsv", sep="\t", index=False)
        keep = set(inc.loc[inc["included"], "recording"])
        per = an.per_recording(sc[sc["recording"].isin(keep)])
        per.to_csv(RES / f"per_recording_{p}.tsv", sep="\t", index=False)
        sub_auc = an.subject_values(per, "auc")
        primary = an.group_one_sided(sub_auc, seed=SEED)
        summary[p] = {"n_recordings": len(per), **primary}
        if p != "primary":
            continue
        t2, t3 = an.tests(per)
        t2 = t2[t2["test"] != "S1 untouched subset"]        # no go/no-go subset here
        t2.to_csv(RES / "t2_tests_primary.tsv", sep="\t", index=False)
        t3.to_csv(RES / "t3_controls_primary.tsv", sep="\t", index=False)
        # (a) per site and P3 vs P4
        for s in ("P3", "P4"):
            summary[f"site_{s}"] = an.group_one_sided(per.loc[per.site == s, "auc"], seed=SEED)
        w = per.pivot_table(index="subject", columns="site", values="auc").dropna()
        summary["site_P3_vs_P4"] = {"n": len(w), "mean_P3": float(w["P3"].mean()),
                                    "mean_P4": float(w["P4"].mean()),
                                    "p_wilcoxon": float(wilcoxon(w["P3"], w["P4"]).pvalue)}
        # (c) hand split
        hg = hand_groups(per, codes)
        hg.to_csv(RES / "hand_groups.tsv", sep="\t")
        d = pd.concat([sub_auc.rename("auc"), hg], axis=1)
        gr, gl = d.loc[d.hand == "right", "auc"], d.loc[d.hand == "left", "auc"]
        summary["hand_split"] = {
            "n_right": len(gr), "n_left": len(gl), "n_unassigned": int((d.hand == "unassigned").sum()),
            "mean_right": float(gr.mean()) if len(gr) else None,
            "mean_left": float(gl.mean()) if len(gl) else None,
            "p_mannwhitney": (float(mannwhitneyu(gr, gl).pvalue)
                              if len(gr) and len(gl) else None)}
        summary["primary_two_sided_check"] = float(
            stats._signflip_p(sub_auc.to_numpy() - 0.5, 10_000, SEED, tail=0))
        print(t2.to_string(index=False))
        print(t3.to_string(index=False))
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "clean":
        run_clean(local_only="--local-only" in sys.argv)
    else:
        run_analyze()
