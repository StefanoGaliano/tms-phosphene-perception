"""Phase 2 build step 11: stage caching on a synthetic 3-subject project (PHASE2 §7 Step 1).

The project mimics Phase 1 outputs: runs/<task>/derivatives/eegunify-epochs/... stimraw
epochs (256 Hz, -1..1 s) + epochs provenance, and sourcedata/staged/... events.tsv.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from start_vision.decoding import stages
from start_vision.decoding.config import DecodingConfig
from tests.synthetic.decoding import make_subject

CFG = DecodingConfig(n_jobs=1, n_repeats=1, n_permutations=200, tasks=("nocue",),
                     min_trials_per_class=20)
A = ("a1_sliding",)  # the synthetic project has no desc-preproc files (c3) or catch trials


def _write_subject(root: Path, sub: str, seed: int) -> None:
    ts = make_subject(sub, "seen_late", 120, n_channels=16, sfreq=256.0, d=1.0, seed=seed,
                      window_s=(-1.0, 1.0))
    m = ts.meta
    onsets = 5.0 + 2.5 * np.arange(len(m))
    tt = ["stimulus/" + ("seen" if s else "unseen") + f"/{h}/{o}/m1"
          for s, h, o in zip(m.seen, m.hemifield, m.orientation, strict=True)]
    ids = {t: i + 1 for i, t in enumerate(sorted(set(tt)))}
    ev = np.c_[np.round(onsets * 256).astype(int), np.zeros(len(m), int), [ids[t] for t in tt]]
    names = list(ts.ch_names[:-1])
    ep = mne.EpochsArray(ts.X[:, :-1] * 1e-6, mne.create_info(names, 256.0, "eeg"), ev,
                         tmin=ts.times[0], event_id=ids, verbose="error")
    d = root / "runs" / "nocue" / "derivatives" / "eegunify-epochs" / f"sub-{sub}" / "eeg"
    d.mkdir(parents=True)
    ep.save(d / f"sub-{sub}_task-nocue_desc-stimraw_epo.fif", verbose="error")
    (d / f"sub-{sub}_task-nocue_desc-epochs_provenance.json").write_text(
        json.dumps({"stage_key": f"key-{sub}", "status": "ok"}))
    s = root / "sourcedata" / "staged" / "ds006171" / f"sub-{sub}" / "eeg"
    s.mkdir(parents=True)
    pd.DataFrame({"onset": onsets, "duration": 0.0, "value": 1, "trial_type": tt,
                  "trial_number": m.trial_number, "gabor_contrast": m.gabor_contrast,
                  "rt_ms": m.rt_ms, "cue_valid": "n/a", "objective_correct": "n/a",
                  "raw_code": 11}).to_csv(s / f"sub-{sub}_task-nocue_events.tsv", sep="\t",
                                          index=False, na_rep="n/a")


def _project(tmp_path: Path) -> stages.Project:
    for i, sub in enumerate(("01", "02", "03")):
        _write_subject(tmp_path, sub, seed=i)
    return stages.Project.default(tmp_path)


def _status(outs: list[stages.Outcome]) -> list[str]:
    return [o.status for o in outs]


def test_cache_hit_rerun_on_config_change_and_provenance(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    subs = ["01", "02", "03"]
    assert _status(stages.run(proj, subs, CFG, A)) == ["ran"] * 4
    assert _status(stages.run(proj, subs, CFG, A)) == ["cached"] * 4
    # parallelism is not part of the key
    n_jobs2 = CFG.model_copy(update={"n_jobs": 2})
    assert _status(stages.run(proj, subs, n_jobs2, A)) == ["cached"] * 4
    # a decoding setting changes every sv_decode* key
    assert _status(stages.run(proj, subs, CFG.model_copy(update={"n_folds": 4}), A)) == ["ran"] * 4
    prov = json.loads((proj.sub_dir("01") / "sub-01_desc-decoding_provenance.json").read_text())
    assert prov["status"] == "ok" and prov["outputs"]
    assert all((tmp_path / o).exists() for o in prov["outputs"])
    assert prov["upstream_keys"]["nocue:epochs"] == "key-01"
    g = json.loads((proj.group_dir() / "provenance.json").read_text())
    assert g["included"] == subs and (tmp_path / "results" / "phase2" / "clusters.tsv").exists()
    assert all((tmp_path / o).exists() for o in g["outputs"])


def test_upstream_change_reruns_only_that_subject_and_group(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    subs = ["01", "02", "03"]
    stages.run(proj, subs, CFG, A)
    ev = proj.staged / "sub-02" / "eeg" / "sub-02_task-nocue_events.tsv"
    ev.write_text(ev.read_text() + "\n")  # metadata source changed
    assert _status(stages.run(proj, subs, CFG, A)) == ["cached", "ran", "cached", "ran"]
    (proj.sub_dir("03") / "sub-03_desc-a1_sliding_scores.npz").unlink()  # output lost
    assert _status(stages.run(proj, subs, CFG, A, group=False)) == ["cached", "cached", "ran"]


def test_missing_upstream_is_skipped_and_excluded(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    out = stages.run(proj, ["01", "02", "03", "04"], CFG, A)
    assert out[3].status == "skipped"
    g = json.loads((proj.group_dir() / "provenance.json").read_text())
    assert g["excluded"] == {"04": "sv_decode not run"}


def test_inclusion_list_limits_tasks(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    inc = tmp_path / "results" / "phase2"
    inc.mkdir(parents=True)
    pd.DataFrame({"subject": ["01", "02", "03"], "task": "nocue",
                  "included": [True, False, True]}).to_csv(inc / "inclusion.tsv", sep="\t",
                                                           index=False)
    out = stages.run(proj, ["01", "02", "03"], CFG, A)
    assert [o.status for o in out] == ["ran", "skipped", "ran", "ran"]
