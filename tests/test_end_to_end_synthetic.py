"""Step 8: synthetic stage -> eegunify convert -> erp -> group, with planted VAN and LP.

pytest -m slow (pyprep + ICA on 2 x N_SUBJECTS 128-channel recordings: several minutes).
Truth is the noise-free planted signal after the same average reference and window mean.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pybv
import pytest
import yaml

from start_vision.configs import render
from start_vision.matching import match
from start_vision.stage import build_trials, staged_events
from tests.synthetic.vision import (
    LP_MS,
    LP_SIGMA_MS,
    LP_UV,
    SFREQ,
    VAN_MS,
    VAN_SIGMA_MS,
    VAN_UV,
    make_vision_recording,
    window_mean,
)

pytestmark = pytest.mark.slow
N_SUBJECTS = 8  # sign-flip permutation p can reach 1/256
ROOT = Path(__file__).parents[1]
EEGUNIFY = str(Path(sys.executable).with_name("eegunify"))


def _project(root: Path, null: bool) -> tuple[Path, list]:
    subs = [f"{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    recs = []
    for i, s in enumerate(subs):
        rec = make_vision_recording(seed=100 * int(null) + i, null=null)
        tr = build_trials(rec.events, "nocue")
        tr["matched"] = match(tr)
        ev = staged_events(tr, rec.events, "nocue")
        d = root / "sourcedata" / "synth" / f"sub-{s}" / "eeg"
        d.mkdir(parents=True)
        stem = f"sub-{s}_task-nocue_eeg"
        pybv.write_brainvision(
            data=rec.data_uV * 1e-6,
            sfreq=SFREQ,
            ch_names=rec.names,
            fname_base=stem,
            folder_out=d,
            unit="µV",
            overwrite=True,
        )
        ev.to_csv(d / f"sub-{s}_task-nocue_events.tsv", sep="\t", index=False, na_rep="n/a")
        recs.append(rec)
    cfg = render((ROOT / "configs" / "template_ds006171.yaml").read_text(), "nocue", subs)
    cfg["dataset"]["source_root"] = "sourcedata/synth"
    cfg["recordings"]["pattern"] = "sub-{subject}/eeg/sub-{subject}_task-nocue_eeg.vhdr"
    cfg["channels"]["rename"] = {}
    cfg["channels"]["drop"] = []
    cfg["components"]["library_paths"] = [str(ROOT / "configs" / "components" / "awareness.yaml")]
    p = root / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    for cmd in (["convert"], ["erp", "run"]):
        r = subprocess.run(
            [EEGUNIFY, *cmd, str(p), "--root", str(root)], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return root, recs


@pytest.fixture(scope="module")
def effect(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list]:
    return _project(tmp_path_factory.mktemp("effect"), null=False)


@pytest.fixture(scope="module")
def null(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list]:
    return _project(tmp_path_factory.mktemp("null"), null=True)


def _measures(root: Path, comp: str) -> np.ndarray:
    m = pd.concat(
        pd.read_csv(f, sep="\t", dtype={"subject": str})
        for f in sorted(root.glob("derivatives/eegunify-erp/sub-*/eeg/*_desc-stim_measures.tsv"))
    )
    sel = (
        (m.component == comp) & (m.wave == "seen_minus_unseen_m") & (m.measure == "mean_amplitude")
    )
    return m.loc[sel, "value"].to_numpy()


def _truth(recs: list, kind: str) -> float:
    r = recs[0]
    names = r.names[: r.n_eeg]
    if kind == "van":  # average of left- and right-Gabor topographies, average-referenced
        w = sum(t - t.mean() for t in r.van_topo.values()) / 2
        return (VAN_UV["seen"] - VAN_UV["unseen"]) * window_mean(
            w, names, ("O1", "O2"), VAN_MS, VAN_SIGMA_MS, (180, 280)
        )
    w = r.lp_topo - r.lp_topo.mean()
    return (LP_UV["seen"] - LP_UV["unseen"]) * window_mean(
        w, names, ("Pz", "CPP3", "CPP4"), LP_MS, LP_SIGMA_MS, (350, 550)
    )


def test_no_roi_channel_interpolated(effect: tuple[Path, list]) -> None:
    import json

    roi = {"O1", "O2", "Pz", "CPP3", "CPP4"}
    for f in effect[0].glob("derivatives/eegunify-preproc/sub-*/eeg/*_preproc_qa.json"):
        bad = set(json.loads(f.read_text())["steps"].get("interpolated", []))
        assert not bad & roi, (f.name, sorted(bad))


@pytest.mark.parametrize("comp", ["van", "lp"])
def test_recovered_effect_matches_truth(effect: tuple[Path, list], comp: str) -> None:
    root, recs = effect
    v = _measures(root, comp)
    truth = _truth(recs, comp)
    se = v.std(ddof=1) / np.sqrt(len(v))
    # 10 % of truth, widened to 3 standard errors of the subject mean if noise is larger
    tol = max(0.10 * abs(truth), 3 * se)
    assert len(v) == N_SUBJECTS
    assert abs(v.mean() - truth) < tol, (comp, v.mean(), truth, se)


def test_lateralized_double_difference_negative(effect: tuple[Path, list]) -> None:
    root, _ = effect
    dd = []
    for f in sorted(root.glob("derivatives/eegunify-erp/sub-*/eeg/*_desc-stim_ave.fif")):
        e = {x.comment: x for x in mne.read_evokeds(f, verbose="error")}
        d = mne.combine_evoked([e["seen_lat"], e["unseen_lat"]], weights=[1, -1])
        sel = (d.times >= 0.18) & (d.times <= 0.28)
        dd.append(d.copy().pick(["O1"]).data[0, sel].mean())
    assert np.mean(dd) < 0 and sum(x < 0 for x in dd) >= N_SUBJECTS - 1


def _clusters(root: Path) -> pd.DataFrame:
    # Start_Vision's own cluster test, one wave per subject (eegunify 0.9.2's group cluster
    # test counts each subject once per epoch set: stim + stimraw = twice)
    from start_vision.analysis import stats

    evs = [
        {x.comment: x for x in mne.read_evokeds(f, verbose="error")}["seen_minus_unseen_m"]
        for f in sorted(root.glob("derivatives/eegunify-erp/sub-*/eeg/*_desc-stim_ave.fif"))
    ]
    c = stats.cluster(evs)
    assert (c.n_subjects == N_SUBJECTS).all()
    return c


def test_cluster_finds_planted_effect(effect: tuple[Path, list]) -> None:
    c = _clusters(effect[0])
    sig = c[c.p < 0.05]
    neg = sig[(sig.t_sum < 0) & sig.channels.str.contains("O1|O2")]
    pos = sig[(sig.t_sum > 0) & sig.channels.str.contains("Pz")]
    assert len(neg) and len(pos), c.to_string()
    assert ((neg.tmin_ms < VAN_MS) & (neg.tmax_ms > VAN_MS)).any(), c.to_string()


def test_cluster_null_finds_nothing(null: tuple[Path, list]) -> None:
    c = _clusters(null[0])
    assert (c.p >= 0.05).all(), c.to_string()
    assert np.abs(_measures(null[0], "lp").mean()) < 0.5  # planted difference 0
