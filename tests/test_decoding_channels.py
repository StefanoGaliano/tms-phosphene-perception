"""Phase 2 build step 9: channel mapping, mirror, harmonize."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from start_vision.decoding import channels as ch
from start_vision.decoding.data import TrialSet
from tests.synthetic.decoding import make_subject

ROOT = Path(__file__).parents[1]
CAP_TSV = ROOT / "sourcedata" / "ds006171" / "sub-01" / "eeg" / "sub-01_electrodes.tsv"


def _fake_electrodes(tmp_path: Path, names: list[str]) -> Path:
    pos = ch.positions(names)
    rng = np.random.default_rng(0)
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rot *= np.sign(np.linalg.det(rot))
    xyz = np.array([pos[n] for n in names]) @ rot * 1100 + 7  # mm, rotated, scaled, shifted
    f = tmp_path / "electrodes.tsv"
    pd.DataFrame({"name": names, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                  "type": "EEG"}).to_csv(f, sep="\t", index=False)
    return f


def test_procrustes_recovers_template_frame(tmp_path: Path) -> None:
    names = ["Fp1", "Fp2", "Cz", "Pz", "O1", "O2", "T7", "T8", "CPP3", "PPO3", "Fz", "Oz"]
    _, resid = ch.cap_positions(_fake_electrodes(tmp_path, names))
    assert resid < 1e-6


def test_nearest_subset_exact_first_unique_deterministic() -> None:
    cap = ["Cz", "CPP3", "PPO3", "O1", "O2", "Pz"]
    t1 = ch.nearest_subset(cap, ["Cz", "P3", "O1"])
    t2 = ch.nearest_subset(cap, ["Cz", "P3", "O1"])
    assert t1.equals(t2)
    assert t1.set_index("target").loc["Cz", "cap"] == "Cz" and t1.cap.is_unique
    assert not t1.set_index("target").loc["P3", "exact"]
    with pytest.raises(ValueError):
        ch.nearest_subset(["Cz"], ["Cz", "Pz"])


def test_target_layouts_are_1005_names() -> None:
    std = set(ch._template().ch_names)
    for n in (64, 32):
        names = ch.target_layout(n)
        assert len(names) == n and set(names) <= std


@pytest.mark.skipif(not CAP_TSV.exists(), reason="ds006171 not downloaded")
def test_ds006171_mapping_matches_committed_tables_and_phase1() -> None:
    pos, resid = ch.cap_positions(CAP_TSV)
    assert len(pos) == 128 and round(resid, 1) == 8.7  # PHASE1_REFERENCES §4
    for n in (64, 32):
        committed = pd.read_csv(ROOT / "configs" / "channels" / f"ds006171_to_biosemi{n}.tsv",
                                sep="\t")
        fresh = ch.mapping_table(CAP_TSV, n)
        assert fresh.target.tolist() == committed.target.tolist()
        assert fresh.cap.tolist() == committed.cap.tolist() and fresh.cap.is_unique
    p3 = fresh.set_index("target").loc["P3"]
    assert p3.cap == "CPP3" and p3.distance_mm == pytest.approx(10.8, abs=0.05)


def test_mirror_names_and_data() -> None:
    assert ch.mirror_name("O1") == "O2" and ch.mirror_name("CPP4h") == "CPP3h"
    assert ch.mirror_name("Oz") == "Oz" and ch.mirror_name("PO10") == "PO9"
    names = ["O1", "Oz", "O2", "CPP3h", "CPP4h"]
    info = mne.create_info(names, 128.0, "eeg")
    x = np.arange(5, dtype=np.float32)[None, :, None] * np.ones((2, 5, 4), np.float32)
    ts = make_subject("1", "null", 2)
    ts = TrialSet(x, ts.times[:4], tuple(names), 128.0, ts.meta, info)
    m = ch.mirror(ts)
    assert m.X[0, :, 0].tolist() == [2, 1, 0, 4, 3]
    with pytest.raises(ValueError):
        ch.mirror_index(["O1", "Cz"])


def test_subset_renames_to_targets() -> None:
    ts = make_subject("1", "null", 5)
    table = pd.DataFrame({"target": ["A", "B"], "cap": ["E03", "E01"]})
    s = ch.subset(ts, table)
    assert s.ch_names == ("A", "B") and np.array_equal(s.X[:, 0], ts.X[:, 3])
    assert s.info.ch_names == ["A", "B"]


def test_harmonize_channels_sfreq_and_times() -> None:
    a = make_subject("1", "null", 6, n_channels=32, sfreq=256.0)
    b = make_subject("2", "null", 6, n_channels=16, sfreq=128.0, window_s=(-0.2, 0.8))
    ha, hb = ch.harmonize([a, b])
    assert ha.ch_names == hb.ch_names and len(ha.ch_names) == 16  # E00..E14 + HEOG
    assert ha.sfreq == hb.sfreq == 128.0 and np.allclose(ha.times, hb.times)
    assert ha.times[0] == pytest.approx(b.times[0])  # the later start wins
    c = make_subject("3", "null", 6, sfreq=100.0)
    with pytest.raises(ValueError):
        ch.harmonize([a, c])
