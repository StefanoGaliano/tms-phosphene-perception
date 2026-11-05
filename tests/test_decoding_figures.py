"""Phase 2 build step 12: every figure renders (PNG + SVG) and the tables are right."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from start_vision.decoding import figures as fg
from tests.synthetic.decoding import make_subject

T = np.arange(-64, 128) / 128


def _auc(rng: np.random.Generator, n: int = 6, bump: float = 0.1) -> np.ndarray:
    a = 0.5 + 0.02 * rng.standard_normal((n, len(T)))
    a[:, (T > 0.3) & (T < 0.5)] += bump
    return a


def _ok(p: Path) -> None:
    assert p.exists() and p.with_suffix(".svg").exists() and p.stat().st_size > 5000


def test_all_figures_render(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    mask = (T > 0.3) & (T < 0.5)
    _ok(fg.f1_sliding(T, {"pooled": _auc(rng), "nocue": _auc(rng), "informative": _auc(rng)},
                      {"pooled": mask}, tmp_path))
    gen = 0.5 + 0.02 * rng.standard_normal((4, 40, 40))
    gen[:, 15:25, 15:25] += 0.1
    m2 = np.zeros((40, 40), bool)
    m2[15:25, 15:25] = True
    _ok(fg.f2_generalization(np.linspace(-0.5, 1.0, 40), gen, m2, tmp_path))
    info = mne.create_info(["O1", "O2", "Pz", "Cz", "Fz", "P3", "P4", "Oz"], 128.0, "eeg")
    info.set_montage("standard_1005", verbose="error")
    _ok(fg.f3_patterns(rng.standard_normal((3, 8, len(T))), T, info, [(0.2, 0.3), (0.35, 0.55)],
                       tmp_path))
    _ok(fg.f_compare(T, {"all trials": _auc(rng), "within bins": _auc(rng, bump=0.05)},
                     "F4  Within-contrast-bin decoding", "F4_within_bin", tmp_path))
    tab = pd.DataFrame({"alpha_only": rng.uniform(0.45, 0.6, 10),
                        "post_only": rng.uniform(0.55, 0.7, 10),
                        "alpha_plus_post": rng.uniform(0.55, 0.72, 10)})
    _ok(fg.f7_paired(tab, list(tab.columns), tmp_path))
    w = pd.DataFrame({"subject": [f"{i}" for i in range(10)], "auc": rng.uniform(0.5, 0.7, 10)})
    _ok(fg.f8_loso(T, _auc(rng, 10), mask, w, tmp_path))
    _ok(fg.f9_cross(0.55 + 0.03 * rng.standard_normal((8, 3, 3)),
                    ["nocue", "noninformative", "informative"],
                    {"hemifield": rng.uniform(0.5, 0.65, 8),
                     "orientation": rng.uniform(0.5, 0.65, 8)}, tmp_path))
    cal = pd.DataFrame({"bin_low": np.linspace(0, 0.9, 10), "bin_high": np.linspace(0.1, 1, 10),
                        "n": 50, "mean_p": np.linspace(0.05, 0.95, 10),
                        "observed": np.linspace(0.08, 0.9, 10)})
    _ok(fg.f10_calibration(cal, {"seen": rng.beta(4, 2, 200), "unseen": rng.beta(2, 4, 200),
                                 "catch FA": rng.beta(3, 3, 20), "catch CR": rng.beta(2, 5, 80)},
                           tmp_path))


def test_t1_descriptives() -> None:
    m = make_subject("01", "null", 100).meta.copy()
    m.loc[m.index[:10], "stim_present"] = False
    t = fg.t1_descriptives(m)
    r = t.iloc[0]
    assert r.n_trials == 100 and r.n_catch == 10 and r.n_seen + r.n_unseen == 90
    assert r.hit_rate == pytest.approx(r.n_seen / 90)
    assert 0 <= r.share_in_overlap <= 1


def test_t2_holm_within_family_and_t3() -> None:
    t2 = fg.t2_tests([{"family": "A", "test": "x", "p": 0.01},
                      {"family": "A", "test": "y", "p": 0.04},
                      {"family": "B", "test": "z", "p": 0.04}])
    assert list(t2.p_holm) == pytest.approx([0.02, 0.04, 0.04])
    t3 = fg.t3_channels_windows({(64, 0.15): [0.6, 0.62, 0.58, 0.61],
                                 (32, 0.2): [0.55, 0.57, 0.52, 0.56]})
    assert list(t3.n_channels) == [32, 64] and list(t3.window_start_ms) == [200, 150]
