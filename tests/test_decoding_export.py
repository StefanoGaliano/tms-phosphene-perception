"""Phase 2 build step 15: nested selection and export (A11)."""

from __future__ import annotations

import numpy as np

from start_vision.decoding import export as ex
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.detector import Detector
from tests.synthetic.decoding import make_subject, make_synthetic_trialset

CFG = DecodingConfig(n_jobs=1, C_grid=(0.1, 1.0))


def test_nested_selection_matches_brute_force() -> None:
    sets = make_synthetic_trialset(4, 80, generator="seen_late", d=1.0, seed=3)
    catch = [None] * 4
    wins = [(0.2, 0.6), (0.3, 0.8)]
    sel = ex.nested_selection(sets, catch, CFG, wins)
    ys = [s.labels("seen") for s in sets]
    for i in range(4):  # brute force: inner LOSO over the 3 others, then score subject i
        others = [k for k in range(4) if k != i]
        best, best_score = None, -1.0
        for w in wins:
            fs = ex._features(sets, catch, CFG, w)
            for c in CFG.C_grid:
                sc = np.mean([ex._fit_score(CFG, c, fs, ys, [k for k in others if k != j],
                                            [j])[j] for j in others])
                if sc > best_score + 1e-12:
                    best, best_score = (w, c), sc
        assert best is not None
        fs = ex._features(sets, catch, CFG, best[0])
        ref = ex._fit_score(CFG, best[1], fs, ys, others, [i])[i]
        assert abs(sel["nested_auc"][i] - ref) < 1e-10
    assert sel["nested_auc"].mean() > 0.7


def test_catch_trials_scored_held_out_and_reported() -> None:
    sets = make_synthetic_trialset(4, 100, generator="seen_late", d=1.0, seed=4)
    catch = []
    for k, s in enumerate(sets):  # stimulus-absent trials: noise, a few "seen" reports
        c = make_subject(str(s.meta["subject"].iloc[0]), "null", 40, seed=50 + k)
        catch.append(c)
    det = Detector.fit(sets, CFG, (0.2, 0.6), variant="t", catch=catch)
    c = det.performance["catch"]
    assert c["n_catch"] == 160 and 0 <= c["detector_fa_rate_at_threshold"] <= 1
    assert set(det.held_out) == {"seen", "unseen", "catch FA", "catch CR"}
    assert det.contract["alternative_rules"]["behavioural_fa"] == c["threshold_at_behavioural_fa"]


def test_ds006171_f10_saved_under_its_name(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Bug 2026-11-07: the score-loop variable shadowed `name`; F10 was saved as 'catch CR'."""
    from start_vision.decoding.detector import reliability
    from start_vision.decoding.figures import f10_calibration

    rng = np.random.default_rng(0)
    calib = reliability(rng.random(200) < 0.5, rng.random(200))
    scores = {"seen": rng.random(50), "unseen": rng.random(50), "catch CR": rng.random(20)}
    out = f10_calibration(calib, scores, tmp_path, name="F10_tms-robust")
    assert out.name == "F10_tms-robust.png" and out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["F10_tms-robust.png",
                                                         "F10_tms-robust.svg"]
