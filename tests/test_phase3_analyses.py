"""Step 11 tests at the score level (blueprint §12; the full chain is tested in step 8)."""
import numpy as np
import pandas as pd

from start_vision.phase3 import analyses as an


def _scores(n_sub: int, effect: float, motor_only: bool = False, seed: int = 0,
            n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sub):
        y = (rng.random(n) < 0.5).astype(int)
        hand = rng.normal(size=n) + (2 * y - 1) * effect          # motor signal tied to label
        brain = rng.normal(size=n) + (0 if motor_only else effect) * y
        p = 1 / (1 + np.exp(-(brain + (hand if motor_only else 0))))
        p_nm = 1 / (1 + np.exp(-brain))
        rows.append(pd.DataFrame({
            "recording": f"sub-{s:02d}_site-O1", "subject": f"{s:02d}", "site": "O1",
            "seen": y, "p_seen": p, "p_seen_no_motor": p_nm, "p_seen_early": p,
            "c3c4": -hand}))
    return pd.concat(rows, ignore_index=True)


def _h1(df: pd.DataFrame) -> float:
    per = an.per_recording(df, n_perm=0)
    return an.group_one_sided(an.subject_values(per, "auc"))["p"]


def test_phase3_null_fpr() -> None:
    p = np.array([_h1(_scores(16, 0.0, seed=k)) for k in range(200)])
    rate = (p < 0.05).mean()
    assert abs(rate - 0.05) <= 0.035, rate   # 200 sims: binomial SD ~0.015


def test_phase3_effect_detected() -> None:
    assert _h1(_scores(16, 0.4, seed=1)) < 0.05


def test_phase3_motor_only_effect_flagged() -> None:
    df = _scores(16, 0.6, motor_only=True, seed=2)
    per = an.per_recording(df, n_perm=0)
    t2, t3 = an.tests(per)
    assert t2.iloc[0]["p"] < 0.05                       # the detector "separates"...
    r1 = t3[t3["control"].str.startswith("R1")].iloc[0]
    assert r1["flag"]                                   # ...but R1 flags it


def test_phase3_inclusion_rules() -> None:
    df = _scores(2, 0.0, n=60, seed=3)
    df = df[~((df.subject == "01") & (df.seen == 1) & (df.index % 2 == 0))]
    codes = pd.DataFrame({"recording": ["sub-00_site-O1", "sub-01_site-O1"],
                          "code_diff_uv": [2.0, -1.0], "code_p": [0.01, 0.2]})
    inc = an.inclusion(df, codes).set_index("recording")
    assert not inc.loc["sub-00_site-O1", "included"]          # opposite code check
    assert "class" in inc.loc["sub-01_site-O1", "reason"] or not inc.loc[
        "sub-01_site-O1", "included"]


def test_phase3_early_lesion_identity(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """With the cut after the window, the lesioned model equals the frozen detector."""
    from pathlib import Path

    import mne

    from start_vision.decoding.detector import Detector
    from start_vision.phase3 import score

    root = Path(__file__).parents[1]
    det = Detector.load(root / "detectors/sv-perception-tms-robust-v1.0.0")
    c = det.contract
    rng = np.random.default_rng(0)
    t = np.arange(-1.0, 1.0, 1 / 128)
    info = mne.create_info(c["channels"], 128.0, "eeg")
    from start_vision.decoding.channels import positions
    ep = mne.EpochsArray(rng.normal(size=(50, 64, len(t))) * 5e-6, info, tmin=-1.0,
                         verbose="error")
    ep.set_montage(mne.channels.make_dig_montage(positions(c["channels"]), coord_frame="head"))
    monkeypatch.setattr(score, "EARLY_CUT_S", 10.0)
    assert np.allclose(score.early_lesion_scores(ep, det), score.score_epochs(ep, det))
    monkeypatch.setattr(score, "EARLY_CUT_S", 0.30)
    assert not np.allclose(score.early_lesion_scores(ep, det), score.score_epochs(ep, det))


def test_gin_phosphenes_too_few_trials_excluded_not_crash() -> None:
    """Real data (sub-?? primary, 2026-11-09): a recording with 23 kept trials crashed
    sv-p3-score (ContractError). It must get NaN scores and be excluded by inclusion()."""
    from pathlib import Path

    import mne

    from start_vision.decoding.channels import positions
    from start_vision.decoding.detector import Detector
    from start_vision.phase3 import score

    root = Path(__file__).parents[1]
    det = Detector.load(root / "detectors/sv-perception-tms-robust-v1.0.0")
    c = det.contract
    n = c["min_trials_for_normalization"] - 17
    t = np.arange(-1.0, 1.0, 1 / 128)
    info = mne.create_info(c["channels"], 128.0, "eeg")
    ep = mne.EpochsArray(np.random.default_rng(0).normal(size=(n, 64, len(t))) * 5e-6, info,
                         tmin=-1.0, verbose="error")
    ep.set_montage(mne.channels.make_dig_montage(positions(c["channels"]), coord_frame="head"))
    p, p_nm, p_early = score.recording_scores(ep, det)
    assert len(p) == n and np.isnan(p).all() and np.isnan(p_nm).all() and np.isnan(p_early).all()
    df = pd.DataFrame({"recording": "sub-99_site-O1", "subject": "99", "site": "O1",
                       "seen": np.arange(n) % 2, "p_seen": p})
    inc = an.inclusion(df)
    assert not inc["included"].iloc[0] and "kept <" in inc["reason"].iloc[0]
