"""Step 12 remaining analyses (S2, S6, S8, R3): planted effects are found, nulls are not flagged."""
import mne
import numpy as np
import pandas as pd

from start_vision.phase3 import secondary as sc

T = np.arange(-1.0, 1.0, 1 / 128)


def _rec_subject(n: int) -> dict[str, str]:
    return {f"sub-{i:02d}_site-O1": f"{i:02d}" for i in range(n)}


def test_phase3_s2_site_difference() -> None:
    rng = np.random.default_rng(0)
    per = pd.DataFrame([{"subject": f"{s:02d}", "site": site,
                         "auc": 0.5 + (0.06 if site == "O1" else 0.0) + rng.normal(0, 0.01)}
                        for s in range(14) for site in ("O1", "O2")])
    r = sc.s2_site(per)
    assert r["n"] == 14 and r["mean_diff"] > 0.04 and r["p_two_sided"] < 0.01


def _erps(n: int, effect: float, seed: int) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    van = (T >= 0.18) & (T <= 0.28)
    out = {}
    for rid in _rec_subject(n):
        e: dict[str, np.ndarray] = {"times": T}
        for roi in ("contra", "lp"):
            base = rng.normal(0, 1, len(T))
            e[f"{roi}_unseen"] = base
            e[f"{roi}_seen"] = base + rng.normal(0, 0.5, len(T)) - effect * van * (roi == "contra")
        out[rid] = e
    return out


def test_phase3_s6_van_detected_and_null() -> None:
    rs = _rec_subject(16)
    eff, _ = sc.s6_tests(_erps(16, 2.0, 1), rs, n_perm=2000)
    van = eff[eff.test.str.startswith("S6 VAN")].iloc[0]
    assert van["mean_uv"] < -1.5 and van["p_expected_direction"] < 0.001
    null, _ = sc.s6_tests(_erps(16, 0.0, 2), rs, n_perm=2000)
    assert (null["p_two_sided"] > 0.01).all()


def _epochs(n: int, effect: float, seed: int) -> mne.EpochsArray:
    rng = np.random.default_rng(seed)
    ch = ["O1", "O2", "Pz", "C3", "C4", "Fz"]
    y = np.arange(n) % 2
    x = rng.normal(0, 5e-6, (n, len(ch), len(T)))
    x[y == 1, 0, :] += effect * 1e-6 * ((T > 0.2) & (T < 0.6))
    ep = mne.EpochsArray(x, mne.create_info(ch, 128.0, "eeg"), tmin=-1.0, verbose="error")
    ep.metadata = pd.DataFrame({"seen": y})
    return ep


def test_phase3_s8_within_detects_and_null() -> None:
    ch = ["O1", "O2", "Pz", "C3", "C4", "Fz"]
    t, a = sc.s8_sliding(_epochs(120, 8.0, 0), ch)
    post = (t > 0.25) & (t < 0.55)
    assert a[post].mean() > 0.7
    curves = {f"sub-{i:02d}_site-O1": sc.s8_sliding(_epochs(80, 0.0, 10 + i), ch)[1]
              for i in range(8)}
    _, cl = sc.s8_tests(curves, t, _rec_subject(8), n_perm=500)
    assert not (cl["p"] < 0.05).any()


def test_phase3_r3_flags_matching_map_only() -> None:
    rng = np.random.default_rng(3)
    ch = [f"O{i}" for i in range(20)] + ["C3", "C4"] + [f"F{i}" for i in range(20)]
    lat = rng.normal(size=len(ch))
    pat = np.c_[lat + rng.normal(0, 0.1, len(ch)), rng.normal(size=len(ch))]
    r = sc.r3_patterns(pat, ch, lat, np.array([0.15, 0.2]), n_null=500)
    assert r["flag"].tolist() == [True, False]


def test_phase3_s6_cluster_times_follow_planted_effect() -> None:
    """Real-data bug 2026-11-09: 1-D MNE clusters are slices; all clusters reported at -94 ms."""
    rng = np.random.default_rng(5)
    d = rng.normal(0, 1, (16, len(T)))
    d[:, (T >= 0.3) & (T <= 0.5)] += 2.0
    cl = sc.s6_clusters(T, {"LP Pz": d}, n_perm=500)
    top = cl.sort_values("p").iloc[0]
    assert top["p"] < 0.05 and 0.25 <= top["t_start"] <= 0.32 and 0.48 <= top["t_end"] <= 0.55
