"""Step 8c: Q38 (P§9) - the symmetric hand covariate removes a lateralized label-tied signal and
keeps a bilateral posterior effect."""
import numpy as np
import pandas as pd

from start_vision.questions import q38

T = np.arange(-1.0, 1.0, 1 / 128)
CH = ["C3", "C4", "P7", "P8", "PO7", "PO8", "O1", "O2", "Oz", "Pz", "Cz"]


def test_questions_q38_pairs_from_contract_names() -> None:
    p = q38.mirror_pairs(["Fp1", "Fp2", "C3", "C4", "P9", "P10", "Oz", "Cz", "T7"])
    assert ("Fp1", "Fp2") in p and ("P9", "P10") in p and ("C3", "C4") in p
    assert all("z" not in a + b for a, b in p) and len(p) == 3   # T7 has no mirror here


def _data(hand: bool, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 400
    y = (rng.random(n) < 0.5).astype(int)
    x = rng.normal(0, 1, (n, len(CH), len(T)))
    w = (T > 0.2) & (T < 0.6)
    eff = rng.normal(1.0, 0.3, n) * y
    if hand:   # left negative, right positive, tied to the label
        for c, s in (("C3", -1), ("C4", 1), ("P7", -1), ("P8", 1)):
            x[:, CH.index(c), :] += s * eff[:, None] * w
    else:      # bilateral posterior: same on both sides
        for c in ("PO7", "PO8", "O1", "O2", "Oz"):
            x[:, CH.index(c), :] += eff[:, None] * w
    sig = x[:, [CH.index(c) for c in ("C3", "P7", "PO7", "O1")]][..., w].mean((1, 2)) \
        - (x[:, [CH.index(c) for c in ("C4", "P8", "PO8", "O2")]][..., w].mean((1, 2)) if hand
           else -x[:, [CH.index(c) for c in ("PO8", "O2", "Oz")]][..., w].mean((1, 2)))
    # near-linear dependence, as on real data (H1 AUC 0.54): the prereg residualization is linear
    # (as A1), so a steep nonlinear dependence would not be fully removed
    p = 0.5 + 0.1 * (-sig if hand else sig) + rng.normal(0, 0.1, n)
    return x, pd.DataFrame({"seen": y, "p_seen": p})


def test_questions_q38_hand_covariate() -> None:
    xh, gh = _data(hand=True, seed=1)
    rh = q38.recording_values(gh.assign(hand=q38.hand_index(xh, CH, T)))
    assert rh["auc_frozen"] > 0.6 and abs(rh["auc_res"] - 0.5) < 0.06 and rh["auc_hand"] < 0.4
    xb, gb = _data(hand=False, seed=2)
    rb = q38.recording_values(gb.assign(hand=q38.hand_index(xb, CH, T)))
    assert rb["auc_frozen"] > 0.6 and rb["auc_res"] > 0.6
