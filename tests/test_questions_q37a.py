"""Step 8: Q37a (P§6) - projection coefficient and residualization."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from start_vision.phase3.analyses import rec_auc
from start_vision.questions import q37a

T = np.arange(-0.5, 1.0, 1 / 128)


def test_questions_q37a_projection_exact_and_noisy() -> None:
    rng = np.random.default_rng(0)
    tpl = rng.normal(size=(8, len(T)))
    a = rng.uniform(0.5, 1.5, 60)
    x = a[:, None, None] * tpl[None]
    amp = q37a.tep_amplitude(x, T)
    assert np.allclose(amp / amp.mean(), a / a.mean(), atol=1e-10)
    assert np.isclose(amp.mean(), 1.0)
    noisy = q37a.tep_amplitude(x + rng.normal(0, 0.3, x.shape), T)
    assert np.corrcoef(noisy, a)[0, 1] > 0.99


def test_questions_q37a_residualization_removes_amplitude_only_effect() -> None:
    rng = np.random.default_rng(1)
    n = 400
    y = (rng.random(n) < 0.5).astype(int)
    amp = 1 + 0.3 * y + rng.normal(0, 0.2, n)
    p = 0.5 + 0.2 * (amp - 1) + rng.normal(0, 0.02, n)
    g = pd.DataFrame({"seen": y, "amp": amp, "p_seen": p})
    r = q37a.recording_values(g)
    assert r["auc_frozen"] > 0.6 and r["auc_amp"] > 0.6 and abs(r["auc_res"] - 0.5) < 0.06
    g2 = g.assign(p_seen=0.5 + 0.1 * y + rng.normal(0, 0.05, n))
    assert q37a.recording_values(g2)["auc_res"] > 0.6


def test_questions_q37a_template_label_free() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 4, len(T)))
    assert np.array_equal(q37a.tep_amplitude(x, T), q37a.tep_amplitude(x.copy(), T))


@pytest.mark.real
def test_questions_q37a_reproduces_stored_h1() -> None:
    from start_vision.phase3.analyses import group_one_sided, subject_values
    from start_vision.questions import io

    root = Path(__file__).parents[1]
    s = io.gin_scores(root)
    per = pd.DataFrame([{"recording": r, "subject": g["subject"].iloc[0],
                         "auc": rec_auc(g, "p_seen")} for r, g in s.groupby("recording")])
    h1 = pd.read_csv(root / "results/phase3/t2_tests_primary.tsv", sep="\t").set_index("test")
    assert np.isclose(group_one_sided(subject_values(per, "auc"))["mean"],
                      h1.loc["H1 primary", "mean"], atol=1e-9)
