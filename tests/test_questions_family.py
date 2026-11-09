"""Step 9: family Holm, and sv-questions end to end on a synthetic root (Q01-Q04, Q37b)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from start_vision.decoding import stats as dstats
from start_vision.questions import family, io
from start_vision.questions.params import FAMILY
from tests.synthetic.questions import ds006171_table, eklund_table, gin_scores_table


def test_questions_family_holm_and_missing(tmp_path: Path) -> None:
    ps = [0.001, 0.04, 0.2, 0.01, 0.03, 0.5, 0.002]
    for q, p in zip(FAMILY, ps, strict=True):
        d = tmp_path / io.QROOT / f"prereg_{q}"
        d.mkdir(parents=True)
        (d / "primary.json").write_text(json.dumps({"question": q, "test": "t", "tail": "x",
                                                    "n": 10.0, "mean": 0.1, "p": p}))
    t = family.family_table(tmp_path)
    assert np.allclose(t["p_holm"], dstats.holm(np.array(ps)))
    (tmp_path / io.QROOT / "prereg_q38" / "primary.json").unlink()
    with pytest.raises(FileNotFoundError):
        family.family_table(tmp_path)


def _synthetic_root(r: Path) -> None:
    e = eklund_table(gabors=("large", "small"), correct_effect=0.05, seed=11)
    e["exclusion"] = np.nan
    (r / "results/eklund2018").mkdir(parents=True)
    e.to_csv(r / "results/eklund2018/trial_scores.tsv.gz", sep="\t", index=False)
    ds = ds006171_table(n_subjects=6, seed=12)
    (r / "derivatives/sv_decode_group").mkdir(parents=True)
    ds[["subject", "task", "trial_number", "seen", "score"]].to_csv(
        r / "derivatives/sv_decode_group/loso_predictions.tsv", sep="\t", index=False)
    for (s, t), g in ds.groupby(["subject", "task"]):
        d = r / f"sourcedata/staged/ds006171/sub-{s}/eeg"
        d.mkdir(parents=True, exist_ok=True)
        side = g["side"].where(g["side"] != "sideunknown", "sideunknown")
        pd.DataFrame({"trial_number": g["trial_number"],
                      "trial_type": "stimulus/" + np.where(g["seen"], "seen", "unseen") + "/"
                      + side + "/vertical/m1",
                      "gabor_contrast": g["gabor_contrast"],
                      "objective_correct": g["correct"].map({1.0: True, 0.0: False}).fillna("n/a")
                      }).to_csv(d / f"sub-{s}_task-{t}_events.tsv", sep="\t", index=False)
    gs = gin_scores_table(n_subjects=6, seed=13)
    (r / "results/phase3").mkdir(parents=True)
    gs.to_csv(r / "results/phase3/scores_primary.tsv", sep="\t", index=False)
    inc = gs.groupby("recording").first().reset_index()[["recording", "subject", "site"]]
    inc.assign(included=True).to_csv(r / "results/phase3/inclusion_primary.tsv", sep="\t",
                                     index=False)


def test_questions_cli_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from start_vision.cli import questions

    _synthetic_root(tmp_path)
    for q in ("q01", "q02", "q03", "q04", "q37b"):
        monkeypatch.setattr(sys, "argv", ["sv-questions", q, "--root", str(tmp_path)])
        questions()
        d = tmp_path / io.QROOT / f"prereg_{q}"
        for f in ("README.md", "group.tsv", "primary.json", "figure.png"):
            assert (d / f).exists(), (q, f)
    assert (tmp_path / io.QROOT / "prereg_q37b" / "k_star.json").exists()
    monkeypatch.setattr(sys, "argv", ["sv-questions", "q01", "--root", str(tmp_path)])
    with pytest.raises(FileExistsError):
        questions()
