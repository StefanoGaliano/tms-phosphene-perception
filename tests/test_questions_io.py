"""Step 3: loaders, output guard (D6), and the real inputs' row counts (marked real)."""
from pathlib import Path

import pandas as pd
import pytest

from start_vision.questions import io

ROOT = Path(__file__).parents[1]


def test_questions_out_dir_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        io.out_dir(tmp_path, "q01_q03_eklund")
    d = io.out_dir(tmp_path, "q01")
    (d / "x.tsv").write_text("a")
    with pytest.raises(FileExistsError):
        io.out_dir(tmp_path, "q01")
    assert not any(io.out_dir(tmp_path, "q01", force=True).iterdir())


@pytest.mark.real
def test_questions_real_inputs_row_counts() -> None:
    e = io.eklund_trials(ROOT)
    raw = pd.read_csv(ROOT / "results/eklund2018/trial_scores.tsv.gz", sep="\t")
    assert len(e) == (raw["cls"].isin(["aware", "unaware", "unaware_incorrect", "aware_incorrect"])
                      & raw["exclusion"].isna()).sum()
    d = io.ds006171_trials(ROOT)
    loso = pd.read_csv(ROOT / "derivatives/sv_decode_group/loso_predictions.tsv", sep="\t")
    assert len(d) == len(loso) and d["subject"].nunique() == 34
    assert d["gabor_contrast"].notna().all()
    g = io.gin_scores(ROOT)
    inc = pd.read_csv(ROOT / "results/phase3/inclusion_primary.tsv", sep="\t")
    assert g["recording"].nunique() == int(inc["included"].sum())
