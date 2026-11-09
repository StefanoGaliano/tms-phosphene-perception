"""Family step (P§1 + amendments): Holm over the seven [P] tests from prereg_*/primary.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from start_vision.decoding import stats as dstats
from start_vision.questions import io
from start_vision.questions.params import FAMILY, FAMILY_ALPHA
from start_vision.questions.report import _md, write_readme


def family_table(root: Path) -> pd.DataFrame:
    rows = []
    for q in FAMILY:
        f = root / io.QROOT / f"prereg_{q}" / "primary.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run sv-questions {q} first (all seven needed)")
        rows.append(json.loads(f.read_text()))
    t = pd.DataFrame(rows)
    t["p_holm"] = dstats.holm(t["p"].to_numpy(float))
    t["reject"] = t["p_holm"] < FAMILY_ALPHA
    return t


def run(root: Path, force: bool = False) -> Path:
    t = family_table(root)
    d = io.out_dir(root, "family", force)
    t.to_csv(d / "family.tsv", sep="\t", index=False, float_format="%.5f")
    for q, ph in zip(t["question"], t["p_holm"], strict=True):
        readme = root / io.QROOT / f"prereg_{q}" / "README.md"
        if readme.exists():
            txt = readme.read_text()
            if "Holm p (family of seven)" not in txt:
                readme.write_text(txt.replace("Holm p for the [P] test: `../prereg_family/`.",
                                              f"Holm p (family of seven) for the [P] test: "
                                              f"{ph:.4g} (`../prereg_family/`).", 1))
    cols = [c for c in ("question", "test", "tail", "n", "mean", "ci_low", "ci_high", "p",
                        "p_holm", "reject") if c in t]
    write_readme(d, root, "Questions_to_solve: family of primary tests (Holm)", "§1",
                 [f"prereg_{q}/primary.json" for q in FAMILY], {"Family": t[cols]},
                 f"{int(t['reject'].sum())} of {len(t)} primary tests rejected at Holm "
                 f"alpha {FAMILY_ALPHA}. Q04's p is an equivalence p: rejection = equivalence "
                 "shown.",
                 ["Specified after the first look (P§0); run once."])
    (d / "article_notes_block.md").write_text(_md(t[cols]) + "\n")
    return d
