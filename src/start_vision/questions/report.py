"""README writer per question (blueprint §8)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from start_vision.questions.io import commit_hash

PREREG = "docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_readme(d: Path, root: Path, title: str, section: str, inputs: list[str],
                 tables: dict[str, pd.DataFrame], interpretation: str, limits: list[str],
                 deviations: str = "none") -> Path:
    lines = [f"# {title}", "",
             f"Pre-registered run ({section} of {PREREG}); {date.today().isoformat()}; code "
             f"commit `{commit_hash(root)}`. Holm p for the [P] test: `../prereg_family/`.", "",
             "Inputs: " + "; ".join(inputs), ""]
    for name, t in tables.items():
        lines += [f"## {name}", "", _md(t), ""]
    lines += ["## Interpretation (rule fixed in the prereg)", "", interpretation, "",
              "## Stated limits", "", *[f"- {x}" for x in limits], "",
              f"## Deviations\n\n{deviations}", "", "Figure: `figure.png`.", ""]
    p = d / "README.md"
    p.write_text("\n".join(lines))
    return p


def group_row(label: str, role: str, r: dict[str, float]) -> dict[str, object]:
    return {"test": label, "role": role, **{k: r.get(k) for k in
            ("n", "mean", "ci_low", "ci_high", "d_z", "p", "tail", "n_positive")}}
