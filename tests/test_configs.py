"""configs: generated configs are valid eegunify configs; noninformative has no lateral parts."""

from __future__ import annotations

from pathlib import Path

import yaml

from eegunify.config.loader import load_config
from start_vision.configs import render, write_all

ROOT = Path(__file__).parents[1]


def test_generated_configs_validate(tmp_path: Path) -> None:
    inv = tmp_path / "inv.tsv"
    inv.write_text(
        "subject\tnocue\tnoninformative\tinformative\n01\tTrue\tTrue\tFalse\n02\tTrue\tFalse\tTrue\n"
    )
    paths = write_all(ROOT / "configs" / "template_ds006171.yaml", inv, tmp_path)
    for p in paths:
        cfg = load_config(p)
        assert cfg.dataset.task in p.name
    nocue = yaml.safe_load(paths[0].read_text())
    assert nocue["recordings"]["subjects"] == ["01", "02"]
    assert nocue["events"]["mapping"]["5124"] == "response/seen"


def test_noninformative_without_lateral() -> None:
    t = (ROOT / "configs" / "template_ds006171.yaml").read_text()
    cfg = render(t, "noninformative", ["01"])
    assert "seen_lat" not in cfg["contrasts"] and "seen_left" not in cfg["conditions"]
    assert "seen_lat" in render(t, "nocue", ["01"])["contrasts"]
