"""Step 2 acceptance: the awareness components load in eegunify with the sourced values."""

from __future__ import annotations

from pathlib import Path

from eegunify.erp.components.library import load_library

YAML = Path(__file__).resolve().parents[1] / "configs" / "components" / "awareness.yaml"


def test_awareness_components_load_with_sourced_values() -> None:
    lib = load_library([YAML])
    van, lp = lib["van@awareness"], lib["lp@awareness"]
    assert van.window_ms == (180.0, 280.0)  # Eklund & Wiens 2018, analyze_VAN.m:26
    assert lp.window_ms == (350.0, 550.0)  # analyze_VAN.m:29
    assert van.polarity == "negative" and lp.polarity == "positive"


def test_no_source_placeholders_left_in_configs() -> None:
    for f in YAML.parents[1].rglob("*.yaml"):
        assert "TODO_FROM_SOURCE" not in f.read_text(), f
