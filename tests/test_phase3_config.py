from pathlib import Path

from start_vision.phase3 import config

ROOT = Path(__file__).parents[1]


def test_phase3_config_loads() -> None:
    c = config.load(ROOT / "configs" / "phase3.yaml")
    assert c.pulse.cut_ms == (-2.0, 10.0)
    assert c.decay_ica.window_ms == (11.0, 30.0) and c.decay_ica.ratio_threshold == 8.0
    assert set(c.pipelines) == {"primary", "paperlike", "gonogo"}
    assert not c.pipelines["primary"].muscle_ica
