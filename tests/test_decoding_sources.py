"""Phase 2 build step 16: A12 adapters (interfaces) and a cross-dataset transfer through them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from start_vision.decoding import across, channels, sources
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from tests.synthetic.decoding import make_subject

CFG = DecodingConfig(n_jobs=1)


class _SyntheticAdapter:
    """Stands in for a second dataset: its own channel count, sfreq and epoch window."""

    modality: sources.Modality = "visual"

    def __init__(self, name: str, n_channels: int, sfreq: float, window_s: tuple[float, float],
                 seed: int) -> None:
        self.name, self.n_channels, self.sfreq = name, n_channels, sfreq
        self.window_s, self.seed = window_s, seed

    def subjects(self, root: Path) -> list[str]:
        return ["01", "02", "03"]

    def load(self, subject: str, root: Path) -> TrialSet:
        ts = make_subject(subject, "seen_late", 150, n_channels=self.n_channels,
                          sfreq=self.sfreq, window_s=self.window_s,
                          seed=self.seed + int(subject))
        meta = ts.meta.assign(dataset=self.name)
        return TrialSet(ts.X, ts.times, ts.ch_names, ts.sfreq, meta, ts.info)


def test_adapters_follow_the_protocol() -> None:
    for name, cls in sources.ADAPTERS.items():
        a = cls()
        assert isinstance(a, sources.DatasetAdapter) and a.name == name
        assert a.modality in ("visual", "tactile", "tms-visual")
    assert isinstance(_SyntheticAdapter("x", 16, 128.0, (-0.5, 1.0), 0), sources.DatasetAdapter)


@pytest.mark.parametrize("cls", [sources.Eklund2018Adapter, sources.Ds005273Adapter,
                                 sources.TactileAdapter, sources.TmsPhospheneAdapter])
def test_stubs_say_what_is_missing(cls: type, tmp_path: Path) -> None:
    a = cls()
    for call in (lambda: a.subjects(tmp_path), lambda: a.load("01", tmp_path)):
        with pytest.raises(NotImplementedError, match=r"A12_OPEN_ISSUES\.md #"):
            call()


def test_a12_transfer_across_two_synthetic_datasets(tmp_path: Path) -> None:
    """Train on dataset A (32 ch, 256 Hz), test on dataset B (16 ch, 128 Hz, later window)."""
    a = _SyntheticAdapter("A", 32, 256.0, (-0.5, 1.0), seed=100)
    b = _SyntheticAdapter("B", 16, 128.0, (-0.4, 0.9), seed=200)
    sa = [a.load(s, tmp_path) for s in a.subjects(tmp_path)]
    sb = [b.load(s, tmp_path) for s in b.subjects(tmp_path)]
    h = channels.harmonize(sa + sb)
    train, test = h[:3], h[3:]
    assert train[0].ch_names == test[0].ch_names and train[0].sfreq == test[0].sfreq == 128.0
    df = across.transfer(train, test, CFG)
    assert list(df.dataset) == ["B"] * 3
    late = (train[0].times >= 0.32) & (train[0].times <= 0.48)
    assert all(np.asarray(auc)[late].mean() > 0.6 for auc in df.auc)
    with pytest.raises(ValueError):  # not harmonized
        across.transfer(sa, sb, CFG)
