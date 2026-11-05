"""Step 1 acceptance: the package imports next to the pinned eegunify."""

from __future__ import annotations

import eegunify
import start_vision


def test_eegunify_is_the_frozen_version() -> None:
    assert eegunify.__version__ == "0.9.2"


def test_start_vision_imports() -> None:
    assert start_vision.__version__
