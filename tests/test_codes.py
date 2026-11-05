"""codes: composite code round-trip and the eegunify mapping."""

from __future__ import annotations

import pytest

from start_vision import codes as C  # noqa: N812


def test_round_trip_every_code() -> None:
    for v in C.all_codes():
        if v - C.RESPONSE_OFFSET in C.RESPONSES:
            continue
        c = C.decode(v)
        assert C.encode(c.cls, c.side, c.orient, c.matched) == v


def test_mapping_unique_and_complete() -> None:
    m = C.eegunify_mapping()
    assert len(m) == len(set(m.values())) == len(C.all_codes())
    assert m["1111"] == "stimulus/seen/left/vertical/m1"
    assert m["111"] == "stimulus/seen/left/vertical/m0"
    assert m["310"] == "stimulus/fa/left" and m["420"] == "stimulus/cr/right"
    assert m["5124"] == "response/seen"


def test_invalid_codes_rejected() -> None:
    with pytest.raises(ValueError):
        C.encode(C.FA, C.LEFT, C.VERTICAL, False)
    with pytest.raises(ValueError):
        C.encode(C.CR, C.LEFT, C.NO_GABOR, True)
    with pytest.raises(ValueError):
        C.decode(1310)


def test_ds006171_glitch_codes_unknown() -> None:
    # sub-23 nocue Status: a stray 6; events.tsv: single 40 / 33 / 41 / 255 (Step 3-4 checks)
    for task, bad in (("nocue", (6, 40)), ("noninformative", (33, 41, 255)), ("informative", (6,))):
        assert not set(bad) & C.known_codes(task)


def test_tasks_cover_both_sides_and_orientations() -> None:
    for task in C.TASKS:
        g = [v for v in C.RAW_STIMULI[task].values() if not v.catch]
        sides = (C.SIDE_UNKNOWN,) if task == "noninformative" else (C.LEFT, C.RIGHT)
        assert {(v.side, v.orient) for v in g} == {(s, o) for s in sides for o in (1, 2)}


def test_noninformative_side_unknown_not_in_left_right() -> None:
    v = C.encode(C.SEEN, C.SIDE_UNKNOWN, C.VERTICAL, True)
    assert C.trial_type(v) == "stimulus/seen/sideunknown/vertical/m1"
    assert "left" not in C.trial_type(v) and "right" not in C.trial_type(v)
