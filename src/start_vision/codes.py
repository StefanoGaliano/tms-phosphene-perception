"""Raw ds006171 trigger codes and the composite code written to the staged events.tsv.

Composite value (PHASE1_BLUEPRINT section 4): ``1000*M + 100*C + 10*H + O``

- C class: 1 seen Gabor, 2 unseen Gabor, 3 catch false alarm, 4 catch correct rejection,
  9 excluded (no photodiode, no report).
- H side: 1 left, 2 right (Gabor hemifield; for catch trials the cue side), 0 unknown
  (noninformative task, user decision 6).
- O orientation: 1 vertical, 2 horizontal, 0 catch.
- M matched: 1 in the contrast-matched subset, 0 not (catch and excluded: always 0).

Responses are written as ``5000 + raw code`` (5121, 5122, 5124, 5125): the raw 121/122 would
collide with the composite seen/right/vertical|horizontal/m0 codes (a clash in the blueprint's
section 4 scheme, found by test_codes).

Raw codes per task are in ``RAW_STIMULI``. Sources: ``events.json`` (nocue codes, also used by
informative) and docs/DATASETS.md "Step 4 code inference" for the undocumented
``noninformative`` codes.
"""

from __future__ import annotations

from dataclasses import dataclass

SEEN, UNSEEN, FA, CR, EXCLUDED = 1, 2, 3, 4, 9
LEFT, RIGHT, SIDE_UNKNOWN = 1, 2, 0
VERTICAL, HORIZONTAL, NO_GABOR = 1, 2, 0

PHOTODIODE = 128
REPORT_SEEN, REPORT_UNSEEN = 124, 125
ORIENT_VERTICAL, ORIENT_HORIZONTAL = 121, 122
CUE_LEFT, CUE_RIGHT = 1, 2
RESPONSE_OFFSET = 5000  # staged response value = 5000 + raw code (see module docstring)
RESPONSES = {
    REPORT_SEEN: "response/seen",
    REPORT_UNSEEN: "response/unseen",
    ORIENT_VERTICAL: "response/vertical",
    ORIENT_HORIZONTAL: "response/horizontal",
}
TASKS = ("nocue", "noninformative", "informative")


@dataclass(frozen=True)
class RawStim:
    """What a raw stimulus code means: Gabor side and orientation, or a catch trial."""

    side: int  # Gabor hemifield; for catch, the cue side
    orient: int  # VERTICAL, HORIZONTAL, or NO_GABOR for catch
    cue_side: int | None = None  # None when the task has no cue

    @property
    def catch(self) -> bool:
        return self.orient == NO_GABOR

    @property
    def cue_valid(self) -> bool | None:
        if self.cue_side is None or self.catch:
            return None
        return self.cue_side == self.side


_DOCUMENTED = {
    11: RawStim(LEFT, VERTICAL),
    12: RawStim(LEFT, HORIZONTAL),
    21: RawStim(RIGHT, VERTICAL),
    22: RawStim(RIGHT, HORIZONTAL),
    31: RawStim(LEFT, NO_GABOR),
    32: RawStim(RIGHT, NO_GABOR),
}

# Informative task: same codes, cue always valid (1 -> 11/12/31, 2 -> 21/22/32; checked on all
# subjects, docs/DATASETS.md).
_INFORMATIVE = {
    k: RawStim(v.side, v.orient, cue_side=v.side) for k, v in _DOCUMENTED.items()
}

# Noninformative task (undocumented; docs/DATASETS.md "Step 4 code inference"). Orientation is
# known (odd = vertical, from the orientation answers); the Gabor hemifield is NOT: neither
# reading of the codes was confirmed by the EEG, so the side is SIDE_UNKNOWN (user decision 6,
# 2026-11-05: noninformative is left out of lateralized analyses). The raw code is kept in the
# staged events.tsv so a later side mapping can be applied without restaging logic changes.
_NONINFORMATIVE = {
    k: RawStim(SIDE_UNKNOWN, VERTICAL if k % 2 else HORIZONTAL)
    for k in (101, 102, 103, 104, 111, 112, 113, 114)
} | {98: RawStim(SIDE_UNKNOWN, NO_GABOR), 99: RawStim(SIDE_UNKNOWN, NO_GABOR)}

RAW_STIMULI: dict[str, dict[int, RawStim]] = {
    "nocue": _DOCUMENTED,
    "informative": _INFORMATIVE,
    "noninformative": _NONINFORMATIVE,
}
CUES: dict[str, tuple[int, ...]] = {
    "nocue": (),
    "informative": (CUE_LEFT, CUE_RIGHT),
    "noninformative": (CUE_LEFT, CUE_RIGHT),
}


def known_codes(task: str) -> frozenset[int]:
    """Every raw code a task legitimately uses; anything else is a trigger glitch."""
    return frozenset(
        [*RAW_STIMULI[task], *CUES[task], PHOTODIODE, *RESPONSES]
    )


@dataclass(frozen=True)
class Code:
    cls: int
    side: int
    orient: int
    matched: bool


def encode(cls: int, side: int, orient: int, matched: bool) -> int:
    if cls not in (SEEN, UNSEEN, FA, CR, EXCLUDED):
        raise ValueError(f"bad class {cls}")
    if side not in (LEFT, RIGHT, SIDE_UNKNOWN) or orient not in (VERTICAL, HORIZONTAL, NO_GABOR):
        raise ValueError(f"bad side/orient {side}/{orient}")
    gabor = cls in (SEEN, UNSEEN)
    if gabor == (orient == NO_GABOR) and cls != EXCLUDED:
        raise ValueError(f"class {cls} does not fit orientation {orient}")
    if matched and not gabor:
        raise ValueError("only Gabor trials can be matched")
    return 1000 * int(matched) + 100 * cls + 10 * side + orient


def decode(value: int) -> Code:
    m, rest = divmod(value, 1000)
    cls, rest = divmod(rest, 100)
    side, orient = divmod(rest, 10)
    code = Code(cls, side, orient, bool(m))
    if m not in (0, 1) or encode(cls, side, orient, bool(m)) != value:
        raise ValueError(f"not a composite code: {value}")
    return code


_CLASS = {SEEN: "seen", UNSEEN: "unseen", FA: "fa", CR: "cr", EXCLUDED: "excluded"}
_SIDE = {LEFT: "left", RIGHT: "right", SIDE_UNKNOWN: "sideunknown"}
_ORIENT = {VERTICAL: "vertical", HORIZONTAL: "horizontal"}


def trial_type(value: int) -> str:
    """Composite code -> eegunify trial type (``/``-separated MNE tags)."""
    if value - RESPONSE_OFFSET in RESPONSES:
        return RESPONSES[value - RESPONSE_OFFSET]
    c = decode(value)
    if c.cls == EXCLUDED:
        return f"excluded/{_SIDE[c.side]}/{_ORIENT.get(c.orient, 'catch')}"
    if c.cls in (FA, CR):
        return f"stimulus/{_CLASS[c.cls]}/{_SIDE[c.side]}"
    return f"stimulus/{_CLASS[c.cls]}/{_SIDE[c.side]}/{_ORIENT[c.orient]}/m{int(c.matched)}"


def all_codes() -> list[int]:
    out = [
        encode(cls, side, orient, m)
        for cls in (SEEN, UNSEEN)
        for side in (LEFT, RIGHT, SIDE_UNKNOWN)
        for orient in (VERTICAL, HORIZONTAL)
        for m in (False, True)
    ]
    sides = (LEFT, RIGHT, SIDE_UNKNOWN)
    out += [encode(cls, side, NO_GABOR, False) for cls in (FA, CR) for side in sides]
    out += [encode(EXCLUDED, side, o, False) for side in sides for o in (0, 1, 2)]
    return out + [RESPONSE_OFFSET + r for r in sorted(RESPONSES)]


def eegunify_mapping() -> dict[str, str]:
    """Explicit ``events.mapping`` for the eegunify config (``unmapped: error``)."""
    return {str(v): trial_type(v) for v in all_codes()}
