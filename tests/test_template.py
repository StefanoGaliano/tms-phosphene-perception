"""configs/template_ds006171.yaml: the channel rename works on both BDF header layouts."""

from __future__ import annotations

from pathlib import Path

import mne
import yaml

from eegunify.channels.naming import normalize_names

TEMPLATE = Path(__file__).parents[1] / "configs" / "template_ds006171.yaml"
AUX = ["GSR1", "GSR2", "Erg1", "Erg2", "Resp", "Plet", "Temp", "Status"]


def _kept(names: list[str]) -> list[str]:
    ch = yaml.safe_load(TEMPLATE.read_text())["channels"]
    m = normalize_names(names, rename=ch["rename"], protect=["Status"])
    return [v for v in m.values() if v not in ch["drop"]]


def _header(extra_banks: bool) -> list[str]:
    banks = "ABCDEFGH" if extra_banks else "ABCD"
    return [f"{b}{i}" for b in banks for i in range(1, 33)] + [f"EXG{i}" for i in range(1, 9)] + AUX


def test_ds006171_272_channel_header_no_collision() -> None:
    std = set(mne.channels.make_standard_montage("standard_1005").ch_names)
    for extra in (False, True):
        kept = _kept(_header(extra))
        assert len(kept) == len(set(kept)) == 133
        assert sum(k in std for k in kept) == 128
        assert {"VEOG1", "VEOG2", "HEOG1", "HEOG2", "Status"} <= set(kept)
