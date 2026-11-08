"""Phase 3 config (configs/phase3.yaml), frozen pydantic models; hash feeds the stage key."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class _F(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Codes(_F):
    present: str
    absent: str


class Pulse(_F):
    cut_ms: tuple[float, float]
    interp: str
    artifact_k: float
    artifact_hold_ms: float
    max_cut_ms: float
    extend_cut: bool


class DecayIca(_F):
    enabled: bool
    epoch_s: tuple[float, float]
    fit_highpass_hz: float
    window_ms: tuple[float, float]
    ratio_threshold: float
    method: str
    seed: int


class Epochs(_F):
    window_s: tuple[float, float]


class Rejection(_F):
    ptp_uv: float
    ptp_windows_s: tuple[tuple[float, float], ...]
    ptp_win_s: float
    ptp_step_s: float
    heog_step_uv: float
    heog_window_s: tuple[float, float]
    heog_win_s: float
    heog_step_s: float


class Pipeline(_F):
    muscle_ica: bool = False
    reference: str = "average"
    rough: bool = False


class Phase3Config(_F):
    data_dir: Path
    index_dir: Path
    out_dir: Path
    detector: Path
    eegunify_template: Path
    codes: Codes
    pulse: Pulse
    ica_sfreq: float
    decay_ica: DecayIca
    epochs: Epochs
    rejection: Rejection
    lowpass_hz: float
    eog_bipolar: dict[str, tuple[str, str]]
    drop_channels: tuple[str, ...]
    pipelines: dict[str, Pipeline]


def load(path: Path) -> Phase3Config:
    return Phase3Config.model_validate(yaml.safe_load(path.read_text()))
