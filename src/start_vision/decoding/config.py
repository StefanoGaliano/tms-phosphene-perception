"""DecodingConfig (PHASE2 §13 step 2): every default is a value written in the blueprint.

Section references are to plans/PHASE2_BLUEPRINT.md. Frozen; ``payload()`` is the dict that
goes into stage hashing and provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

Window = tuple[float, float]


class DecodingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 97
    n_jobs: int = 1
    tasks: tuple[str, ...] = ("nocue", "noninformative", "informative")
    # data (D4, §4): Phase 1 `stimraw` epochs (256 Hz, no baseline) -> 40 Hz low-pass -> 128 Hz
    epoch_set: str = "stimraw"
    lowpass_hz: float = 40.0
    sfreq: float = 128.0
    window_s: Window = (-0.5, 1.0)
    baseline_s: Window = (-0.2, 0.0)  # §4 "Phase 2 applies its own (-200..0 ms)"
    # labels (§6)
    min_trials_per_class: int = 30
    n_contrast_bins: int = 3
    min_trials_per_bin: int = 10
    # classifier and CV (D5, A1)
    C: float = 1.0
    n_folds: int = 5
    n_repeats: int = 5
    # statistics (D6, D7)
    n_permutations: int = 10_000
    cluster_p: float = 0.05
    n_label_permutations: int = 200
    n_bootstrap: int = 10_000
    # features (§9, C4, A7)
    norm_window_s: Window = (-0.5, -0.1)
    feature_bin_s: float = 0.05
    candidate_windows_s: tuple[Window, ...] = (
        (0.15, 0.6), (0.2, 0.6), (0.2, 0.8), (0.3, 0.8)
    )
    C_grid: tuple[float, ...] = (0.01, 0.1, 1.0)  # A11 inner-LOSO choice (set 2026-11-05)
    t_min_robust_s: float = 0.15  # D9 placeholder (open question Q3)
    alpha_band_hz: Window = (8.0, 13.0)
    alpha_peak_halfwidth_hz: float = 2.0
    alpha_window_s: Window = (-0.5, -0.1)
    alpha_psd_window_s: Window = (-1.0, 0.0)
    # equivalence (A8; open question Q5)
    tost_bound_auc: float = 0.03
    # detector (§9)
    max_missing_channels: int = 4
    min_trials_for_normalization: int = 40
    extra: dict[str, Any] = Field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load(path: Path) -> DecodingConfig:
    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    section: dict[str, Any] = data.get("decoding") or {}
    return DecodingConfig(**section)
