"""Every pre-registered number, with its section of docs/QUESTIONS_PREREG.md (P§n).

Frozen by the tags questions-prereg, -amend1, -amend2; not configurable.
"""
from __future__ import annotations

SEED = 20261109                 # P§1
N_PERM = 10_000                 # P§1 sign flip
N_BOOT = 10_000                 # P§1 bootstrap
TOST_BOUND = 0.03               # P§1 (PROJECT CHOICE, Phase 2 A8 precedent)
MIN_CELL = 3                    # P§1
FAMILY_ALPHA = 0.05             # P§1
CHANCE_EKLUND = 0.25            # P§4 four quadrants
CHANCE_DS006171 = 0.5           # P§4 orientation
MATCH_BINS = 10                 # P§5 (start_vision.matching.match defaults)
MATCH_SEED = 97                 # P§5
TEMPLATE_WINDOW_S = (0.0, 0.6)  # P§6
Q37B_KS = (1, 5, 10, 20)        # P§7
N_BLOCKS = 1000                 # P§7 (as the first look)
K_STAR_AUC = 0.75               # P§7
HAND_WINDOW_S = (0.15, 0.6)     # P§9 (the detector window)
HAND_BASELINE_S = (-0.2, -0.01)  # P§9 (contract baseline)
VAN_OZ_WINDOW_S = (0.18, 0.22)  # P§9 descriptive (Samaha 2017 Methods)
LP_SAMAHA_WINDOW_S = (0.3, 0.8)  # P§9 descriptive (Samaha 2017 Methods)
S6_VAN_WINDOW_S = (0.18, 0.28)  # P§9 descriptive (PHASE1_REFERENCES)
EKLUND_CRITICAL = ("aware", "unaware", "unaware_incorrect", "aware_incorrect")  # P§2

FIRST_LOOK_DIRS = ("q01_q03_eklund", "saved_results", "q37ab_phase3", "q03_where_eklund",
                   "van_topography_explore")  # P§0: never written by the prereg run
FAMILY = ("q01", "q02", "q03", "q04", "q37a", "q37b", "q38")  # P§1 + amendments 1-2
