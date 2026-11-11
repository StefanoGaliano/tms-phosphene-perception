# EXPLORATORY (not pre-registered), 2026-11-11: per-subject fixed threshold and channel-group lesions

Script `scripts/explore_lesion_threshold.py`; primary pipeline, 17 subjects / 33 recordings. Groups fixed before running.

## A. Per-subject condition averages vs the frozen natural-vision threshold (theta 0.521)
- Mean P(seen): phosphene 0.556, none 0.530 (both above theta on average).
- Fixed theta classifies both condition averages correctly in **4/17** subjects (phosphene above theta 12/17, none below theta 9/17).
- Relative rule (phosphene average > none average): **14/17**.
- Reading: the natural-vision threshold does not transfer to TMS (absolute levels shifted up, as S4 showed); the ranking does. A Pigorini-style fixed threshold is not supported here.

## B. Virtual lesions (group interpolated from the remaining channels), H1
| Group | channels | AUC [95% CI] | p (one-sided) | subjects > 0.5 |
|---|---|---|---|---|
| none (H1) | - | 0.540 [0.521, 0.559] | 0.0009 | 14/17 |
| medial occipital (stimulated site) | O1 Oz O2 POz PO3 PO4 | 0.539 [0.518, 0.559] | 0.0017 | 13/17 |
| lateral occipito-temporal | P7 P8 PO7 PO8 TP7 TP8 T7 T8 | 0.540 [0.521, 0.559] | 0.0009 | 14/17 |
| frontal control | Fp1 Fp2 AF3 AF4 F3 F4 F7 F8 | 0.541 [0.522, 0.560] | 0.0008 | 14/17 |

Reading: no single group carries H1; the transferred signal is distributed (cf. R1 0.536, Q38 all 54 lateral channels 0.525). In particular the channels under and around the coil, where residual pulse artifact is largest, are not driving it. The lateral-source hypothesis (Knight 2024) is not specifically supported.
