# Phase 3 quick go/no-go (2026-11-08), exploratory

Frozen `sv-perception-tms-robust-v1.0.0` applied as is to GIN phosphene data, O1 site,
subjects 03-11 (N = 9). Script `gonogo.py` (run: `python -I gonogo.py <Start_Vision> <file.set>`).
Rough cleaning only: pulse -2..15 ms linearly interpolated, IIR 0.1-40 Hz, 128 Hz, 5 channels
interpolated (decision 10), average reference; **no ICA** (the detector was trained on
ocular-ICA-cleaned data), no trial rejection. Labels S100 = seen (inferred).

| Measure (per subject) | median | range | > chance | Wilcoxon vs chance |
|---|---|---|---|---|
| AUC detector | 0.526 | 0.497-0.572 | 8/9 | p = 0.008 |
| AUC, 18 motor channels interpolated away | 0.520 | 0.481-0.576 | 7/9 | p = 0.027 |
| AUC of C3-C4 alone (0.15-0.6 s) | 0.535 | 0.376-0.582 | 5/9 | p = 0.73 |
| Spearman P(seen) vs C3-C4 within class | -0.018 | -0.156-0.042 | 3/9 > 0 | p = 0.20 |

Reading: a small, consistent above-chance separation that survives removing the motor
channels and does not follow the hand signal trial by trial; the hand signal itself is not
consistent in the detector window. **Go**, with the caveat that the effect is small
(in-domain LOSO AUC 0.75, Eklund transfer 0.65). Not corrected for multiple tests; N = 9,
one site. Proper cleaning (ICA as in training, TESA-like decay removal) may change it.
