# Cross-dataset test: the frozen detector on Eklund & Wiens 2018

Run 2026-11-07 (A12 issue #1; user decision 9). Numbers are copied from
`results/eklund2018/summary.json` and `session_aucs.tsv` (command `sv-eklund`) and from the
eegunify group stats in `runs/eklund2018/<size>/derivatives/eegunify-group/`.

## Question

Does `detectors/sv-perception-tms-robust-v1.0.0`, trained only on ds006171 (Madrid lab, 128-ch
cap, contrast staircase, yes/no report, hemifields), read awareness in a different lab's data
with no retraining? Eklund & Wiens 2018 (Stockholm): 64-ch BioSemi cap, Gabor **duration**
staircase, 4 **quadrants**, location response then a 3-point awareness rating, two sessions
(large and small Gabor). This is the same use Phase 3 will make of the detector.

## Setup

- Data: figshare doi:10.17045/sthlmuni.5418967 (CC BY 4.0), raw BDF + logs, MD5 checked.
- Sessions: the authors' inclusion, reproduced from their files (`analyze_VAN.m` rules on
  `allGM.mat`, `GMnBeh.mat`): large 24, small 15. Small sub-06 failed preprocessing (bad F7 made
  the F7-F8 HEOG noisy; the saccade rule rejected 82 % of epochs, 8 unaware left): **small N = 14**.
- Onsets from the photodiode, decoded as in the authors' `trialfun_VAN.m`; all 39 sessions
  matched their logs trial by trial (580 trials, same condition sequence).
- Labels (authors' contrast): aware correct (rating 2-3, location correct) vs unaware correct
  (rating 1, location correct), critical trials. Sensitivity: rating 2-3 vs 1, any location.
- Preprocessing as Phase 1 (eegunify 0.9.3, ICA). No EOG electrodes in this dataset: VEOG = Fpz-Cz,
  HEOG = F7-F8. Then the detector's own input path (40 Hz low-pass, 128 Hz, per-session
  prestimulus z-score, label-free). The 64 Eklund channels are exactly the contract channels:
  no mapping or interpolation. **No parameter was fitted on Eklund data.**
- Inference: one AUC per session; sign-flip permutation (10 000) of AUC - 0.5, one-sided;
  bootstrap 95 % CI over sessions.

## Preprocessing check (ERPs, aware correct minus unaware correct)

| | this pipeline | Eklund & Wiens 2018, Table 2 |
|---|---|---|
| LP 350-550 ms, P3/Pz/P4, large | +1.36 µV (d_z 1.59, N 24) | +1.35 µV |
| LP, small | +1.19 µV (d_z 1.54, N 14) | +1.00 µV |
| **VAN 180-280 ms, contralateral, large** | **-1.48 µV** (d_z -0.74, p 0.001, N 24) | -1.58 µV |
| **VAN, contralateral, small** | **-1.38 µV** (d_z -1.05, p 0.002, N 14) | -1.41 µV |
| VAN, bilateral O1/O2, large | -1.06 µV (d_z -0.71) | |
| VAN, bilateral, small | -0.85 µV (d_z -0.79) | |

Contralateral = O1 for right-field and O2 for left-field Gabors, the authors' measure
(`results/eklund2018/van_contralateral.{tsv,json}`). Both the LP and the contralateral VAN
reproduce the paper, so the preprocessing is not what limits the transfer.

## Result

| Gabor | N | frozen detector AUC [95 % CI] | d_z | p | sessions > 0.5 | range |
|---|---|---|---|---|---|---|
| **large (primary)** | 24 | **0.651 [0.629, 0.671]** | 2.81 | 1e-4 | 24 / 24 | 0.531-0.743 |
| small (replication) | 14 | 0.654 [0.626, 0.682] | 2.77 | 2e-4 | 14 / 14 | 0.562-0.729 |
| large, rating only | 24 | 0.662 [0.640, 0.683] | 2.91 | 1e-4 | 24 / 24 | |
| small, rating only | 14 | 0.666 [0.635, 0.697] | 2.68 | 2e-4 | 14 / 14 | |

Mean calibrated P(seen) per trial class:

| | control (long Gabor) | aware correct | unaware correct | unaware incorrect | aware incorrect | catch |
|---|---|---|---|---|---|---|
| large | 0.703 | 0.619 | 0.518 | 0.476 | 0.469 | 0.474 |
| small | 0.635 | 0.624 | 0.515 | 0.469 | 0.486 | 0.466 |

Catch trials, "saw" false alarms vs correct rejections: mean AUC 0.469 (large, 21 sessions with
any false alarm) and 0.539 (small, 12): no consistent separation, as in ds006171 (C5).

## Reading

1. **The detector transfers.** With no retraining, a model from another lab, cap and paradigm
   separates aware from unaware in every one of 38 sessions; the effect is the same in both
   Gabor sizes. Held-out AUC drops from 0.750 (new subjects within ds006171) to about 0.65
   (new lab), so roughly 60 % of the above-chance signal survives the change of dataset.
2. **What it reads is stimulus-driven awareness evidence, not the report alone.** Trials rated
   aware but mislocalized score like unaware ones (0.47-0.49), catch false alarms do not separate
   from correct rejections, and clearly visible control Gabors score highest. The same caveat as
   Phase 2 holds for Phase 3: a phosphene must produce a stimulus-like cortical response to
   score high.
3. Calibration is shifted: probabilities are compressed toward 0.5 on the new data (aware
   0.62 vs unaware 0.52), so Phase 3 should not rely on the global threshold θ without a
   check; the subject-adaptive rule (open question Q4) is the safer pre-registration.

## Secondary: decoder retrained on ds006171, tested over time on Eklund

A sliding decoder (per time point, as Phase 2 A7) trained on all 34 included ds006171
subjects (28 066 stimulus-present trials, 64-channel subset), tested on each Eklund session;
AUC(t) of aware vs unaware correct (`results/eklund2018/retrained_transfer.{json,_auc.npz}`).

| | Large (N 24) | Small (N 14) |
|---|---|---|
| VAN 180-280 ms | 0.538 [0.526, 0.551], d_z 1.22, p 1e-4 | 0.534 [0.519, 0.549], p 5e-4 |
| LP 350-550 ms | 0.601 [0.587, 0.613], d_z 3.00, p 1e-4 | 0.595 [0.577, 0.617], p 2e-4 |
| post 200-600 ms | 0.572 [0.561, 0.583], p 1e-4 | 0.569 [0.554, 0.586], p 2e-4 |
| prestim -500..0 ms | 0.504, p 0.22 | 0.503, p 0.29 |
| clusters (p < .05) | 156-273 ms (p 0.027), 305-742 ms (p 1e-4) | 156-266 ms (p 0.045), 320-805 ms (p 1e-4) |
| peak | 0.628 at 453 ms | 0.624 at 477 ms |

- The transferable code has two parts: a small early one in the VAN window and a larger late
  one peaking near 450 ms (the LP window). Nothing before the stimulus.
- Per time point the transfer is weaker than the frozen window detector (0.65), which pools
  150-600 ms; the time course says where the shared signal is, not a better detector.
- Within ds006171 the A7 sliding decoder reached 0.568 (VAN) and 0.652 (LP); across labs
  0.538 and 0.601. The late part carries most of what generalizes.
