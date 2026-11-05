# Phase 1 Step 7: single-subject sanity (nocue, 2026-11-05)

Made by `start_vision.analysis.sanity.report` from the eegunify derivatives in `runs/nocue/` and
the staged events. Figures: seen vs unseen, plots low-passed at 30 Hz (measures unfiltered),
shaded = measurement window (PHASE1_REFERENCES §2). JSON: the numbers below.

| | sub-01 | sub-03 |
|---|---|---|
| Bad channels (interpolated) | CCP1h, FC4, P1 | none |
| ICA components removed | 2: blink (ICLabel 0.95), HEOG-correlated | 4: two blinks (ICLabel 0.98, 0.81), two EOG-correlated |
| Epochs kept | 339 / 400 | 308 / 398 |
| Rejection: seen / unseen / CR | 12 % / 19 % / 15 % | 23 % / 24 % / 19 % |
| Contrast SMD seen - unseen, kept trials: all / matched | 0.08 / -0.04 | 0.40 / 0.03 |
| VAN, matched, bilateral O1/O2, 180-280 ms | -4.76 µV | -3.45 µV |
| VAN, contralateral O1/O2, all trials | -5.86 µV | -3.91 µV |
| LP, matched, Pz/CPP3/CPP4, 350-550 ms | +2.72 µV | +1.14 µV |

Accept (PHASE1 Step 7): figure set written; ICA removes a blink component in both; seen trials
show a larger late positivity than unseen in both (direction only, one subject each).
Rejection stays under 25 % per condition. Matching keeps the contrast balanced after rejection
(it was unbalanced in sub-03 without it: SMD 0.40).

The single-subject VAN is larger than Eklund's group mean (-1.58 / -1.41 µV); single subjects
are noisy (ERP_NOISE_SME warning, VAN SME 0.92 µV in sub-03), so this is not a validation:
that is Step 10, on the group.
