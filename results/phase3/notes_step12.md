# Phase 3 step 12: run on all recordings (2026-11-09)

Commands: `sv-p3-clean` (17 subjects x O1/O2 + sub-18 O2; `--pipeline primary` then `paperlike`),
`sv-p3-score`, `sv-p3-analyze` per pipeline. Logs in `logs/`.

## Run
- 35 recordings cleaned per pipeline, no failures. Paperlike: 47 min wall clock, peak 25.2 GB RSS
  (`logs/clean_paperlike.log`). Primary recordings sub-03/04 O1 were cached from step 9.
- Inclusion (both pipelines): 33 of 35 recordings, 17 subjects. Excluded: sub-18 O2 (kept < 40,
  class < 30), sub-19 O1 (code check opposite, p < 0.05). `inclusion_{primary,paperlike}.tsv`.

## Results (one-sided p for H1; Holm within the secondary list)

| Test | primary | paperlike (S5) |
|---|---|---|
| H1 AUC [95 % CI], subjects > 0.5 | 0.540 [0.521, 0.559], p 0.0009, 14/17 | 0.540 [0.523, 0.557], p 0.0002 |
| S1 untouched subset | 0.538, p_holm 0.0046 | 0.539, p_holm 0.0016 |
| S3 balanced acc. θ / median | 0.523 / 0.531 | 0.522 / 0.528 |
| S9 seen share 30-70 % (N 14) | 0.540, p_holm 0.0034 | 0.539, p_holm 0.0016 |
| R1 motor channels removed | 0.536, p 0.002 | 0.536, p 0.0006 |
| R4a early bins (< 0.30 s) | 0.519, p 0.006 | 0.519, p 0.014 |
| E1 HEOG alone (flagged) | 0.477, p 0.025 | 0.481, p 0.024 |
| E1 follow-up, HEOG step < 16 µV | 0.543, p 0.0007 | 0.543, p 0.0002 |
| A1 H1 residualized on alpha | 0.538, p 0.001 | 0.538, p 0.0002 |

Tables: `t2_tests_<pipeline>.tsv`, `t3_controls_<pipeline>.tsv`, `per_recording_<pipeline>.tsv`,
`scores_<pipeline>.tsv`, `code_check.tsv`.

## Deviations from PHASE3_PREREG
1. `sv-p3-score` crashed on a recording below 40 kept trials; fixed so it is excluded by the
   pre-registered rule (commit 88e2df7, regression test).
2. E1 follow-up measures the HEOG step on the stored 40 Hz / 128 Hz HEOG copy; steps are slightly
   smaller than at rejection, so the 16 µV cut is a little lenient (commit 77947d4).
3. `gonogo` pipeline (S5 part) not run: `sv-p3-clean` offers primary and paperlike only.

## Not yet run (step 12 incomplete)
S2 site test, S4 calibration shift, S6 ERP signatures, S7 retrained transfer (needs ds006171
epochs, `runs/` removed on purpose), S8 within-subject decoding, R3 pattern lateralization,
R4b early-only retrained detector (needs ds006171 epochs), figures.
