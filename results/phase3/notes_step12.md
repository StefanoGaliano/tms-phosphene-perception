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

## Remaining secondary analyses (`sv-p3-secondary`, `start_vision.phase3.secondary`)

Holm over all secondary tests that ran (S1, S2, S3 x2, S6 x2, S8, S9): `t2_secondary_holm_<pipeline>.tsv`.

| Test | primary | paperlike |
|---|---|---|
| S2 site O1 vs O2 (Wilcoxon, N 16) | O1 0.529, O2 0.555, p 0.14 (Holm 0.19) | p 0.13 (Holm 0.26) |
| S4 mean P(seen) phosphene / none / all; phosphene rate | 0.550 / 0.524 / 0.536; 0.463 | `s4_means_paperlike.tsv` |
| S4 reliability | TMS phosphene rate flat across P(seen) bins (about 0.42-0.52) while training tracks P(seen) (F4): the scores rank but are not calibrated on TMS | same |
| S6 VAN contralateral 180-280 ms | -0.25 µV [-0.52, 0.02], p 0.096 two-sided (0.046 expected direction), Holm 0.19 | -0.14 µV, p 0.30 |
| S6 LP Pz 350-550 ms | +0.50 µV [0.16, 0.85], p 0.016, Holm 0.048; cluster 227-469 ms, p 0.009 | +0.49 µV, p 0.007, Holm 0.021; cluster 227-508 ms, p 0.005 |
| S8 within-recording decoding | cluster 70-984 ms, mean AUC 0.617, p 1e-4 (Holm 8e-4) | 78-984 ms, 0.621, p 1e-4 |
| R3 (a) (C3 - C4) / posterior, per bin | -0.49 to -1.18 (all bins negative) | - (detector-only) |
| R3 (b) spatial r with the S100 - S200 map 0.3-1.0 s | 0.18 at 150 ms, 0.51-0.75 from 200 ms; **flagged in 8 of 9 bins** (null 95 % about 0.25) | `r3_paperlike.tsv` |

Reading R3 (b): the reference map is the S100 - S200 difference after the pulse, which contains the
phosphene effect itself as well as the response-hand lateralization, so a high correlation is
expected whenever the detector separates the classes; the test cannot tell the two parts apart.
R1 (motor channels removed, 0.536), R2 (rho -0.07) and R5 (C3-C4 alone 0.472) bound the hand part.

Figures F1-F9 in `figures/` (primary; F9 compares the pipelines). F1 shows every recording at the
four cleaning stages; F2 the artifact end and the cut used.

## Deviations (continued)
4. Secondary Holm: `sv-p3-analyze` corrected S1, S3, S9 only; the prereg says "within this list".
   Reported both; the full-list correction is `t2_secondary_holm_<pipeline>.tsv`.
5. R3 (b) map window 0.3-1.0 s instead of 0.3-1.2 s: the detector-ready epochs end at 1.0 s.
6. S6 window tests reported two-sided (prereg names no tail) and in the expected direction.
7. S6 cluster times: a bug (1-D MNE clusters are slices) put every cluster at -94 ms in the first
   run; fixed, regression test `test_phase3_s6_cluster_times_follow_planted_effect`, clusters
   recomputed from the saved subject waves for both pipelines.
8. `sv-p3-secondary` entry point added to `pyproject.toml`; this run called the CLI function
   directly (the editable reinstall needs `hatchling`, not installed offline).

## Not run (need the ds006171 epochs, `runs/` removed on purpose; or not built)
S7 retrained transfer, R4b early-only retrained detector, the `gonogo` pipeline (S5 part).
