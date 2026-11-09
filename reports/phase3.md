# Phase 3 report: do TMS-evoked phosphenes carry the naturally-seen signature? (GIN Phosphenes_occipital)

Run 2026-11-09 (build steps 12-13). Every number below is copied from `results/phase3/`
(`t2_tests_<pipeline>.tsv`, `t2_secondary_holm_<pipeline>.tsv`, `t3_controls_<pipeline>.tsv`,
`s2/s4/s6/s8/r3_<pipeline>.tsv`, `inclusion_<pipeline>.tsv`, `notes_step12.md`); figures in
`results/phase3/figures/`. Pre-registration: `docs/PHASE3_PREREG.md`, tag `phase3-prereg`
(2026-11-08), before any O2 or O1 >= 13 recording was cleaned. Results are reported whatever
they are; the primary pipeline is the pre-registered one (training-like, Q5 option A).

## Setup

- **Data:** GIN `DB_123/Phosphenes_occipital` (Siviero et al. 2023, Sensors 23:8833; CC0). Single
  TMS pulses over O1 or O2 at phosphene threshold, 64-ch EEG at 5000 Hz, phosphene yes/no on every
  pulse. Codes S100 = phosphene, S200 = none (**inferred** from the data; authors asked 2026-11-08).
  Response hand fixed to the label (phosphene = right hand). 35 recordings on disk (18 subjects x 2
  sites; sub-18 O1 is not on GIN). 5 detector channels missing from the cap, interpolated (Decision 10).
- **Detector:** `detectors/sv-perception-tms-robust-v1.0.0` (trained on ds006171 natural vision,
  64 ch, features 150-600 ms in 50 ms bins, per-recording prestimulus normalization), applied
  unchanged. Nothing was fitted on TMS data for H1.
- **Cleaning** (`start_vision.tms`, `phase3.clean`): pulse cut -2..10 ms, cubic interpolation,
  extended to the measured artifact end up to 20 ms; decay/muscle-twitch ICA by the TESA rule;
  then the training preprocessing (eegunify 0.9.3: pyprep bad channels, ocular ICA, average
  reference), 40 Hz low-pass, 128 Hz. Rejection as training: 100 µV peak-to-peak, HEOG step 32 µV.
  `paperlike` adds the paper's muscle-ICA round (S5). F1 shows every recording at each stage:
  the raw pulse (about 10³ µV GMFP) falls to the 1-10 µV range after the cut and the decay ICA.
- **Inclusion:** >= 40 kept trials and >= 30 per class; code-direction check. **33 of 35
  recordings, 17 subjects** in both pipelines. Excluded: sub-18 O2 (kept < 40, class < 30),
  sub-19 O1 (code check opposite). Phosphene share 0.23-0.71 (mean 0.46).
- **Inference:** per-recording AUC (S100 vs S200); subject = mean of included sites; one-sided
  sign-flip test over subjects (10 000), bootstrap 95 % CI, d_z; secondary tests Holm-corrected.

## Primary result (H1)

| Pipeline | AUC [95 % CI] | d_z | p (one-sided) | subjects > 0.5 |
|---|---|---|---|---|
| **primary (pre-registered)** | **0.540 [0.521, 0.559]** | 0.97 | **0.0009** | 14/17 |
| paperlike (S5) | 0.540 [0.523, 0.557] | 1.09 | 0.0002 | - |

**H1 is supported.** Among physically identical pulses, those reported as a phosphene get a higher
P(seen) from a detector trained only on natural near-threshold vision. Per-recording AUC ranges
0.458-0.670 (F3); site means O1 0.529, O2 0.552.

## Secondary tests

Holm over all secondary tests that ran (`t2_secondary_holm_primary.tsv`); `sv-p3-analyze` also
reports a Holm over S1/S3/S9 only (deviation 4).

| # | Test | primary | p (Holm, all) | paperlike (Holm) |
|---|---|---|---|---|
| S1 | H1 on recordings not used in the go/no-go | 0.538 | 0.018 | 0.004 |
| S2 | site O1 vs O2 (Wilcoxon, N 16) | O1 0.529, O2 0.555 | 0.19 | 0.26 |
| S3 | balanced accuracy at θ 0.521 | 0.523 | 0.002 | 0.003 |
| S3 | balanced accuracy at recording median | 0.531 | 0.003 | 0.003 |
| S6 | VAN contralateral (O1/PO7 or O2/PO8), 180-280 ms | -0.25 µV [-0.52, 0.02] | 0.19 | 0.30 |
| S6 | LP Pz, 350-550 ms | +0.50 µV [0.16, 0.85]; cluster 227-469 ms (p 0.009) | 0.048 | 0.021 |
| S8 | within-recording decoding (retrained per recording) | cluster 70-984 ms, mean AUC 0.617 | 0.0008 | 0.0008 |
| S9 | recordings with phosphene share 30-70 % (N 14) | 0.540 | 0.009 | 0.004 |

- **S4 calibration (descriptive, F4):** mean P(seen) 0.550 (phosphene) vs 0.524 (none), 0.536
  overall; observed phosphene rate 0.463. The phosphene rate is nearly flat across P(seen) bins
  (about 0.42-0.52) while the training reliability curve tracks P(seen): on TMS the scores rank
  pulses slightly better than chance but their probabilities are not calibrated.
- **S5 pipeline robustness:** every test and control is the same to the second decimal with the
  paper's muscle-ICA round (F9). The `gonogo` pipeline was not run (deviation 3).
- **S6 ERPs (F7):** a late positivity at Pz is larger after phosphene pulses (cluster from 227 ms);
  at the stimulated-side posterior channels, under the coil, the waves of the two classes nearly
  overlap and the VAN-window difference is small and not significant after correction.
- **S8:** the TMS data themselves separate the classes well within a recording (0.617 from 70 ms),
  more than the frozen detector reads (0.540). This shows what the data hold; it is not evidence for
  the "naturally-seen" claim.
- **S7** (retrained ds006171 decoder, time-resolved transfer) **not run**: it needs the ds006171
  epochs, which were removed for disk space (to rebuild later).

## Confound controls

| # | Control | primary | paperlike | Flag |
|---|---|---|---|---|
| R1 | 18 motor channels interpolated | 0.536 [0.516, 0.556], p 0.002 | 0.536, p 0.0006 | no |
| R2 | P(seen) vs C3-C4 within class | mean rho -0.07 | -0.07 | no |
| R3 | detector pattern vs TMS lateralization map | see below | - | **yes (8/9 bins)** |
| R4a | frozen model, bins before 300 ms only | 0.519 [0.507, 0.532], p 0.006 | 0.519, p 0.014 | reported |
| R4b | early-only retrained detector | not run (needs ds006171 epochs) | - | - |
| R5 | C3-C4 alone | 0.472, p 0.20 | 0.475 | reported |
| E1 | HEOG alone | 0.477, p 0.025 (below chance) | 0.481, p 0.024 | **yes** |
| E1 | VEOG alone | 0.514, p 0.18 | 0.514 | no |
| E1 | follow-up: H1 on trials with HEOG step < 16 µV (11 % removed) | 0.543 [0.524, 0.562], p 0.0007 | 0.543, p 0.0002 | required, holds |
| A1 | prestimulus alpha alone | 0.519, p 0.14 | 0.519 | reported |
| A1 | H1 with P(seen) residualized on alpha | 0.538 [0.520, 0.558], p 0.001 | 0.538 | reported |

- **Hand:** removing the motor channels leaves H1 intact, the score does not follow the C3-C4 signal
  within class, and the hand signal alone does not separate the classes.
- **R3 flag (F6):** (a) the detector pattern is more weighted at C4 than C3 in every bin (relative
  C3 - C4 between -0.49 and -1.18 of the posterior magnitude); this comes from the ds006171
  training data, where the response side was randomized. (b) the pattern correlates with the TMS
  S100 - S200 map (0.3-1.0 s) at r 0.51-0.75 from 200 ms, above the permutation threshold (about
  0.25) in 8 of 9 bins. That map is the post-pulse class difference, so it contains the phosphene
  effect itself together with any hand lateralization: a high r is expected whenever the detector
  separates the classes, and this test cannot tell the two apart. R1, R2 and R5 are the tests that
  bound the hand part, and they do not flag.
- **Caveat, response hand in the VAN window (exploratory, 2026-11-09,
  `results/questions_to_solve/van_topography_explore/`):** at the O2 site the phosphene - none map
  in 180-280 ms shows left central negativity (C3 -0.31, C5 -0.38, CP5 -0.41 µV) with right
  central positivity (C4 +0.42), the pattern of a prepared right-hand response, extending to left
  lateral-posterior channels (P7 -0.51, TP7 -0.47, PO7 -0.43 µV) that R1 did not remove. H1 with
  the 18 central channels removed holds (R1, 0.536), but the hand signal is present this early and
  outside R1's channel set, so a hand contribution to H1 through lateral-posterior channels is not
  excluded. At the O1 site central channels are flat and the posterior negativity (O1 -0.60 µV) is
  bilateral, the better VAN candidate. Uncorrected per-channel tests; not pre-registered.
- **Eyes:** the HEOG effect runs opposite to H1, and H1 is unchanged on trials with small HEOG steps.
- **Alpha:** prestimulus alpha does not separate the classes here (unlike Romei 2008 and Samaha 2017),
  and H1 survives regressing it out.
- **Early window:** the frozen model limited to bins before 300 ms still separates the classes
  (0.519), so the effect is not only the late, report-sensitive part.

## What can and cannot be claimed (PHASE3_PREREG §10)

- **Can:** among physically identical pulses at one intensity, pulses later reported as a phosphene
  carry more of the EEG pattern that marks a naturally seen near-threshold Gabor than pulses reported
  as none, bounded by R1-R5, E1, A1.
- **Cannot**, without sham or catch pulses: estimate the detector's false-alarm rate on TMS;
  separate the percept from the report, decision or response preparation (hand fixed to label);
  rule out that a fluctuating state (arousal, attention) drives both the report and the signal;
  attribute absolute P(seen) levels to visual cortex rather than click/scalp sensation (these are
  identical in both classes, so they do not explain the *difference*, but they do shift the level).
- Phase 2 C5 caveat applies: on ds006171 catch trials "seen" false alarms scored like correct
  rejections, so the detector reads stimulus-like evidence, not the report itself; a null in TMS
  could have meant no percept signal or a percept without that evidence.
- Labels are inferred (D8); RTs unknown; 5 channels interpolated (Decision 10); go/no-go subjects
  re-used (S1 is the clean check).

## Deviations from the pre-registration (`results/phase3/notes_step12.md`)

1. `sv-p3-score` crashed on a recording below 40 kept trials; fixed so the pre-registered exclusion
   applies (commit 88e2df7, regression test).
2. The E1 follow-up measures the HEOG step on the stored 40 Hz / 128 Hz copy; steps are slightly
   smaller than at rejection, so the 16 µV cut is a little lenient (commit 77947d4).
3. The `gonogo` pipeline (part of S5) was not run.
4. Secondary Holm: reported over S1/S3/S9 (as coded) and over all secondary tests that ran (as the
   prereg text says); conclusions above use the full list.
5. R3 (b) map window 0.3-1.0 s instead of 0.3-1.2 s (epochs end at 1.0 s).
6. S6 window tests reported two-sided (no tail in the prereg) and in the expected direction.
7. S6 cluster times were wrong in the first run (1-D cluster bug); fixed with a regression test and
   recomputed from the saved subject waves.
8. Not run: S7, R4b (need the ds006171 epochs), `gonogo`.

## Open items

- S7 and R4b once the ds006171 epochs are rebuilt (`runs/`, deferred by the user).
- Confirm S100/S200 with the authors (asked 2026-11-08).
- Hand caveat above: a pre-registered VAN-topography follow-up with a wider hand control (council idea 013).
- Follow-ups pre-registered in `docs/QUESTIONS_PREREG.md` (tags `questions-prereg`,
  `questions-prereg-amend1`): Q37a (trial-level pulse-response amplitude) and Q37b (pulses needed);
  build plan `plans/QUESTIONS_BLUEPRINT.md` waiting for the user's answers.
- Literature context and article framing: `docs/ARTICLE_NOTES.md` (Phase 3 sections).
