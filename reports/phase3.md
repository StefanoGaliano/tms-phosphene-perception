# Phase 3 report: do TMS-evoked phosphenes carry the naturally-seen signature? (GIN Phosphenes_occipital)

Run 2026-11-09 (build steps 12-13); all analyses re-run 2026-11-16 with sub-18 O1 added (corrected GIN file). Every number below is copied from `results/phase3/`
(`t2_tests_<pipeline>.tsv`, `t2_secondary_holm_<pipeline>.tsv`, `t3_controls_<pipeline>.tsv`,
`s2/s4/s6/s8/r3_<pipeline>.tsv`, `inclusion_<pipeline>.tsv`, `notes_step12.md`); figures in
`results/phase3/figures/`. Pre-registration: `docs/PHASE3_PREREG.md`, tag `phase3-prereg`
(2026-11-08), before any O2 or O1 >= 13 recording was cleaned. Results are reported whatever
they are; the primary pipeline is the pre-registered one (training-like, Q5 option A).

## Setup

- **Data:** GIN `DB_123/Phosphenes_occipital` (Siviero et al. 2023, Sensors 23:8833; CC0). Single
  TMS pulses over O1 or O2 at phosphene threshold, 64-ch EEG at 5000 Hz, phosphene yes/no on every
  pulse. Codes S100 = phosphene, S200 = none (inferred from the data, **confirmed by the authors**, D. Bonfanti, email 2026-11-16).
  Response hand fixed to the label (phosphene = right hand). 36 recordings on disk (18 subjects x 2
  sites; sub-18 O1 was a wrong file on GIN, replaced by the authors and re-downloaded 2026-11-16). 5 detector channels missing from the cap, interpolated (Decision 10).
- **Detector:** `detectors/sv-perception-tms-robust-v1.0.0` (trained on ds006171 natural vision,
  64 ch, features 150-600 ms in 50 ms bins, per-recording prestimulus normalization), applied
  unchanged. Nothing was fitted on TMS data for H1.
- **Cleaning** (`start_vision.tms`, `phase3.clean`): pulse cut -2..10 ms, cubic interpolation,
  extended to the measured artifact end up to 20 ms; decay/muscle-twitch ICA by the TESA rule;
  then the training preprocessing (eegunify 0.9.3: pyprep bad channels, ocular ICA, average
  reference), 40 Hz low-pass, 128 Hz. Rejection as training: 100 µV peak-to-peak, HEOG step 32 µV.
  `paperlike` adds the paper's muscle-ICA round (S5). F1 shows every recording at each stage:
  the raw pulse (about 10³ µV GMFP) falls to the 1-10 µV range after the cut and the decay ICA.
- **Inclusion:** >= 40 kept trials and >= 30 per class; code-direction check. **34 of 36
  recordings, 18 subjects** in both pipelines. Excluded: sub-18 O2 (kept < 40, class < 30),
  sub-19 O1 (code check opposite). Phosphene share 0.23-0.71 (mean 0.46).
- **Inference:** per-recording AUC (S100 vs S200); subject = mean of included sites; one-sided
  sign-flip test over subjects (10 000), bootstrap 95 % CI, d_z; secondary tests Holm-corrected.

## Primary result (H1)

| Pipeline | AUC [95 % CI] | d_z | p (one-sided) | subjects > 0.5 |
|---|---|---|---|---|
| **primary (pre-registered)** | **0.538 [0.521, 0.556]** | 0.95 | **0.0007** | 15/18 |
| paperlike (S5) | 0.538 [0.523, 0.555] | 1.07 | 0.0002 | 16/18 |
| gonogo (S5) | 0.536 [0.520, 0.554] | 0.96 | 0.0003 | 16/18 |

**H1 is supported.** Among physically identical pulses, those reported as a phosphene get a higher
P(seen) from a detector trained only on natural near-threshold vision. Per-recording AUC ranges
0.458-0.670 (F3); site means O1 0.528, O2 0.552.

## Secondary tests

Holm over all secondary tests that ran (`t2_secondary_holm_primary.tsv`); `sv-p3-analyze` also
reports a Holm over S1/S3/S9 only (deviation 4).

| # | Test | primary | p (Holm, all) | paperlike (Holm) |
|---|---|---|---|---|
| S1 | H1 on recordings not used in the go/no-go | 0.537 | 0.018 | 0.002 |
| S2 | site O1 vs O2 (Wilcoxon, N 16) | O1 0.529, O2 0.555 | 0.19 | 0.26 |
| S3 | balanced accuracy at θ 0.521 | 0.524 | 0.003 | 0.001 |
| S3 | balanced accuracy at recording median | 0.531 | 0.003 | 0.002 |
| S6 | VAN contralateral (O1/PO7 or O2/PO8), 180-280 ms | -0.29 µV [-0.55, -0.02] | 0.11 | 0.26 |
| S6 | LP Pz, 350-550 ms | +0.51 µV [0.18, 0.85]; cluster 305-508 ms (p 0.012) | 0.028 | 0.008 |
| S8 | within-recording decoding (retrained per recording) | cluster 78-984 ms, mean AUC 0.613 | 0.0008 | 0.0008 |
| S9 | recordings with phosphene share 30-70 % (N 15) | 0.538 | 0.005 | 0.0008 |

- **S4 calibration (descriptive, F4):** mean P(seen) 0.543 (phosphene) vs 0.518 (none), 0.529
  overall; observed phosphene rate 0.470. The phosphene rate is nearly flat across P(seen) bins
  (about 0.42-0.52) while the training reliability curve tracks P(seen): on TMS the scores rank
  pulses slightly better than chance but their probabilities are not calibrated.
- **S5 pipeline robustness:** every test and control is the same to the second decimal with the
  paper's muscle-ICA round (F9). With the rough go/no-go recipe (`gonogo`: pulse linearly
  interpolated -2..15 ms, IIR 0.1-40 Hz, no ICA, no trial rejection; run 2026-11-11) H1 = 0.536
  [0.520, 0.554], d_z 0.96, p 0.0003, 16/18 subjects; 35 recordings, 18 subjects (sub-18 O2
  passes inclusion, since nothing is rejected; sub-19 O1 still fails the code check). S1 0.538,
  S3 0.518/0.526, S9 0.538 (all Holm < 0.005); controls as primary (R1 0.531, R5 C3-C4 alone 0.487,
  E1 HEOG alone 0.482 flagged below chance as in primary, E1 follow-up 0.539, A1 residualized 0.534).
  The result does not depend on the cleaning (`results/phase3/*_gonogo.tsv`).
- **S6 ERPs (F7):** a late positivity at Pz is larger after phosphene pulses (cluster 305-508 ms);
  at the stimulated-side posterior channels, under the coil, the waves of the two classes nearly
  overlap and the VAN-window difference is small (-0.29 µV) and not significant after correction (Holm 0.11).
- **S8:** the TMS data themselves separate the classes well within a recording (0.613 from 78 ms),
  more than the frozen detector reads (0.538). This shows what the data hold; it is not evidence for
  the "naturally-seen" claim.
- **S7** (run 2026-11-10 after the ds006171 epochs were rebuilt, re-run 2026-11-16 with sub-18 O1; `results/phase3/s7_r4b/s7.json`,
  `scripts/phase3_s7_r4b.py`): a sliding decoder trained on all 34 ds006171 subjects (64-channel
  `tms-robust` set, stimulus-present trials) and tested on each included TMS recording. One cluster,
  156-992 ms (p 0.0001, mean AUC 0.547; paperlike 164-992 ms); window AUCs VAN 0.534 [0.521, 0.547],
  LP 0.545 [0.530, 0.561], post 150-600 0.539 [0.527, 0.551] (all p 0.0001), prestimulus 0.501
  (p 0.33). Paperlike: VAN 0.529, LP 0.544. The transfer starts near 150 ms, as in
  natural vision (no lag visible at this resolution), and keeps rising to 0.579 at 0.99 s; the
  part after about 0.6 s overlaps the response-locked lateralization flagged by R3 and is not read
  as perception. S7 ran after the Holm family above was computed and is reported outside it; its p
  would survive any correction over the secondary tests.

## Confound controls

| # | Control | primary | paperlike | Flag |
|---|---|---|---|---|
| R1 | 18 motor channels interpolated | 0.533 [0.514, 0.553], p 0.002 | 0.534, p 0.0006 | no |
| R2 | P(seen) vs C3-C4 within class | mean rho -0.08 | -0.08 | no |
| R3 | detector pattern vs TMS lateralization map | see below | - | **yes (8/9 bins)** |
| R4a | frozen model, bins before 300 ms only | 0.516 [0.502, 0.529], p 0.022 | 0.516, p 0.033 | reported |
| R4b | early-only retrained detector (0.15-0.30 s, C 0.01, ds006171 LOSO AUC 0.655) | 0.536 [0.520, 0.552], p 0.0006, 13/18 | 0.531 [0.516, 0.546], p 0.0013, 13/18 | reported |
| R5 | C3-C4 alone | 0.473, p 0.18 | 0.475 | reported |
| E1 | HEOG alone | 0.480, p 0.041 (below chance) | 0.483, p 0.035 | **yes** |
| E1 | VEOG alone | 0.511, p 0.31 | 0.511 | no |
| E1 | follow-up: H1 on trials with HEOG step < 16 µV (11 % removed) | 0.538 [0.518, 0.557], p 0.001 | 0.537, p 0.0007 | required, holds |
| A1 | prestimulus alpha alone | 0.513, p 0.33 | 0.513 | reported |
| A1 | H1 with P(seen) residualized on alpha | 0.536 [0.518, 0.555], p 0.001 | 0.536 | reported |

- **Hand:** removing the motor channels leaves H1 intact, the score does not follow the C3-C4 signal
  within class, and the hand signal alone does not separate the classes.
- **R3 flag (F6):** (a) the detector pattern is more weighted at C4 than C3 in every bin (relative
  C3 - C4 between -0.49 and -1.18 of the posterior magnitude); this comes from the ds006171
  training data, where the response side was randomized. (b) the pattern correlates with the TMS
  S100 - S200 map (0.3-1.0 s) at r 0.46-0.69 from 200 ms, above the permutation threshold (about
  0.25) in 8 of 9 bins. That map is the post-pulse class difference, so it contains the phosphene
  effect itself together with any hand lateralization: a high r is expected whenever the detector
  separates the classes, and this test cannot tell the two apart. R1, R2 and R5 are the tests that
  bound the hand part, and they do not flag.
- **Caveat, response hand in the VAN window (exploratory, 2026-11-09,
  `results/questions_to_solve/van_topography_explore/`):** at the O2 site the phosphene - none map
  in 180-280 ms shows left central negativity (C3 -0.31, C5 -0.38, CP5 -0.41 µV) with right
  central positivity (C4 +0.42), the pattern of a prepared right-hand response, extending to left
  lateral-posterior channels (P7 -0.51, TP7 -0.47, PO7 -0.43 µV) that R1 did not remove. H1 with
  the 18 central channels removed holds (R1, 0.533), but the hand signal is present this early and
  outside R1's channel set, so a hand contribution to H1 through lateral-posterior channels is not
  excluded. At the O1 site central channels are flat and the posterior negativity (O1 -0.60 µV) is
  bilateral, the better VAN candidate. Uncorrected per-channel tests; not pre-registered.
- **Eyes:** the HEOG effect runs opposite to H1, and H1 is unchanged on trials with small HEOG steps.
- **Alpha:** prestimulus alpha does not separate the classes here (unlike Romei 2008 and Samaha 2017),
  and H1 survives regressing it out.
- **Early window:** the frozen model limited to bins before 300 ms still separates the classes
  (0.516), and a detector retrained on 0.15-0.30 s only (R4b, 2026-11-10, re-run 2026-11-16) reaches 0.536, about the
  size of H1, so the effect is not only the late, report-sensitive part.

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
- Labels confirmed by the authors (D8, 2026-11-16); per-trial RTs now available (E-Prime `vresp.RT`, `sourcedata/phosphenes_occipital/beh_eprime_18ss.xlsx`) but not yet used; 5 channels interpolated (Decision 10); go/no-go subjects
  re-used (S1 is the clean check).

## Deviations from the pre-registration (`results/phase3/notes_step12.md`)

1. `sv-p3-score` crashed on a recording below 40 kept trials; fixed so the pre-registered exclusion
   applies (commit 88e2df7, regression test).
2. The E1 follow-up measures the HEOG step on the stored 40 Hz / 128 Hz copy; steps are slightly
   smaller than at rejection, so the 16 µV cut is a little lenient (commit 77947d4).
3. The `gonogo` pipeline (part of S5) ran late (2026-11-11), after the report draft. sub-18 O1 has no
   `.fdt` in the GIN download (the O1 folder holds the O2 file), so it was missing in all pipelines. The authors uploaded the correct file; it was re-downloaded and every analysis (primary, paperlike, gonogo, secondary, S7, R4b, figures) re-run on 2026-11-16. Sub-18 O1 kept 94 trials (primary), AUC 0.511. The R4b detector was retrained (old one kept as `s7_r4b/r4b_detector_pre_sub18O1_2026-11-16/`); `t2_secondary_holm_<pipeline>.tsv` recomputed with the same Holm over the same eight tests.
4. Secondary Holm: reported over S1/S3/S9 (as coded) and over all secondary tests that ran (as the
   prereg text says); conclusions above use the full list.
5. R3 (b) map window 0.3-1.0 s instead of 0.3-1.2 s (epochs end at 1.0 s).
6. S6 window tests reported two-sided (no tail in the prereg) and in the expected direction.
7. S6 cluster times were wrong in the first run (1-D cluster bug); fixed with a regression test and
   recomputed from the saved subject waves.
8. `gonogo` ran 2026-11-11 (deviation 3). S7 and R4b ran on 2026-11-10, a day after the other tests, on ds006171 epochs
   rebuilt with the same configs (the 5 eegunify failures and their rejection rates reproduce
   Phase 1 exactly); the R4b C was chosen by LOSO over the Phase 2 grid on the GPU backend.

## Open items

- S100/S200 confirmed by the authors 2026-11-16 (closed).
- RTs: `sourcedata/phosphenes_occipital/beh_eprime_18ss.xlsx` (12,929 trials, subjects 03-22; RT column `vresp.RT`).
  Before use: map Session 1/2 to O1/O2; RT = 0 marks no response (233 trials, the only ones with
  `vresp.ACC` = 1); sub-07 Session 1 has 269 trials plus 60 logged as Session 11 (329, not 360).
- Read the extra paper on phosphene electrophysiology the authors recommended (same dataset) before citing it.
- Follow-ups done 2026-11-09, **before sub-18 O1 was added (17 subjects); not re-run** (pre-registered in `docs/QUESTIONS_PREREG.md`, tags `questions-prereg`,
  `-amend1`, `-amend2`; Holm over seven primary tests, `results/questions_to_solve/prereg_family/`):
  - Q38 hand caveat (idea 013): H1 with P(seen) residualized on a hand index from all lateral
    channel pairs 0.537 [0.519, 0.555], 14/17, Holm p 0.005; all 54 lateral channels interpolated
    0.525 (p 0.003). The hand signal does not carry H1; hand fixed to the label stays a bound, not a
    removal (`prereg_q38/`).
  - Q37a: H1 residualized on trial TEP amplitude 0.526 [0.510, 0.543], Holm p 0.010; TEP amplitude
    alone separates the classes (0.541), so the phosphene also modulates the evoked response (`prereg_q37a/`).
  - Q37b: O2 block AUC at 20 pulses 0.686 [0.580, 0.784], Holm p 0.009; k* not reached on O1
    (`prereg_q37b/`).
- Literature context and article framing: `docs/ARTICLE_NOTES.md` (Phase 3 sections).
