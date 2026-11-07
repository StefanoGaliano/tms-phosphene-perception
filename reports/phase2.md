# Phase 2 report: decoding seen vs unseen (ds006171)

Run 2026-11-05 (build step 14). Every number below is copied from the files in `results/phase2/`
(`t2_tests.tsv`, `clusters.tsv`, `a2_clusters.tsv`, `t3_channels_windows.tsv`,
`a7_loso_window_per_subject.tsv`, `c5_catch_per_subject.tsv`, `group_notes.txt`); figures are in
`results/phase2/figures/`. Plan: `plans/PHASE2_BLUEPRINT.md` §7. Results are reported whatever
they are; no hypothesis was tuned on them.

## Setup

- **Inclusion** (`results/phase2/inclusion.tsv`, `step0.md`): 91 recordings, 34 subjects (nocue 30,
  noninformative 30, informative 31). User decision 2026-11-05: Phase 2 keeps the Phase 1
  exclusion rules (subject rule: pooled hit rate 30-70 %, FA <= 20 %; recording rules) on top of
  the §6 minimum-trial rule, so both phases analyse the same recordings.
- **Data**: `desc-stimraw` epochs (ICA applied), 40 Hz low-pass, 128 Hz, -500..1000 ms, 128 EEG
  channels; stimulus-present trials only for the main analyses (28 066 trials in LOSO).
- **Classifier**: per time point StandardScaler + L2 logistic regression (liblinear, C = 1,
  balanced class weights), ROC AUC. Within subject: stratified 5-fold CV on seen x task x
  contrast bin. Across subjects: leave one subject out (LOSO) after label-free per-subject
  baseline + prestimulus z-scoring.
- **Inference**: one AUC per subject; sign-flip permutation (10 000) cluster test on AUC - 0.5,
  TFCE as sensitivity; window tests with bootstrap 95 % CI over subjects; d_z.
- Windows: VAN 180-280 ms, LP 350-550 ms (Phase 1, `docs/PHASE1_REFERENCES.md`), post 200-600 ms
  (pre-set before any group result, `analyses.py`).
- Not run: **A6** response-key control (the yes/no key side is not recorded, decision 3).

## Main results

| Analysis | Window | N | mean AUC [95 % CI] | d_z | p |
|---|---|---|---|---|---|
| A1 within subject, pooled | VAN 180-280 | 34 | 0.593 [0.580, 0.608] | 2.18 | 1e-4 |
| | LP 350-550 | 34 | 0.677 [0.661, 0.693] | 3.55 | 1e-4 |
| | post 200-600 | 34 | 0.652 [0.638, 0.667] | 3.55 | 1e-4 |
| A7 LOSO sliding | VAN | 34 | 0.568 [0.551, 0.585] | 1.33 | 1e-4 |
| | LP | 34 | 0.652 [0.634, 0.670] | 2.82 | 1e-4 |
| | post | 34 | 0.626 [0.611, 0.641] | 2.77 | 1e-4 |
| **A7 LOSO window detector** | 200-600 (channel x 50 ms bins) | 34 | **0.730 [0.711, 0.749]** | 3.96 | 1e-4 |

- **A1** (F1): one significant cluster, 86-992 ms (p < 1e-4), peak about 0.69 near 400 ms; the
  same shape in each task (nocue 133-992 ms, noninformative 125-992, informative 47-992 ms).
- **A7** (F8): a decoder trained on 33 subjects reads the held-out subject; cluster 70-992 ms.
  Held-out window AUC ranges 0.583-0.853 over subjects; every subject is above 0.5.
- **A2** (F2): temporal generalization has one large cluster (p < 1e-4, mean AUC 0.568); a
  smaller one at p = 0.048.
- **A3** (F3): Haufe patterns; see the figure for the topography sequence.

## Confound controls

| Control | Result (post 200-600 unless noted) | Reading |
|---|---|---|
| A4 within contrast bins | 0.615 [0.601, 0.629]; 0.037 below A1 (p < 1e-4) | Part of A1 is contrast, most is not |
| A5 contrast regressed out | 0.650; 0.002 below A1 | Linear contrast removal barely matters |
| A5 decode contrast within seen / unseen | 0.545 / 0.523 | Contrast is weakly decodable, below seen/unseen |
| C3 EOG channels only | 0.516 [0.509, 0.523], cluster 438-992 ms | Small late eye signal, **flagged**: inside the detector window (200-600) |
| C3 without ICA | 0.661; 0.008 *above* A1 | ICA does not create the effect |
| C3 HEOG-step trials removed | 0.648; 0.005 below A1 | Saccade trials are not the source |
| C4 alpha only (prestim) | 0.509 [0.500, 0.519], p = 0.028 | Prestimulus alpha alone barely predicts |
| C4 post vs post + alpha | 0.770 vs 0.761; post residualized on alpha 0.769 | Post-stimulus decoding is not alpha |
| C4 low vs high alpha trials | 0.772 vs 0.764, p = 0.19 | Detector works in both alpha states |
| C5 undersampled to balance | 0.766 | Class imbalance does not drive AUC |
| **C5 catch: FA vs CR** | AUC 0.492, p = 0.90 (Wilcoxon, N = 26 with >= 3 FA; blueprint asks >= 10) | **The detector does not fire on false alarms** |

Prestimulus checks (-500..0 ms) are near 0.5 everywhere (largest 0.508); some reach p < .05 at the
window level (e.g. A1 pooled 0.504, p = 0.025) but no prestimulus cluster survives in any series.

## Generalization

- **A8 cross-task** (F9): train on one task, test on another, window AUC 0.607 off-diagonal
  (N = 33); the diagonal is 0.017 higher (p = 2e-4). TOST: the difference is inside ±0.03
  (p = 0.0012), so tasks share the code.
- **A9 cross-condition**: cross-hemifield 0.613, cross-orientation 0.629 (post window). Mirroring
  right-field trials to left-field geometry *lowers* cross-hemifield AUC to 0.560, so the shared
  signal is mostly not lateralized (LP-like) and mirroring breaks the central part.
- **A10 channels x window start** (LOSO window detector to 600 ms, `t3_channels_windows.tsv`):

| channels | from 0 ms | 100 ms | 150 ms | 200 ms |
|---|---|---|---|---|
| 128 | 0.744 | 0.745 | 0.744 | 0.730 |
| 64 | 0.747 | 0.745 | 0.743 | 0.729 |
| 32 | 0.742 | 0.740 | 0.738 | 0.724 |

  64 and 32 BioSemi-layout subsets lose almost nothing; starting at 200 ms costs about 0.015.

## What this means for Phase 3

1. A frozen across-subject detector works on new people (LOSO 0.73), on every task, and with a
   32-channel cap, which is what a TMS-EEG lab can record.
2. The confounds that could fake it (ICA, saccades, alpha, imbalance, contrast) explain little;
   the late EOG decoding (0.516) is small but sits inside the window and stays flagged.
3. **Main caveat**: on catch trials, "seen" false alarms score like correct rejections (0.492).
   The detector reads stimulus-driven evidence that leads to "seen", not the report itself. For
   TMS phosphenes, where there is no visual stimulus, a positive score would therefore point to a
   stimulus-like cortical response, and a null could mean either no percept or a percept without
   that response. N of false alarms is small (median 4 per subject), so this is a weak test.

## A11: exported detectors v1.0.0 (build step 15, 2026-11-07)

Files: `detectors/sv-perception-<variant>-v1.0.0/` (immutable; manifest hashes checked);
summaries `results/phase2/detector_<variant>_summary.json`; reliability F10
`results/phase2/figures/F10_<variant>.png`. Window and C were chosen by LOSO mean AUC over
4 windows x C {0.01, 0.1, 1}; nested LOSO (outer subject left out of the choice) is the unbiased
estimate. N = 34 subjects.

| Variant | Channels | Chosen window, C | LOSO AUC [95 % CI] | nested LOSO AUC | θ | balanced acc. | Brier | catch FA rate at θ | catch FA vs CR AUC |
|---|---|---|---|---|---|---|---|---|---|
| `tms-robust` | 64 | 150-600 ms, 0.01 | 0.750 [0.730, 0.770] | 0.750 | 0.521 | 0.688 | 0.203 | 0.228 | 0.478 |
| `full` | 128 | 150-600 ms, 0.01 | 0.758 [0.738, 0.778] | 0.758 | 0.518 | 0.696 | 0.201 | 0.222 | 0.503 |

- Both variants choose the same window and C; the nested estimate equals the plain LOSO one, so
  the choice does not inflate the result. 64 channels cost 0.008 AUC.
- Behavioural catch FA rate is 0.065; θ at max balanced accuracy gives a detector FA rate of
  about 0.22 on catch trials. The θ matched to the behavioural FA rate is in `performance.json`
  (`threshold_at_behavioural_fa`; tms-robust 0.674, hit rate 0.387 there). Phase 3 must
  pre-register the rule (Q4).
- Catch "seen" false alarms still score like correct rejections (0.478 / 0.503), as in C5.
- **Deviation**: the window x C selection fits ran on the GPU (`gpu_select`, exact Newton solve of
  liblinear's objective in float32; `sv-export --backend torch`), because the CPU selection did
  not finish `tms-robust` in 4.5 h. Tests check it against sklearn (same choices, AUC within
  0.002). The exported models are fitted by sklearn as before. `tms-robust` ran before a
  stopping-rule fix (2026-11-07: fp32 fits always ran 50 Newton steps); same optimum, slower.

## Deviations and open items

- A6 not run (no key side). C5 catch uses >= 3 FA per subject instead of the blueprint's >= 10
  (only a few subjects have 10).
- LOSO implementation changed after an out-of-memory failure at 34 subjects (same numbers, tested
  by `test_ds006171_loso_low_memory_matches_reference`). Group stage: 64 min, peak ~23 GB.
- Saccade-rejection version B (Phase 1 decision 5) is still to be run as a robustness check.
- Step 15 done (A11 above). Next: step 16 (A12 stubs).
