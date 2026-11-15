# Questions_to_solve: pre-registration addendum

**Status: FINAL (2026-11-09), tagged `questions-prereg`.** Any later change is a reported
deviation. Council evaluation: ideas 002-006 and 008 applied (`ideas/INDEX.md`); 002 is this document;
idea 009 (Q37b) added by amendment 1 (tag `questions-prereg-amend1`).

Questions and full references: `Question_to_answer/` (`QUESTIONS.md`, "References per question").
Scripts go in `scripts/`, results in `results/questions_to_solve/<question>/`.

## 0. What has already been seen (honesty table)

These analyses are specified **after** a first look at the same data. They are not confirmatory on
those data. What this addendum adds: every choice is fixed before the rerun, the analyses run once,
the multiplicity is corrected, and where a part of the data has not been used for the question it is
named as the confirmation set.

| Q | First look (2026-11-09) | Data already used | Not yet used for this question |
|---|---|---|---|
| Q01 | `q01_q03_eklund/` | Eklund large + small, all rating levels | none (no other dataset has ratings) |
| Q02 | `q01_q03_eklund/`, `saved_results/` | Eklund; ds006171 (run after Eklund, settings fixed before running) | none |
| Q03 | same | Eklund; ds006171 | the mixture-model test itself (not computed yet) |
| Q04 | `saved_results/` | ds006171 LOSO scores, unmatched | contrast-matched analysis |
| Q37a | `q37ab_phase3/` (mean subtraction, SSP-3) | GIN primary, all recordings | trial-level TEP amplitude covariate |
| Q37b | `q37ab_phase3/` | GIN primary, all recordings (pooled) | the O1 -> O2 split and the independence model |
| Q38 | `van_topography_explore/` | GIN primary, all channels, both sites | the hand-lateralization covariate |

## 1. General rules (all questions)

- Unit of inference: subject (Eklund: session within a Gabor size; GIN: subject = mean of included
  sites). Two-stage summary statistics: one value per unit (a regression slope, AUC or difference),
  then a group test. Nothing is fitted across units; no refitting of any detector; the frozen
  detectors are used as exported.
- Group tests: sign-flip permutation, 10 000 permutations, two-sided unless stated; bootstrap 95 % CI
  (10 000); d_z. Seed 20261109.
- Equivalence: TOST with bound ±0.03 AUC (**PROJECT CHOICE**, the Phase 2 A8 precedent,
  `reports/phase2.md`). No other equivalence bounds.
- Minimum cell sizes: 3 trials per cell for a unit to enter a contrast (as the first look).
- **Primary test per question** and **family**: the seven primary tests marked **[P]** below are
  Holm-corrected together (family alpha 0.05). Everything else is secondary, reported with
  uncorrected p and labelled.
- Inclusion: as in the source analyses (Eklund: authors-included sessions, critical trials,
  duration exclusion; ds006171: Phase 2 inclusion; GIN: `inclusion_primary.tsv`). No new exclusions.
- Reporting: one table and one figure per question, the result whatever it is, in
  `results/questions_to_solve/<q>/README.md` and `docs/ARTICLE_NOTES.md`.

## 2. Q01 Graded awareness (Eklund) — idea 003

- Hypothesis: P(seen) of the frozen `tms-robust` detector increases with the subjective rating
  (1 did not see, 2 saw something, 3 saw clearly), beyond stimulus duration.
- Data: `results/eklund2018/trial_scores.tsv.gz`; critical trials (condition 1-4) with correct
  localization (`obj_corr` = 1), duration exclusion applied.
- Measure: per session, OLS of P(seen) on rating (numeric 1-3) and `duration_ms` (centred within
  session); the rating slope.
- **[P]** Large Gabors: rating slope > 0 (one-sided). Small Gabors: the same, reported as replication.
- Secondary: steps 2-1 and 3-2 (sessions with >= 3 trials per level) as in the first look.
- Interpretation rule: a positive slope with the covariate = graded read-out; if the slope vanishes
  with the covariate, the gradation is attributed to duration.

## 3. Q02 Seen with wrong content (Eklund, ds006171) — idea 004

- Hypothesis: among trials reported as seen, wrong content (location or orientation) gives lower
  P(seen) than correct content, not explained by stimulus strength.
- Eklund: rating >= 2 critical trials; per session OLS of P(seen) on correctness (1/0) and
  `duration_ms`; sessions with >= 3 wrong trials. **[P]** correctness slope > 0 (one-sided), large
  Gabors. Small Gabors: replication.
- ds006171: seen trials with the orientation question; per subject, AUC(correct vs wrong) on LOSO
  scores residualized on contrast within subject x task (as the first look); secondary.
- Interpretation rule: both datasets positive = the read-out needs content-bearing evidence; stated
  limit: wrong-content "seen" trials can be false reports.

## 4. Q03 "Unseen but correct": a guessing-mixture test (Eklund; ds006171 check) — idea 005

- Question: is the higher P(seen) of unseen-correct vs unseen-wrong trials (first look +0.041)
  what weak percepts reported as unseen would produce (criterion account), or less (stimulus
  processing without the awareness signature)?
- Model, per session (rating 1 critical trials): localization accuracy a; chance c = 0.25 (four
  quadrants, `src/start_vision/eklund2018.py:10-13`); informed fraction f = max(0, (a - c)/(1 - c)),
  assuming informed trials are always correct and guesses are uniform. Correct trials = informed +
  lucky guesses, so the expected difference is
  D_pred = f / (f + (1 - f) c) × (m_I - m_G), with m_G = mean P(seen) of unseen-wrong trials (all
  guesses) and m_I = mean P(seen) of rating-2 correct trials (**PROJECT CHOICE**: the reference
  class for "weak percepts").
- Observed: D_obs = mean P(seen) unseen-correct - unseen-wrong.
- **[P]** Large Gabors: D_obs - D_pred, two-sided. Secondary: TOST ±0.03 on D_obs - D_pred; small
  Gabors as replication; ds006171 (c = 0.5, orientation) as a check that a ≈ c predicts D ≈ 0.
- Interpretation rule: D_obs - D_pred not different from 0 and equivalent = consistent with the
  criterion account; D_obs significantly below D_pred = informed unseen trials carry less of the
  signature than weak seen ones (evidence for processing without awareness); above = not predicted,
  reported as such.
- Stated limits: the mixture assumptions; rating-2 as the reference class.

## 5. Q04 Attention, contrast-matched (ds006171 detector) — idea 006

- Hypothesis (equivalence): the cue does not change the detector's seen/unseen AUC.
- Data: `derivatives/sv_decode_group/loso_predictions.tsv` + staged `gabor_contrast`.
- Matching: within subject x task, contrast deciles, the larger class subsampled per bin with a fixed
  seed 97 (the Phase 1 rule, `start_vision.matching.match` defaults: 10 bins, seed 97); then per subject x task AUC.
- **[P]** Informative minus nocue: TOST ±0.03 (equivalence p). Secondary: noninformative minus
  nocue (TOST), the difference tests, the mean contrast per task.
- Deferred: the VAN part (needs the ds006171 epochs, `runs/`).

## 6. Q37a Trial-level TMS-evoked-response amplitude (GIN) — idea 008

- Question: does the H1 difference survive controlling for how large each pulse's evoked response
  was, and does that amplitude itself differ by report?
- Data: `derivatives/sv_p3_clean/primary/` included recordings; detector input (contract channels and
  window).
- Template: the recording's mean over all kept trials (label-free), 0-0.6 s, all contract channels.
  Trial amplitude: the least-squares projection coefficient of the trial on the template over
  0-0.6 s, <trial, template> / <template, template> (one scalar per trial; 1 = as large as the mean).
- **[P]** H1 with P(seen) residualized on TEP amplitude within recording (as A1): AUC > 0.5, one-sided.
- Secondary: AUC of TEP amplitude alone for S100 vs S200 (two-sided); the SSP-3 result already
  computed is reported as context.
- Interpretation rule: H1 survives = the effect is not the trial-to-trial size of the pulse response;
  if TEP amplitude alone separates the classes, report that the phosphene modulates the evoked
  response (Taylor 2010-like) and that residualizing then removes part of a real effect.

## 7. Q37b Pulses needed, independence model, O1 -> O2 (GIN) — idea 009 (amendment 1)

- Single-pulse AUC per recording (primary scores). Independence model: d' = √2 Φ⁻¹(AUC);
  predicted block AUC_k = Φ(d' √k / √2) for k = 1, 5, 10, 20.
- Observed block AUC_k: as the first look (1000 resampled blocks per class, without replacement).
- Secondary: observed minus predicted per k (sign flip over subjects).
- k* = the smallest k in {1, 5, 10, 20} with group mean observed AUC >= 0.75, chosen on **O1
  recordings only**; "not reached" if none.
- **[P]** On O2 recordings: block AUC at k* (or at k = 20 if not reached) > 0.5, one-sided; report
  the value as the planning estimate.
- Stated limits: subjects with single-pulse AUC < 0.5 (reported, not excluded); O1 and O2 are
  mostly the same subjects, so O2 is a held-out set of recordings, not of people.

## 9. Q38 Phosphene VAN topography and a wider response-hand control (GIN) — idea 013 (amendment 2)

- Honesty: all channels at both sites were seen in the exploration of 2026-11-09
  (`results/questions_to_solve/van_topography_explore/`); the VAN parts are descriptive. The hand
  control is the inferential part.
- **[P]** Hand-lateralization covariate: per trial, the hand index = mean over mirrored channel pairs
  (left minus right) of all lateral contract channels (every odd-numbered channel minus its
  even-numbered mirror, e.g. C3-C4, P7-P8, PO7-PO8, TP7-TP8; midline excluded), mean 0.15-0.6 s
  (the detector window), baseline -0.2..-0.01 s (contract). H1 with P(seen) residualized on the
  hand index within recording (as A1): AUC > 0.5, one-sided. Joins the Holm family (seven primary tests).
- Secondary: (a) H1 with all lateral channels of both hemispheres interpolated from the midline
  and the rest (R1 logic, widened); (b) AUC of the hand index alone, two-sided; (c) per site.
- Descriptive (seen data): VAN at Oz 180-220 ms (Samaha 2017, preprint Methods) and the S6 ROI,
  per site; LP Pz 300-800 ms (Samaha) next to S6's 350-550 ms; site maps 180-280 ms.
- Interpretation rule: H1 survives the hand covariate = the effect is not carried by the lateralized
  hand signal on any channel pair; if it does not survive, the claim narrows to "not separable from
  the response". A null after (a) is ambiguous (posterior signal removed too).

## 10. Q39 Time-shifted transfer of the natural-vision code to TMS phosphenes (GIN) — amendment 3

- Motivation: after parietal TMS, VAN-like lateral occipital activity appears earlier than after a
  visual stimulus because the retina is bypassed (Knight et al. 2024, Neuropsychologia 198:108864);
  perception-to-imagery overlap appears only off the diagonal of a train-time x test-time matrix
  (Dijkstra et al. 2018, eLife 7:e33904). S7 trained and tested at the same time point only.
- Honesty: the diagonal of the matrix is S7 (seen 2026-11-10, cluster 148-992 ms). No off-diagonal
  cell has been computed on the TMS data. Written 2026-11-11, before any run.
- Method: the S7 decoder (Phase 2 per-time-point recipe: StandardScaler + L2 logistic regression,
  liblinear, C = 1, balanced; `tms-robust` 64 channels; label-free per-subject baseline and
  prestimulus z-scoring as in Phase 2 LOSO) trained on all 34 included ds006171 subjects
  (stimulus-present trials) as an `mne.decoding.GeneralizingEstimator`, i.e. one classifier per
  training time; each is tested at every time point of each included TMS recording (primary
  pipeline; paperlike as secondary). Train and test times 0-1.0 s at 128 Hz. Per recording one AUC
  matrix; subject = mean of included sites. No TMS labels enter training.
- **[P]** Earlier transfer of the VAN-window code: the mean AUC of classifiers trained at 0.18-0.28 s
  (VAN window, Eklund) and tested at 0.08-0.18 s (100 ms earlier) > 0.5, one-sided sign-flip test
  over subjects (10 000), bootstrap 95 % CI, d_z. Own test at alpha 0.05; not added to the
  seven-test Holm family (which is closed).
- Secondary (uncorrected, labelled): (a) shift asymmetry for the same training window: mean AUC at
  test 0.08-0.18 s minus mean AUC at test 0.28-0.38 s, two-sided; (b) the same [P] statistic for the
  LP-trained classifiers (train 0.35-0.55 s, test 0.25-0.35 s); (c) per subject and over the group,
  the test time of maximal mean AUC for VAN-window-trained classifiers (descriptive); (d) a
  two-dimensional one-sided cluster permutation test over the full matrix (10 000), descriptive map;
  (e) paperlike pipeline.
- Interpretation rule: [P] significant and (a) positive = the natural VAN-window pattern appears
  earlier after TMS than after a visual stimulus (consistent with bypassing the retina). [P]
  significant with (a) not positive = broad generalization, not a specific shift. [P] null = no
  evidence of earlier transfer; the paper keeps "partial match" without a time-shift claim.
- Stated limits: train and test times share the same epoch grid; an earlier window can also carry
  signal from the pulse-evoked response common to both classes, which shifts levels but not the
  phosphene-vs-none ranking (as for H1).

## 11. Q40 Reverse transfer: TMS-trained decoder tested on natural vision (GIN -> ds006171) — amendment 4

- Motivation: cross-decoding in both directions (Dijkstra et al. 2018, eLife 7:e33904). In the TMS
  data the report is tied to the response hand (phosphene = right hand, Bonfanti et al. 2024); in
  ds006171 the yes/no screen positions were randomized on every trial (Melcon et al. 2024), so the
  hand cannot predict seen/unseen there. A TMS-trained decoder that reads natural seen/unseen
  therefore cannot do so through the response hand.
- Honesty: no decoder has been trained on TMS labels and tested on ds006171 before. Written
  2026-11-11, before any run.
- Method: the Phase 2 per-time-point recipe (`across.transfer`: label-free per-subject baseline and
  prestimulus z-scoring, StandardScaler + L2 logistic regression, liblinear, C = 1, balanced),
  `tms-robust` 64 channels harmonized across datasets, trained on all included TMS recordings of the
  primary pipeline pooled (labels phosphene/none), tested on each of the 34 included ds006171
  subjects (stimulus-present trials, seen/unseen). Unit = ds006171 subject. Paperlike pipeline as
  secondary.
- **[P]** Mean AUC over 0.15-0.60 s (the detector window) > 0.5, one-sided sign-flip over the 34
  subjects (10 000), bootstrap 95 % CI, d_z. Own test at alpha 0.05, outside the closed Holm family.
- Secondary (uncorrected, labelled): (a) one-sided cluster permutation over time (0-1.0 s);
  (b) window means VAN 0.18-0.28 s and LP 0.35-0.55 s; (c) prestimulus -0.5-0 s (expected null);
  (d) paperlike pipeline.
- Interpretation rule: [P] significant = the phosphene-vs-none pattern carries information about
  natural seeing that cannot come from the response hand; the shared signal is two-way. [P] null =
  no evidence of reverse transfer (expected to be weaker: 17 training subjects, a small effect, and
  pulse-evoked activity common to both classes); H1 and Q38 stand unchanged.
- Stated limits: the TMS decoder may also learn report- and hand-related activity; such features
  are uninformative in ds006171 and lower, not raise, the reverse AUC.

## 12. Q41 The frozen detector on parietal (IPS) phosphenes (GIN IPS_parietal_phosphenes) — amendment 5

- Motivation: H1 was shown for occipital phosphenes, where TMS starts in early visual cortex, the
  cortex the detector's training signal comes from. Bonfanti et al. 2025 (Clin Neurophysiol
  174:131-147, doi 10.1016/j.clinph.2025.03.041, read in full 2026-11-12) report phosphenes from
  left and right IPS stimulation with sources confined to parietal cortex. If the natural-vision
  signature also separates IPS phosphene pulses, it appears whichever area the evoked percept
  starts in (Dembski et al. 2021, p. 667). Bonus: response keys were counterbalanced across
  participants (m right hand / z left hand, paper p. 133), so if the codes encode the answer, the
  hand is no longer tied to the label at group level.
- Data: GIN `DB_123/IPS_parietal_phosphenes`, CC0; 19 subjects x 2 sites (P3 = left IPS, P4 = right
  IPS), 360 pulses per site at phosphene threshold (method of constant stimuli, intensity adjusted
  during the run to keep ~50 % reports; per-pulse intensity is not known to be logged); same
  64-channel montage, 5000 Hz, events S100/S200 as the occipital set. Raw data are processed one
  recording at a time (download, clean, save epochs, delete the raw `.fdt`; 80 GB free vs ~168 GB).
- Honesty: written 2026-11-12, before any IPS EEG data were cleaned, scored or plotted. Seen so far:
  the paper; the event counts of one file (sub-01 P3: S100 111, S200 224); all 38 `.set` headers
  downloaded but their event counts **not yet tabulated** when the label rule below was fixed.
- Labels: S100 = phosphene, S200 = none (inferred: S100 share 0.33 in the one file seen vs the
  paper's phosphene rates 37.5 % P3 / 32.8 % P4). Code count over all recordings
  (`scripts/ips_code_count.py`, event counts only): if S100 is the minority in every recording,
  the mapping is taken as answer-coded. A recording with S100 share > 0.5 is excluded as
  label-uncertain (reported). If more than a third of recordings have S100 share > 0.5, the codes
  are taken as key-coded, the analysis stops and the authors are asked (after their reply to the
  pending email; no follow-up before).
- Response-hand mapping per recording: the occipital code-direction check (C3-C4, 0.1-10 Hz,
  baseline -0.2..0 s, mean 0.3-1.2 s, S100 minus S200, 5000 permutations) is run, but with
  counterbalanced keys its sign is expected to vary across subjects, so it is **not** an exclusion
  rule here. Its sign (p < 0.05) assigns each subject to a mapping group (phosphene = right hand:
  negative; = left hand: positive; else unassigned), used only in secondary (c).
- Pipeline: the occipital primary pipeline unchanged (`configs/phase3.yaml` at tag `phase3-v1.2`,
  `start_vision.tms` C0-C12), same trial rejection and inclusion (>= 40 kept trials, >= 30 per class),
  frozen `sv-perception-tms-robust-v1.0.0`, missing contract channels interpolated. The stimulated
  electrode is P3 or P4 (not O1/O2); no other change.
- **[P]** Per recording AUC(P(seen), S100 vs S200); subject AUC = mean of included sites;
  one-sided sign-flip permutation of subject AUC - 0.5 (10 000, seed 20261109), bootstrap 95 % CI,
  d_z, n > 0.5. Own test at alpha 0.05, outside the Holm family and separate from the occipital H1
  (no pooling of the two datasets: whether participants overlap is unknown).
- Secondary (uncorrected, labelled): (a) per site P3 and P4, and P3 vs P4 (Wilcoxon);
  (b) `paperlike` and `gonogo` pipelines; (c) **hand split**: subject AUC in the two mapping groups,
  Mann-Whitney two-sided; a hand-driven effect would reverse between groups; (d) the occipital
  controls unchanged: R1 (18 central channels interpolated), R5 (C3-C4 alone), Q38 hand-index
  residualization, E1 (EOG alone; HEOG step < 16 µV rerun), A1 (prestimulus alpha), Q37a
  (TEP amplitude residualization); (e) R4a/R4b early-only detectors; (f) S9 phosphene share 30-70 %;
  (g) channel lesion of the parietal channels around the coil (P1 P3 P5 CP1 CP3 CP5 PO3 PO7 and their
  right homologues) interpolated, as the occipital exploratory lesion.
- Interpretation rule: [P] significant = the natural-vision signature separates percepts evoked from
  parietal cortex too; not limited to stimulation of early visual cortex. [P] null = no evidence;
  either the overlap depends on where the percept starts, or the fainter parietal phosphenes
  (paper p. 135, slower "yes") carry too little of it; the two cannot be told apart here. The
  occipital H1 stands unchanged either way.
- Stated limits: same lab and design as the occipital set (no sham, no catch pulses); intensity
  drifted within a run; the stimulated electrode sits under a C-shaped annulus.

## 13. Q42 Latency-tolerant features for the natural-to-TMS transfer (GIN) — idea 37e, amendment 6

- Motivation: Q39 (exploratory follow-up) put the TMS transfer peak near 0.48 s against ~0.27 s in
  natural vision. The frozen detector reads 50 ms bins at fixed latencies (0.15-0.60 s, 9 bins x 64
  channels), so a latency shift moves signal into the wrong weights. If weak transfer is partly a
  timing problem, features that ignore latency inside the window should transfer better.
- Honesty: written 2026-11-15, before any run. Seen: H1, S7, Q39 (diagonal, off-diagonal, peak
  ~0.48 s). No model below has been fitted or scored on the TMS data.
- Training: all 34 included ds006171 subjects, stimulus-present trials, `tms-robust` 64 channels,
  label-free per-subject normalization as Phase 2 (`across.normalize`); StandardScaler + L2 logistic
  regression (liblinear, balanced) with **C = 0.01, the frozen detector's C**; no C search, no TMS
  labels. Testing: included GIN recordings, primary pipeline (paperlike secondary); per recording
  AUC(score, seen); subject = mean of included sites.
- Models (all identical except the features):
  - **M0 matched baseline**: 0.15-0.60 s, 50 ms bin means (the frozen detector's features, refitted
    here so that only the features differ).
  - **M1 broad mean**: 0.15-0.60 s, one bin (the mean over the whole window per channel).
  - M2 shifted: M0 trained on 0.15-0.60 s, applied to the TMS data at 0.35-0.80 s (+0.20 s, the
    rounded Q39 peak difference; the shift comes from an exploratory look at the same TMS data, so
    M2 is descriptive only).
  - M3 covariance: per-trial 64 x 64 covariance over 0.15-0.60 s (shrinkage 0.1 toward the scaled
    identity), log-Euclidean tangent vector (upper triangle of the matrix log, off-diagonals x sqrt 2).
- **[P]** Per subject AUC(M1) - AUC(M0) > 0, one-sided sign-flip (10 000, seed 20261109), bootstrap
  95 % CI, d_z, n positive. Own test at alpha 0.05, outside the closed Holm family.
- Secondary (uncorrected, labelled): (a) M1, M2, M3 and M0 each vs 0.5 (one-sided); (b) M2 - M0 and
  M3 - M0 (two-sided); (c) paperlike pipeline for all; (d) natural LOSO AUC of M0, M1, M3 on
  ds006171 (does latency tolerance cost accuracy in the training domain).
- Interpretation rule: [P] significant = part of the transfer loss is latency; [P] null = timing
  inside the window does not limit transfer (consistent with Q39 "broad"), so the gap is in what
  is coded (content, domain), pointing to 37g/37h. M2 cannot confirm a lag (circular).
- Stated limits: one fixed window; covariance mixes evoked and induced activity; same TMS data as Q39.

## 14. Q43 Label-free domain alignment of the TMS data to the training data (GIN) — idea 37h, amendment 7

- Motivation: Q42 left most of the transfer gap unexplained by timing. Cap, amplifier, lab, the
  pulse and the stimulus all differ between ds006171 and GIN; a shift of the feature distribution
  between domains moves trials across a linear boundary even when the seen/unseen direction is
  shared. Aligning each subject's feature distribution to a common reference without labels tests
  whether the gap is a domain shift.
- Honesty: written 2026-11-15, after Q42 (seen: M0, M1, M3 TMS and natural AUCs). No aligned model
  has been fitted or scored.
- Base models and training exactly as Q42 (34 ds006171 subjects, tms-robust 64 ch, Phase 2
  normalization, StandardScaler + L2 LR, liblinear, balanced, C = 0.01).
- Alignment (label-free, per unit: each training subject and each TMS recording separately, all its
  trials): features centred on the unit mean and whitened with the unit's Ledoit-Wolf covariance
  (CORAL-style alignment of every unit to zero mean and identity covariance). For M3 (covariance)
  the equivalent is re-centring: subtract the unit's mean log-Euclidean tangent vector.
- **[P]** M0 (50 ms bins 0.15-0.60 s): per subject AUC(aligned M0) - AUC(M0) > 0, one-sided
  sign-flip (10 000, seed 20261109), bootstrap 95 % CI, d_z, n positive. Own test at alpha 0.05.
- Secondary (uncorrected, labelled): (a) aligned M1 - M1 and re-centred M3 - M3 (two-sided);
  (b) each aligned model vs 0.5; (c) paperlike pipeline; (d) natural LOSO AUC with alignment (does
  per-subject alignment help across natural subjects too).
- Interpretation rule: [P] significant = part of the gap is a label-free domain shift that can be
  corrected per recording without labels (useful for a future stimulation study). Null = the shared
  seen/unseen direction, not the distribution, limits transfer (content, 37g).
- Stated limits: alignment uses all of a recording's trials, so the seen share of the recording
  enters the unit mean (label-free but class-balance dependent; reported share 30-70 %).

## 8. Deviations and changes
- **Amendment 1 (2026-11-09, user):** Q37b (idea 009) added after the tag `questions-prereg`, before
  any analysis of this addendum was rerun; the Holm family grows from five to six primary tests.
  Tag `questions-prereg-amend1`.
- **Implementation clarifications (2026-11-09, user, before any rerun):** the twelve answers in
  `plans/QUESTIONS_BLUEPRINT.md` §17 "Answers" fix details this addendum leaves open (Q01/Q02 cell
  rules, Q02 covariate centring and tail, ds006171 inclusion = 34 subjects, Q03 ds006171 reporting,
  Q04 matching and family reading, Q37a inputs, Q37b generator per recording, pooling and O2
  entry). Two of them border on the analysis: the extra >= 3 correct trials for Q02, and the Q37b
  generator scheme; both recorded here as decided before the rerun.
- **Amendment 2 (2026-11-09, user):** Q38 added (idea 013) before any rerun; Holm family seven
  primary tests; tag `questions-prereg-amend2`.
- **Interpretation note Q37b (2026-11-09, found while building, before any rerun):** the blocks are
  random draws within class (as the first look), so dependence between neighbouring pulses cannot
  enter them; the secondary "observed minus predicted" therefore tests whether the single-pulse
  scores fit the model's equal-variance Gaussian shape, not pulse independence. The analysis is
  unchanged; only its reading is corrected. The [P] test (O2 block AUC) is unaffected.
- **Amendment 3 (2026-11-11, user):** Q39 added (time-shifted transfer, §10) after reading Knight
  et al. 2024 and Dijkstra et al. 2018, before any off-diagonal cell was computed; its own primary
  test at alpha 0.05, outside the closed seven-test Holm family. Tag `questions-prereg-amend3`.
- **Amendment 4 (2026-11-11, user):** Q40 added (reverse transfer, §11) before any TMS-trained
  decoder was tested on ds006171; own primary test at alpha 0.05. Tag `questions-prereg-amend4`.
- **Amendment 5 (2026-11-12, user):** Q41 added (IPS parietal phosphenes, §12) after reading
  Bonfanti et al. 2025 and seeing one file's event counts, before any IPS EEG data were cleaned or
  scored and before the code count over all recordings was tabulated; own primary test at alpha
  0.05. Tag `questions-prereg-amend5`.
