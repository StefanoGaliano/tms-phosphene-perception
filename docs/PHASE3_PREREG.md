# Phase 3 pre-registration (GIN Phosphenes_occipital, Siviero et al. 2023)

Committed and tagged `phase3-prereg` on 2026-11-08, before any O2 recording or any O1 subject
>= 13 is cleaned or scored. Only O1 subjects 03-11 have been looked at (go/no-go with rough
cleaning, `results/phase3/gonogo/`; cleaning QA on sub-03/04, `results/phase3/notes_clean_qa.md`).
Sections below are copied from `plans/PHASE3_BLUEPRINT.md` (§6, §7, §8, §9, §10) and fixed here;
any later change is a reported deviation.

## Final answers and parameters (user, 2026-11-08; README Decisions 10-12)

- Detector `sv-perception-tms-robust-v1.0.0`, unchanged; 5 missing channels interpolated (Decision 10).
- Labels S100 = present, S200 = absent (inferred; authors asked 2026-11-08). If they contradict:
  swap and rerun, reported (Q9).
- Primary pipeline = **training-like** (Q5 option A, chosen at the Step 9 stop from the muscle
  measure: GIN ongoing 20-35 Hz edge power within the ds006171 range). The paper's muscle ICA
  round is the `paperlike` sensitivity pipeline (S5); REST is not available in MNE, so
  `paperlike` uses the average reference (recorded).
- Cleaning parameters (configs/phase3.yaml at this tag; sources docs/PHASE3_REFERENCES.md):
  cut -2..10 ms cubic (paper), extended to the measured artifact end up to 20 ms; artifact end
  k = 3, hold 2 ms; ICA at 1000 Hz; decay ICA TESA rule 11-30 ms, ratio >= 8. The four project
  choices were approved by the user at the Step 9 stop.
- Trial rejection as §6 unchanged (user: keep it, also with sub-04's 32 % rejection).
- Q1-Q4, Q6-Q11 as recommended in the blueprint §18: AUC primary; contract θ 0.521 for binary
  metrics, subject-adaptive median as alternative; no phosphene-rate band in the primary, 30-70 %
  as S9; subject AUC = mean of included site AUCs; one-sided group test, two-sided reported;
  R4 early cut 0.30 s; go/no-go recordings included, S1 untouched subset as confirmation;
  DLPFC dataset only if QA shows residual artifact beyond 150 ms; sub-18 O1 excluded if its
  pairing is unresolved; parameters without a source approved one by one.
- Data: 18 subjects per site on GIN (03-11, 13-19, 21, 22), 36 recordings.

## 6. Labels, recording and subject inclusion (fixed in the pre-registration)

- Trials: every `S100` / `S200` marker; `boundary` ignored; any other code -> error.
- **Code-direction check per recording** (the go/no-go method, `notes_data_check.md` §1):
  C3-C4, 0.1-10 Hz, pulse-locked, baseline -0.2..0 s, mean 0.3-1.2 s, S100 minus S200,
  5000 trial-label permutations. A recording whose difference is **positive with p < 0.05**
  (opposite to the inferred mapping) is excluded as label-uncertain. Negative or non-significant:
  kept. Sign-test across subjects reported.
- **Trial rejection** (after cleaning, before the detector; same tests as the training data,
  restricted to the windows the detector reads, so the TMS-evoked response < 150 ms does not
  reject trials): peak-to-peak 100 µV (user decision 8 value) on EEG channels within -0.5..-0.01 s
  and 0.15..0.6 s (moving 200 ms windows, 100 ms step); HEOG step 32 µV (user decision 5 value)
  within -0.05..0.35 s, with HEOG = HEOGL - HEOGR (real canthal electrodes here; if one is broken,
  F7-F8 as in decision 7, recorded).
- **Recording inclusion**: >= 40 kept trials (contract `min_trials_for_normalization`) and >= 30
  kept trials per class (Phase 2 minimum); not excluded by the code-direction check or Step 1.
- **Subject inclusion**: at least one included recording. Phosphene-rate band: see Q2
  (recommended: no band in the primary, 30-70 % per recording as a sensitivity analysis; a
  false-alarm rule is impossible without catch trials, reported as a limit).
- `results/phase3/inclusion.tsv`: per recording counts before/after rejection, S100 share, code
  check difference and p, included yes/no with reason.

## 7. TMS-EEG cleaning chain (`start_vision.tms` + `phase3.clean`), per recording

Order is fixed; label-blind (no step receives S100/S200; test enforces it). All numbers below are
either from Siviero 2023 2.4 (cited "paper"), the detector contract ("contract"), eegunify defaults
used in training ("training"), or `TODO_FROM_SOURCE`.

| # | Step | Parameters | Source |
|---|---|---|---|
| C0 | Read `.set` with MNE, 5000 Hz; set channel types (EEG 59, EOG 4, LM -> misc, dropped); template 10-05 positions; build VEOG = VEOGUP-VEOGDOWN, HEOG = HEOGL-HEOGR | — | data |
| C1 | **Measure the pulse artifact** on this recording: median over pulses of |x(t) - baseline mean| per channel, pulse-locked, 5000 Hz; artifact end = first sample after 0 where the 95th percentile over channels stays below `k` x prestimulus SD for `m` ms | `k`, `m`: TODO_FROM_SOURCE (or a project choice recorded as such) | QA |
| C2 | **Cut and interpolate the pulse**: window -2..+10 ms; if C1's measured end > 10 ms, extend to the measured end, capped at `max_cut_ms`; cubic interpolation from the samples on both sides (as the paper), done on the continuous data | -2..10 ms cubic: paper; cap: TODO_FROM_SOURCE | paper |
| C3 | Resample to 1000 Hz (MNE polyphase, events resampled with the data) | project choice (keeps decay resolution, makes ICA tractable) | — |
| C4 | **Decay/TMS ICA (round 1)**: epochs -1..1 s on a 1 Hz high-passed copy, the pulse window set to a constant before the fit and re-interpolated after (paper); algorithm as training (picard extended, seed 97); components flagged by a TESA-style rule: mean |component| in an early window much larger than in the rest of the epoch | early window, ratio threshold: TODO_FROM_SOURCE (Rogasch et al. 2017 NeuroImage, TESA, to be fetched and read in full) | paper + TESA |
| C5 | Apply the round-1 unmixing to the continuous 1000 Hz data, re-interpolate the pulse window (C2) | — | paper |
| C6 | High-pass 0.1 Hz, IIR order 2 zero-phase; line noise as eegunify's default in training | training (`configs/template_ds006171.yaml` `preproc`) | training |
| C7 | Bad channels: pyprep robust, interpolate (as training); pulse windows excluded from the detection data | training | training |
| C8 | **Ocular ICA**: `eegunify.erp.ica.run_ica` with the training artifacts section (`eog_bipolar` from C0 channels), `fit_exclude` default (ptp 400 µV spans) plus the pulse windows; components removed per eegunify's decision | training | training |
| C9 | Epochs -1..1 s pulse-locked, no baseline; trial rejection (§6) | — | §6 |
| C10 | Low-pass 40 Hz (IIR Butterworth order 4, zero-phase, as `decoding.data.prepare`), resample to 128 Hz | contract | contract |
| C11 | Average reference over the recorded, non-bad scalp EEG channels; then `complete_contract_channels` (interpolate P9, P10, Iz, Fpz, AFz) | contract + Decision 10 | — |
| C12 | Write `derivatives/sv_p3_clean/sub-NN_site-SS_desc-detector_epo.fif` (64 contract channels + VEOG/HEOG + C3/C4 copies kept as misc for controls) and `..._desc-qa.json` (artifact end, cut window, n ICA1/ICA2 components, bad channels, rejection counts, timings) | — | — |

Note on C11: the go/no-go interpolated first and then referenced over 64 channels; the primary
references over recorded channels first (contract wording: "average over the recording's EEG
channels"). The other order is a one-line sensitivity check reported in the QA notes.

**Sensitivity pipeline `paperlike`** (same code, config switch): C8 replaced by round 2 removing
blink **and** muscle components (ICLabel classes "eye blink", "muscle artifact" at the training
threshold) on 0.1-100 Hz + 49-51 Hz band-stop data (paper), REST reference instead of average
(requires a REST implementation; if none is available in MNE, drop this element and record it),
trial rejection as §6. **Rough pipeline `gonogo`**: the existing `gonogo.py` recipe (linear
interpolation -2..15 ms, no ICA), kept as a third reference point.

**Validation** (no real-data claims depend on unvalidated code):
1. Synthetic (§12): known ERP + injected pulse/decay/muscle artifacts; recovery and invariance tests.
2. Real-data QA on 2 recordings before running all: GMFP and butterfly before/after each step
   (F1), residual artifact end (C1 rerun after C5), number of ICA components removed, visual check
   that the TEP after cleaning has no step or ringing at the cut edges. The paper's TEP figures are
   a qualitative reference only (no numbers transcribed).
3. The DLPFC TMS-EEG dataset (`EXPERIMENT_DESIGN.md`) is **optional**; recommended only if
   synthetic tests pass but real-data QA shows a residual artifact beyond 150 ms (Q8).

## 8. Analyses (pre-registered in `docs/PHASE3_PREREG.md`)

All per recording on the detector-ready epochs; subject = unit; nothing fitted outside a training
fold (the frozen detector fits nothing; retrained analyses use subject- or trial-level folds that
never contain the test data); chance by permutation.

### Primary (H1)

P(seen) from `Detector.from_epochs` (D4, D5). Per recording AUC(S100 vs S200) with a per-recording
label-permutation p (2000 permutations). Subject AUC = mean of included site AUCs. Group test:
one-sided sign-flip permutation on subject AUC - 0.5 (10 000), with mean, bootstrap 95 % CI over
subjects, d_z, number of subjects > 0.5, two-sided p reported too. H1 is supported if one-sided
p < 0.05.

### Secondary (reported whatever they show; Holm correction within this list)

| # | Analysis | Detail |
|---|---|---|
| S1 | Untouched subset | H1 test on recordings not used in the go/no-go (all O2, O1 subjects >= 13). Confirmatory check of the go/no-go. |
| S2 | Site | AUC O1 vs O2, paired across subjects with both sites (Wilcoxon, two-sided). |
| S3 | Binary metrics at the contract threshold | θ = 0.521 (`threshold`, max balanced accuracy): balanced accuracy, hit and false-alarm rates relative to report; and at the subject-adaptive rule (θ = recording median P(seen), `alternative_rules.subject_adaptive`). Rule choice: Q1. |
| S4 | Calibration shift | mean P(seen) per class and overall vs the training reliability curve (`calibration.tsv`); reported, not tested. |
| S5 | Pipeline robustness | H1 with `paperlike` and `gonogo` pipelines. |
| S6 | ERP signatures | seen minus unseen (S100 - S200) at posterior channels contralateral to the phosphene (= the stimulated hemisphere: O1 site -> PO7/O1 side, O2 site -> PO8/O2 side) and LP at Pz, in the VAN and LP windows **taken from `docs/PHASE1_REFERENCES.md`** (no new windows); cluster permutation over time as Phase 1. Caveat: stimulated-side channels sit under the coil. |
| S7 | Retrained transfer | `decoding.across.transfer` trained on all ds006171 Phase 2 subjects (time-resolved), tested on each recording: when (if at all) the TMS data carry the ds006171 code. No TMS labels used in training. |
| S8 | Within-subject retrained decoding | `decoding.within` time-resolved S100 vs S200 per recording (stratified 5-fold CV), cluster test over subjects. Shows what the data hold beyond the frozen read-out; it is not evidence for the "naturally-seen" claim. |
| S9 | Phosphene-rate band | H1 restricted to recordings with S100 share 30-70 % (Q2). |

## 9. Confound controls (fixed in advance)

The response hand is perfectly confounded with the label by design (present = right hand,
always). It can be bounded, not removed. Rules for "flagged" are fixed now.

| # | Control | Rule |
|---|---|---|
| R1 | **Motor channels removed**: the 18 fronto-central/central/centro-parietal channels of `gonogo.py` (`FC5 FC3 FC1 C5 C3 C1 CP5 CP3 CP1 FC6 FC4 FC2 C6 C4 C2 CP6 CP4 CP2`) interpolated from the others, then detector applied | Flag if the group AUC loses significance (one-sided p >= 0.05) or drops by more than half of (AUC - 0.5). |
| R2 | **Trial-wise relation to C3-C4**: Spearman ρ between P(seen) and C3-C4 mean in 0.15-0.6 s, within each class, averaged; group sign-flip test on ρ | Flag if group ρ > 0 with p < 0.05 (detector follows the hand signal trial by trial). |
| R3 | **Lateralization of the detector's patterns**: from `patterns.npz` (Haufe), per 50 ms bin, (a) C3-vs-C4 asymmetry of the pattern relative to its posterior magnitude; (b) spatial correlation of the pattern with the TMS data's response-locked lateralization topography (S100 - S200, 0.3-1.2 s, the code-check map) | Descriptive; flag if any bin's |spatial r| exceeds the 95th percentile of r with 1000 random sign-flipped / channel-permuted maps. |
| R4 | **Early window only**: (a) frozen detector with feature bins after 0.30 s set to 0 after the contract normalization (a lesion of the frozen model, no refit); (b) a detector retrained on ds006171 only (Phase 2 recipe, `tms-robust` channels, window 0.15-0.30 s, LOSO-calibrated, not exported to `detectors/`) applied to the TMS data | Reported with H1's test; the cut 0.30 s is a project choice to keep before most responses, to revisit if the authors send RTs (Q6). |
| R5 | **C3-C4 alone** AUC (0.15-0.6 s) | Reference for how much the hand signal itself separates the classes. |
| E1 | **Eyes**: AUC of VEOG/HEOG alone (0.15-0.6 s window features); HEOG-step trial rejection already applied | Flag if EOG-only AUC is significant; then rerun H1 on trials with HEOG step < 16 µV (half the rule) as a check. Phosphenes are lateralized, so a saccade toward the percept is a real risk. |
| A1 | **Prestimulus state**: AUC of prestimulus (-0.5..-0.1 s) alpha power (posterior channels, alpha band as Phase 2 C4) for S100 vs S200; H1 with P(seen) residualized on alpha within recording | Reported; the claim is narrowed to "post-pulse" only if residualizing removes H1. |

## 10. What can and cannot be claimed (goes verbatim into the report)

- **Can**: among physically identical pulses at one intensity, pulses later reported as a phosphene
  carry more of the EEG pattern that marks a naturally seen near-threshold Gabor than pulses
  reported as none (if H1 holds), bounded by R1-R5, E1, A1.
- **Cannot**, without sham or catch pulses: estimate the detector's false-alarm rate on TMS;
  separate the percept from the report, decision or response preparation (hand fixed to label);
  rule out that a fluctuating state (arousal, attention) drives both the report and the signal;
  attribute absolute P(seen) levels to visual cortex rather than click/scalp sensation (these are
  identical in both classes, so they do not explain the *difference*, but they do shift the level).
- Phase 2 C5 caveat applies: on ds006171 catch trials "seen" false alarms scored like correct
  rejections, so the detector reads stimulus-like evidence, not the report itself; a null in TMS
  could mean no percept signal or a percept without that evidence.
- Labels are inferred (D8); RTs unknown; 5 channels interpolated (Decision 10); go/no-go subjects
  re-used (S1 is the clean check).
