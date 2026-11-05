# Phase 1 reference values (PHASE1_BLUEPRINT §8 Step 2)

Every window, site and number in `configs/` comes from a line in this file, which cites its
source. Nothing here was typed from memory. Built 2026-11-05.

**Primary source for the windows and sites (user choice, 2026-11-05):** Eklund R, Wiens S (2018).
*Visual awareness negativity is an early neural correlate of awareness: a preregistered study with
two Gabor sizes.* Cogn Affect Behav Neurosci 18:176-188. doi:10.3758/s13415-018-0562-z. Read in
full on PMC (PMC5823961, CC BY), and checked against the authors' own analysis code
(`fieldtrip/analyze_VAN.m`, figshare doi:10.17045/sthlmuni.5418967).
**Why this source:** it is preregistered and uses threshold-calibrated Gabors that are lateralized,
like ds006171. Its windows are fixed in advance (not chosen from the data), and its raw data are
our replication set. The review by Dembski et al. 2021 (Table 1) is used only as a cross-check.

Section names below refer to the PMC full text; the PMC version has no journal page numbers.

## 1. Dataset paper (ds006171): Melcón et al. 2024, Psychophysiology 61:e14525

| Fact | Value | Source |
|---|---|---|
| Recording | 128 EEG channels + 4 external (above/below one eye, lateral to both eyes) + a nose-tip electrode; 1024 Hz; online low-pass at 205 Hz | §2.4 EEG recording |
| Sample | 36 recorded; EEG analysed in 29 / 28 / 31 (no-cue / noninformative / informative); 25 have all three | §2.1 Participants |
| Exclusion | participants with > 45 % of trials with ocular artifacts (5 / 6 / 5 per task) | §2.5.2 Preprocessing |
| Preprocessing | epochs -2000..1800 ms; visual rejection (eye, cable, swallowing, muscle); ICA (runica); interpolation of noisy electrodes; down-sampled to 512 Hz; linear detrend; baseline -500..0 ms; common average reference | §2.5.2 Preprocessing |
| Post-stimulus ERPs | **none reported** (the paper analyses prestimulus alpha amplitude and phase) | §2.5.2 aims |
| Pipeline per task | applied separately to each task | §2.5.2 |

The -500..0 ms baseline serves their alpha analysis; Phase 1 follows the ERP source (§2 below).

## 2. Primary ERP source: Eklund & Wiens 2018

| Fact | Value | Source |
|---|---|---|
| Conditions | aware correct = correctly localized and rated "saw weakly" **or** "saw clearly"; unaware correct = correctly localized, rated "did not see"; VAN/LP = aware correct minus unaware correct | ERP analysis |
| **VAN window** | **180-280 ms** after Gabor onset (preregistered "N200" interval) | ERP analysis; `analyze_VAN.m:26` (`N200int = [0.18 0.28]`) |
| **VAN sites** | **contralateral occipital only**: O1 for right-field Gabors, O2 for left-field Gabors ("We considered only contralateral electrodes because cortical activity is maximal contralateral to visual stimulation") | ERP analysis; `analyze_VAN.m:24-25` |
| **LP window** | **350-550 ms** | ERP analysis; `analyze_VAN.m:29` |
| **LP sites** | **P3, Pz, P4** pooled | ERP analysis; `analyze_VAN.m:28` (`LPchan = {'P3','Pz','P4'}`). The Results text and Fig. 4 say "P1, Pz, and P2": a typo in the paper; the code and the Methods agree on P3/Pz/P4 |
| Reference | average of the 64 electrodes + nose | Data analysis |
| High-pass | 0.1 Hz, Butterworth, 4th order, two-pass | Data analysis |
| Low-pass | none for measures; plots low-pass filtered at 30 Hz | Data analysis; Fig. 2 caption |
| Epoch / baseline | -100..600 ms; baseline **-100..0 ms** | Data analysis |
| Artifacts | individual amplitude-range thresholds, blind to condition; blinks by ICA (runica); interpolation; outlier check again after re-referencing | Data analysis |
| Onsets | photodiode | Data analysis |
| Minimum trials | >= 5 per quadrant (preregistered) | ERP analysis |

### Validation targets (Step 10): direction and magnitude, aware correct minus unaware correct

From Table 2 (mean, SD, 95 % CI, BF10):

| Measure | Large Gabor (N = 24) | Small Gabor (N = 15) |
|---|---|---|
| VAN (contralateral O1/O2, 180-280 ms) | -1.58 µV (SD 1.87), CI [-2.37, -0.79], BF10 = 79 | -1.41 µV (SD 1.14), CI [-2.04, -0.77], BF10 = 111 |
| LP (P3/Pz/P4, 350-550 ms) | +1.35 µV (SD 0.92), CI [0.96, 1.74], BF10 > 62,000 | +1.00 µV (SD 0.91), CI [0.50, 1.50], BF10 = 47 |

The authors' own reading of their data: minimum plausible VAN -0.8 µV, upper about -2.0 µV;
minimum plausible LP +0.5 µV, upper about +1.5 µV (Results, VAN and LP sections).
**Step 10 target:** the VAN is negative and the LP positive; magnitudes overlap these ranges.
Differences that may change the magnitude: ds006171 uses contrast rather than duration,
hemifields rather than quadrants, a yes/no report rather than a 3-point rating, 128 channels,
and a different average reference.

## 3. Cross-check: Dembski, Koch & Pitts 2021 (TICS 25:660-670), read in full

- Table 1 (p. 667): VAN 170-290 ms, average peak 230 ms, occipital (16 studies). Eklund's
  180-280 ms lies inside it.
- p. 664: "the magnitude of the VAN is largest over regions contralateral to the inducing
  stimulus"; p. 664-665: 5 of 6 failed VAN studies pooled hemispheres.

## 4. Site mapping: 10-20 names to ds006171's 128-channel ABC cap

Method: `sub-01_electrodes.tsv` (ds006171, fetched from S3 on 2026-11-05) was aligned to MNE
`standard_1005` by Procrustes over its 128 channel names (median residual 8.7 mm), then the
nearest cap electrode was taken for each target.

| Source site | ds006171 channel | Distance | Note |
|---|---|---|---|
| O1 | O1 | 5.2 mm | same name |
| O2 | O2 | 5.1 mm | same name |
| Pz | Pz | 8.5 mm | same name |
| P3 | **CPP3** | 10.8 mm | next nearest PPO3 (15.8 mm) |
| P4 | **CPP4** | 10.4 mm | next nearest PPO4 (16.6 mm) |

## 5. Values for `configs/` (filled 2026-11-05)

| Config key | Value | Source line |
|---|---|---|
| `dataset.authors` | Melcón, Stern, Arana, Capilla | `docs/DATASETS.md` (dataset_description.json) |
| `reference.scheme` | average (128 EEG) | §1 Melcón (common average) and §2 Eklund (average) agree |
| `preproc.highpass_hz` | 0.1 | §2 Eklund |
| `preproc.lowpass_hz` | null (measure unfiltered; plot at 30 Hz) | §2 Eklund |
| `preproc.line_noise` | eegunify default | neither source states a notch filter: blueprint fallback |
| `epochs.sets.stim.baseline_ms` | [-100, 0] | §2 Eklund |
| `contrasts.*_lat.pairs` | [[O1, O2]] | §2 Eklund (contralateral O1/O2) |
| `van@awareness.window_ms` / `rois` | [180, 280] / O1, O2 (contralateral) | §2 Eklund |
| `lp@awareness.window_ms` / `rois` | [350, 550] / Pz, CPP3, CPP4 | §2 Eklund + §4 mapping |
| HEOG step rule `threshold_uv` | **32 µV, project choice (user decision 5, 2026-11-05)**, window -50..350 ms | neither source gives one (Eklund: individual amplitude ranges; Melcón: visual inspection). 32 µV is eegunify's ERP CORE N2pc config value, labelled there as a "Blueprint default": **not a literature value**. Version A uses it; version B (ICA only, no HEOG rule) is decided from A's outcome |

## 6. VAN measure form: decided (user decision 4, 2026-11-05): contralateral is primary

Eklund (and Dembski Fig. 2) measure the VAN at the **contralateral** electrode only:
contra(seen) minus contra(unseen). Idea 001 set the primary measure as the **double difference**:
(contra minus ipsi)(seen) minus the same for unseen. Both come from the same eegunify
`lateralized` waves (`seen_contra`, `unseen_contra`, `*_ipsi`), so either costs nothing. But only
the contralateral form can be compared with the Step 10 targets above.

**Decision:** the primary measure is contra(seen_m) minus contra(unseen_m) at O1/O2 in 180-280 ms,
the same as Eklund. The double difference is secondary. Computed in `analysis/lateral.py` from
the eegunify `lateralized` parent waves.
