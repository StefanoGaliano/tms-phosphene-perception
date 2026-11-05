# Datasets: what was searched, what was verified, what is missing

Search done on 2026-11-02. "Verified" means the facts below were read from the dataset's own files
(`dataset_description.json`, `*_eeg.json`, `*_events.json`, `*_events.tsv`, README) on OpenNeuro's
public S3 mirror (`https://s3.amazonaws.com/openneuro.org/<ds>/...`) or its GraphQL API, not from
search snippets. Anything not checked that way is marked **unverified**.

## Phase 1 and 2: near-threshold visual detection with per-trial reports

### ds006171 — chosen (verified)

- **Title:** EEG data during three near-threshold visual detection tasks: a no-cue task, a
  noninformative cue task (50% validity), and an informative cue task (100% validity).
- **Authors:** María Melcón, Enrique Stern, Lydia Arana, Almudena Capilla (Universidad Autónoma de
  Madrid). **License:** CC0. **DOI:** 10.18112/openneuro.ds006171.v1.0.0 (snapshot 1.0.0).
- **Paper:** Melcón, M., Stern, E., Kessel, D., Arana, L., Poch, C., Campo, P., and Capilla, A.
  (2024). *Perception of near-threshold visual stimuli is influenced by prestimulus alpha-band
  amplitude but not by alpha phase.* Psychophysiology 61, e14525.
  https://doi.org/10.1111/psyp.14525
- **Size:** 36 subjects (`sub-01` .. `sub-36`), about 72.8 GB.
- **Recording:** BioSemi ActiveTwo, 128 EEG channels (CAP 128 ABC layout) + 4 EOG, 1024 Hz,
  reference DRL (reference-free: needs an offline reference), line 50 Hz, `.bdf`.
- **Tasks (all in one session):** `task-nocue`, `task-noninformative` (cue 50 % valid),
  and a third, informative-cue task (100 % valid; exact BIDS task label to read from the files).
- **Trial:** a Gabor at peri-threshold contrast for 50 ms, vertical or horizontal, left or right
  hemifield; no stimulus in 10 % of trials (catch). Contrast is recalibrated trial by trial
  (`gabor_contrast` column). Questions: "Have you seen the Gabor?" (yes/no, every trial), then
  orientation (vertical/horizontal) in a random 15 % of trials. Unlimited response time.
- **Event codes** (`*_events.json`):

  | code | meaning |
  |---|---|
  | 11 / 12 | vertical / horizontal Gabor, left hemifield |
  | 21 / 22 | vertical / horizontal Gabor, right hemifield |
  | 31 / 32 | absent Gabor (catch) after a left / right cue |
  | 128 | photodiode: replace each 128 with the next trigger code (delayed) and ignore the rest |
  | 124 / 125 | subjective response seen / unseen |
  | 121 / 122 | objective response vertical / horizontal |

  The `events.tsv` also has `trial_number`, `gabor_contrast`, `subjective_r` (0/1),
  `objective_r` columns.
- **Counts, sub-01 `task-nocue` (read from the file):** 360 Gabor trials (90 of each code
  11/12/21/22) + 40 catch trials (31/32); 188 "seen" and 212 "unseen" responses; 48 orientation
  responses.
- **Task coverage (from the S3 file listing, all 36 subjects):** 32 subjects have all three
  tasks (`nocue`, `noninformative`, `informative`); 4 have two: 2 x `informative`+`nocue`,
  1 x `informative`+`noninformative`, 1 x `nocue`+`noninformative` (sub-01).
- **`task-noninformative` uses other codes, undocumented:** sub-01 has 1 and 2 (200 each:
  probably the cue left/right), 101-104 and 111-114 (45 each: the Gabors, probably valid/invalid
  cue x side x orientation), 98 and 99 (20 each: probably catch), plus 121/122, 124/125, 128. Its
  `events.json` is a copy of the no-cue one (lists 11-32), so these meanings must be inferred
  (counts, timing relative to the cue, the `gabor_contrast` column) and checked before use.
- **Hit rates (seen / Gabor trials, from all `events.tsv` of `nocue` and `informative`,
  2026-11-02):** median 0.49 (nocue) and 0.46 (informative), maximum 0.55; false alarms on catch
  trials median 0.05 / 0.025. Pooled per subject: 0.20-0.53, median 0.47. This is by design: the
  paper's staircase (PEST + up-down) ends calibration when seen responses are 30-70 % and then
  keeps contrast near threshold. No subject reaches a 70 % hit rate.
- **Paper facts (Melcón et al. 2024, open access, CC BY-NC 4.0, `../papers_to_work/`):** the
  yes/no alternatives' screen position "varied randomly in each trial", answered with the left or
  right thumb on a numerical keyboard; the position is not in the events (only 124/125). The
  paper analyses prestimulus alpha only (no post-stimulus ERPs), so VAN/P3b windows need another
  source. EEG analysed in 29, 28, 31 participants per task (ocular-artifact exclusion > 45 %).
- **To check before use:** `participants.tsv` lists `sub-001`, `sub-002`, ... with ages 60, 67
  and session columns, which does not match the `sub-01` folders or a young-adult sample: it may
  belong to another dataset. The `sample` column of `events.tsv` holds non-numeric characters
  (an encoding problem), so onsets must come from `onset` or the BDF Status channel.

Why it fits: identical (near-threshold) stimuli with a seen/unseen report on every trial,
catch trials for false alarms, a photodiode for exact onsets, 128 channels for topographies and
source analysis, a format eegunify already reads (BDF, BioSemi Status masking).

### Eklund & Wiens 2018 — second dataset candidate (verified 2026-11-05)

- **Data:** figshare doi:10.17045/sthlmuni.5418967, CC BY 4.0. Raw BioSemi `.bdf` in 4 zips
  (~9.8 GB), logs `txt/VAN_Exp_<fp>_<session>.txt`, FieldTrip scripts, final ERPs.
- **Paper:** Eklund R, Wiens S (2018). Cogn Affect Behav Neurosci 18:176-188,
  doi:10.3758/s13415-018-0562-z (PMC5823961). Preregistered (osf.io/4a4ur).
- **Design (paper, Procedure):** 35 subjects, two sessions (large / small Gabor), 580 trials per
  session (400 critical, 100 catch, 80 control), Gabor in one of four quadrants, **duration**
  calibrated by staircase to ~50 % seen; location response (4 keys), then awareness rating
  1 "did not see", 2 "saw weakly", 3 "saw clearly".
- **Codes (`fieldtrip/trialfun_VAN.m`):** 1-4 critical UL/UR/LL/LR, 5-8 control, 9 catch,
  10 block; TTL bits on Status; photodiode `Light1` gives onsets. 64 channels (`biosemi64.lay`).
- **To check before use:** which ratings the authors count as aware (TODO_FROM_SOURCE); the
  Status decoding on a real file; the montage mapping to ds006171's 128-channel ABC cap.
- Differs from ds006171: quadrants not hemifields, duration not contrast, 3-point rating, 64 ch.

### ds005273 — possible second dataset, blocked on its paper

- **Title:** Neural representation of consciously seen and unseen information. **Authors:**
  Pablo Rodríguez-San Esteban, Ana B. Chica, José A. González-López. **License:** CC0.
  **DOI:** 10.18112/openneuro.ds005273.v1.0.0. 33 subjects, 63 channels, 1000 Hz, BrainVision,
  about 47.7 GB (verified from the dataset metadata and file listing).
- **Blocker:** no README and no code book. `events.tsv` holds BrainVision markers
  (`Stimulus/S 70`, `S 77`, `S  7`, ...) without meanings, so seen/unseen trials cannot be
  identified without the associated paper, which was not found. **Use as a second dataset for
  Phase 2's cross-dataset test only once the paper (or the authors) gives the code meanings.**

### Other leads (not used)

- University of Queensland detection/discrimination EEG (9 participants): too small; not checked.
  https://researchdata.edu.au/neural-oscillations-eeg-detection-discrimination/4226327
- Seedat et al. open multi-center intracranial EEG (conscious perception): intracranial and
  suprathreshold stimuli, not a near-threshold scalp-EEG design.

## Phase 2 option: near-threshold touch — not found

- ds005291 ("65 By ANT", 65 subjects, 32 ch) and ds005293 ("95 By BP", 95 subjects): verified
  READMEs show **laser pain ratings** (0 = no sensation .. 10 = unbearable), suprathreshold
  intensities set from each person's pain threshold. Graded nociceptive sensation with per-trial
  ratings, **not** near-threshold tactile detection. Could serve a later "sensation intensity"
  analysis, not the seen/unseen-style contrast. Authors: Zhao Xiangyue, Zhou Jingyao, Zhang Libo,
  Duan Haoqing, Wei Shiyu, Bi Yanzhi, Hu Li; CC0.
- Förster et al. (tactile vs electrical peri-threshold detection ERPs) and Craddock et al. 2017
  (somatic signal detection task): papers found, **no public data located**.

## Phase 3: TMS-evoked phosphenes with EEG — not found

No open TMS-EEG dataset with phosphene reports was found. Papers exist (occipital/parietal
phosphene TMS-EEG, e.g. Bagattini et al. 2015; a phosphene-threshold TMS-EEG alpha study on
bioRxiv; a Samaha, Gosseries & Postle poster with 5 participants), but none points to public data.
Next places to look: data-availability statements of those papers, OSF, author contact.

### TMS-EEG for developing pulse-artifact handling (found, not about sensations)

- Scientific Data (2022): 24 healthy participants, intermittent / continuous theta-burst and sham
  stimulation of the left DLPFC (single-blind crossover), resting EEG and single-pulse TMS-EEG
  before and 2, 15, 30 min after, 64-channel EEG, ECG and MRI/DTI. Repository not confirmed
  (**unverified**). https://pubmed.ncbi.nlm.nih.gov/36414684/ ,
  https://link.springer.com/article/10.1038/s41597-022-01820-6
- Zenodo 7891964: a BIDS TMS-EEG dataset with right-DLPFC and left motor-cortex stimulation
  (memory-skill study) (**unverified** beyond the search snippet). https://zenodo.org/record/7891964

## Sources

- [ds006171 on OpenNeuro](https://openneuro.org/datasets/ds006171) (files read from the S3 mirror)
- [Melcón et al. 2024, Psychophysiology](https://doi.org/10.1111/psyp.14525)
- [ds005273 on OpenNeuro](https://openneuro.org/datasets/ds005273) ·
  [EEGDash mirror README](https://huggingface.co/datasets/EEGDash/ds005273/blob/main/README.md)
- [ds005291](https://openneuro.org/datasets/ds005291) · [ds005293](https://openneuro.org/datasets/ds005293)
- [Förster et al. preprint (bioRxiv)](https://www.biorxiv.org/node/4456083)
- [Craddock et al. 2017 (neuinfo)](https://neuinfo.org/literature/pmid:28041948?rpKey=on)
- [Phosphene-threshold TMS-EEG alpha study (bioRxiv)](https://www.biorxiv.org/content/10.1101/082693.full.pdf)
- [Bagattini et al. 2015 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5354329)
- [Samaha, Gosseries & Postle poster](https://old.postlab.psych.wisc.edu/index.php/download_file/view/347/127)
- [TBS-DLPFC TMS-EEG dataset, Sci Data 2022](https://link.springer.com/article/10.1038/s41597-022-01820-6)
- [Zenodo 7891964](https://zenodo.org/record/7891964)
- [UQ detection/discrimination EEG](https://researchdata.edu.au/neural-oscillations-eeg-detection-discrimination/4226327)
