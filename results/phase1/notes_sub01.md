# Phase 1 Step 6: eegunify on one subject x nocue (2026-11-05)

Run: `eegunify convert` + `eegunify erp run` with `configs/generated/ds006171_nocue.yaml`,
`--root runs/nocue`. sub-01 was the planned subject. Its HEOG1 electrode is broken (below), so
the four checks were completed on **sub-03** (all three tasks, good EOG).

## The four checks

| Check | Result |
|---|---|
| (a) rawdata events equal the staged onsets | **Pass** (sub-01): 848 events, identical trial types, onsets within 0.5 ms (one sample at 1024 Hz is 0.98 ms). The photodiode correction survives `bids_events`. |
| (b) tag selectors give the expected counts | **Pass** (sub-03): seen 132 = seen_left 66 + seen_right 66; unseen 117 = 58 + 59; seen_m 102 <= seen; matches the kept epochs in the epoch log. |
| (c) tiny `fa` condition does not crash | **Pass** (sub-03): fa = 4 epochs, cr = 28; `fa_minus_cr` is computed; erp and group stages ok. |
| (d) extra events.tsv columns carried to rawdata | **Dropped**: rawdata keeps `onset, duration, trial_type, value, sample`. The staged tree keeps the full copy (`sourcedata/staged/`), so Start_Vision reads behaviour from there. |

## Fixes needed to get here (config side; eegunify unchanged)

1. **EOG channels were dropped.** eegunify drops a channel when its new *or original* name is in
   `channels.drop`. `drop` listed `EXG1-4` (the renamed spare externals EXG5-8), but `EXG1-4` are
   also the BDF names of VEOG1/VEOG2/HEOG1/HEOG2. The template now renames EXG5-8 to
   `unused_EXG5-8` and drops those (regression test `test_ds006171_eog_not_dropped_by_original_name`).
2. **Reject rule `channels: [eeg]` is not a keyword**: eegunify needs names. `sv-configs` now
   expands it to the 128 EEG names (`test_ptp_rule_lists_128_eeg`).
3. **`lateralized_pairs` only on a lateralized contrast**: removed from `van@awareness` (on the
   plain seen_minus_unseen_m contrast). The primary contralateral VAN is computed in
   Start_Vision from the `seen_lat`/`unseen_lat` waves anyway (decision 4).

## Open: EOG electrodes broken in many subjects (decision needed)

sub-01 nocue: 359/400 epochs rejected, 325 of them by the HEOG step rule (32 uV, version A of
decision 5): median HEOG step 87 uV. The HEOG1 electrode is broken (SD 1893 uV vs 248 uV for
HEOG2, 0.5-15 Hz, re Oz, 300 s of the raw BDF), not saccades: the VEOG is normal and the HEOG
steps are uncorrelated with it (r = 0.01).

Same check on every subject's nocue BDF (SD in uV, 0.5-15 Hz, 100-400 s):

- HEOG1 broken: sub-01 (1893), sub-11 (19059); high: sub-08, 13, 29 (180-280).
- VEOG1 broken: sub-04 (339), sub-19 (13570), sub-24 (862), sub-31 (4333); sub-30 VEOG1-VEOG2
  flat (SD 0).
- Typical good subject: VEOG1-VEOG2 20-80, HEOG1-HEOG2 60-190.

sub-03 (good EOG), same config: 281/398 kept; HEOG rule rejected 1 trial, the 100 uV ptp rule
over the whole -1000..1000 ms epoch rejected 115 (29 %).

Also note: the 100 uV ptp threshold and its whole-epoch window come from the blueprint, not
from a cited source (Eklund used individual amplitude-range thresholds on -100..600 ms epochs).

## Resolved (user decisions 7 and 8, 2026-11-05) and rerun

- HEOG for the step rule = F7-F8 for every subject (decision 7); VEOG stays VEOG1-VEOG2 (ICA only).
- 100 uV peak-to-peak rule tested on -100..600 ms only (decision 8).

| Subject (nocue) | Kept | Rejected by HEOG step (F7-F8, 32 uV) | Rejected by ptp |
|---|---|---|---|
| sub-01 | 339 / 400 | 50 | 11 |
| sub-03 | 308 / 398 | 15 | 74 |

`eegunify erp status`: preproc, ica, epochs, erp ok for sub-01 and sub-03; group ok. QA: no FAIL.
Warnings: `ERP_NOISE_SME` on both (SME of the primary mean amplitude, sub-03: LP 2.20 uV, VAN
0.92 uV; single-subject noise, reviewed in Step 7); `PREPROC_LINE_NOISE_RESIDUAL` on sub-03
(50 Hz peak 29.9 dB above neighbours: eegunify's default line-noise method is `none` and
lowpass is null, per PHASE1_REFERENCES §5; mean-amplitude windows of 100-200 ms average 50 Hz
out, so it is left as is and watched in Step 7).
