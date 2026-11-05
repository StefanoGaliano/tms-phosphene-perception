# Phase 2 Step 0: data contract, inclusion, descriptives (2026-11-05)

Command: `sv-step0` (`start_vision.decoding.step0`). Tables: `inclusion.tsv` (one row per
recording), `stats_descriptives.tsv` (T1). Input: Phase 1 `desc-stimraw` epochs + staged events.

## Contract (PHASE2 §4, as built)

- 99 of 104 recordings load; all 99 pass the contract: 128 EEG
  channels with identical names and order, 128 Hz after a 40 Hz low-pass, 192 samples from
  -500 ms, required metadata filled, finite data.
- The 5 others have no epochs file (eegunify's epochs stage failed in Phase 1, > 50 % epochs
  rejected): sub-14 nocue, sub-14 noninformative, sub-19 nocue, sub-27 informative, sub-32 informative.
- Known gaps by design: `resp_key_seen` / `resp_key_pressed` are empty (not recorded; decision 3);
  noninformative trials have hemifield `unknown` (decision 6).

## Inclusion

Rules: Phase 1 subject and recording rules (`results/phase1/exclusions.tsv`: user decision 2,
> 45 % rejected, < 30 matched epochs), then PHASE2 §6 (>= 30 seen and >= 30 unseen).

- Included: **91 recordings, 34 subjects** (informative 31, nocue 30, noninformative 30), the Phase 1 set.
- The §6 rule excluded no recording beyond the Phase 1 rules.
- Excluded: sub-08 nocue (2: 47% epochs rejected (> 45 %)); sub-14 nocue (3: < 30 matched epochs (eegunify epochs stage failed)); sub-14 noninformative (3: < 30 matched epochs (eegunify epochs stage failed)); sub-15 noninformative (2: 48% epochs rejected (> 45 %)); sub-19 nocue (3: < 30 matched epochs (eegunify epochs stage failed)); sub-20 informative (1: pooled hit rate 47%, FA rate 22%); sub-20 nocue (1: pooled hit rate 47%, FA rate 22%); sub-20 noninformative (1: pooled hit rate 47%, FA rate 22%); sub-21 informative (1: pooled hit rate 49%, FA rate 26%); sub-21 nocue (1: pooled hit rate 49%, FA rate 26%); sub-21 noninformative (1: pooled hit rate 49%, FA rate 26%); sub-27 informative (3: < 30 matched epochs (eegunify epochs stage failed)); sub-32 informative (3: < 30 matched epochs (eegunify epochs stage failed)).

## Descriptives of the included recordings (T1)

| | mean | min | max |
|---|---|---|---|
| seen trials | 148.297 | 93 | 188 |
| unseen trials | 160.121 | 101 | 223 |
| catch trials | 34.571 | 23 | 40 |
| hit rate | 0.482 | 0.348 | 0.559 |
| FA rate (per task) | 0.066 | 0 | 0.375 |
| share of stimulus trials in the seen/unseen contrast overlap | 0.979 | 0.922 | 0.994 |

Seen and unseen trials overlap almost completely in contrast (staircase), so within-bin
decoding (A4) has trials of both classes in every bin. The per-task FA rate can exceed 20 %
(max shown) because decision 2 applies to the pooled rate.
