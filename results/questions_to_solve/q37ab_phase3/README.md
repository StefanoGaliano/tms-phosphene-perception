# Q37a, Q37b on Phase 3 primary (2026-11-09, exploratory)

Script `scripts/questions_q37ab_phase3.py`. Same 33 included recordings / 17 subjects as H1.
Files: `q37a_group.tsv`, `q37a_per_recording.tsv`, `q37b_group.tsv`, `q37b_per_recording.tsv`.

## Q37a: remove the pulse response shared by both classes (label-free), frozen detector unchanged

| Variant | H1 AUC [95 % CI] | d_z | p (one-sided) | subjects > 0.5 |
|---|---|---|---|---|
| frozen (as H1; reproduces 0.5398) | 0.540 [0.521, 0.559] | 0.97 | 0.0009 | 14/17 |
| mean_sub: recording mean subtracted | 0.540 [0.521, 0.559] | 0.97 | 0.0009 | 14/17 |
| ssp3: top 3 spatial components of the mean projected out | 0.525 [0.504, 0.548] | 0.53 | 0.018 | 13/17 |

- **mean_sub changes nothing, by construction (correction 2026-11-09).** Subtracting one average
  from every trial shifts every trial's features by the same vector, so the linear detector's
  score shifts by the same constant for all trials and the ranking (AUC) cannot change. This is
  not evidence about the shared response; an earlier note here said the contract z-scores features
  over trials, which is wrong (the contract baselines each trial and scales each channel by its
  prestimulus statistics). What the shared response *does* change is the absolute P(seen) level:
  that is what Q37c must measure.
- **ssp3 lowers H1 to 0.525 but it stays significant.** Projecting out the TEP's main topographies
  also removes part of the class difference, so the phosphene signal partly shares topography
  with the pulse-evoked response (expected if the phosphene modulates the same cortical response).
  This cannot separate "phosphene modulates the TEP" from "TEP amplitude varies with something
  that also drives the report"; it bounds how much the result depends on TEP-shaped activity.

## Q37b: how many pulses for a reliable read-out (mean P(seen) over k pulses of one class)

| k pulses | mean AUC [95 % CI] | range over subjects | subjects with AUC >= 0.75 |
|---|---|---|---|
| 1 | 0.539 [0.520, 0.559] | 0.475-0.608 | 0/17 |
| 5 | 0.583 [0.548, 0.619] | 0.464-0.714 | 0/17 |
| 10 | 0.609 [0.560, 0.657] | 0.435-0.775 | 2/17 |
| 20 | 0.649 [0.583, 0.713] | 0.393-0.862 | 4/17 |

- Averaging helps slowly: about 0.65 at 20 pulses on average, 0.86 in the best subject. Subjects
  whose single-pulse AUC is below 0.5 go further below with averaging (minimum 0.393 at k = 20):
  for them the frozen detector is reliably wrong, not just noisy.
- For a stimulation study: v1 reads blocks of about 20 pulses usefully in about a quarter of
  subjects. A per-subject anchor (Q37i) or a better detector is needed before per-pulse use.
- Blocks are resampled from a finite set of trials (100-250 per class), so they are not
  independent; the numbers estimate the block read-out, not its exact sampling error.
