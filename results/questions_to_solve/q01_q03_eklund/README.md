# Q01-Q03 on Eklund & Wiens 2018 (2026-11-09)

Data: `results/eklund2018/trial_scores.tsv.gz` = P(seen) of the frozen detector
`sv-perception-tms-robust-v1.0.0` on every trial (no retraining). Critical near-threshold trials
only (condition 1-4), duration exclusion applied; high-contrast controls and catch trials left out
(`obj_corr` 3 = catch). Rating: 1 "did not see", 2 "saw something", 3 "saw clearly"
(Subj_resp). Unit: session; large Gabor primary (N = 24), small replication (N = 14). Two-sided
sign-flip permutation (10 000), bootstrap 95 % CI, d_z; minimum 3 trials per cell. Files:
`per_session.tsv`, `group.tsv`.

## Q01 graded awareness (correct-localization trials)

| | large | small |
|---|---|---|
| mean P(seen) rating 1 / 2 / 3 | 0.511 / 0.618 / 0.671 (N 24 / 24 / 11) | 0.507 / 0.616 / 0.609 (N 14 / 14 / 4) |
| Spearman rho(rating, P(seen)) | 0.250 [0.215, 0.286], d_z 2.72, p 1e-4, 24/24 > 0 | 0.259 [0.210, 0.306], d_z 2.74, p 1e-4, 14/14 |
| step 2 - 1 | +0.107 [0.091, 0.125], p 1e-4, 24/24 | +0.109 [0.088, 0.130], p 2e-4, 14/14 |
| step 3 - 2 | +0.059 [0.012, 0.105], p 0.048, N 11, 8/11 | +0.058 [0.015, 0.102], p 0.12, N 4 |

Graded on large Gabors: each rating step raises P(seen); the 2 -> 3 step is about half the
1 -> 2 step and rests on 11 sessions with >= 3 "clear" trials. Small Gabor: too few rating-3 trials.

## Q02 detection vs identification (rating >= 2)

| | large (N 15) | small (N 9) |
|---|---|---|
| correct minus wrong localization | +0.140 [0.101, 0.188], d_z 1.56, p 4e-4, 14/15 | +0.135 [0.074, 0.198], d_z 1.34, p 0.009, 8/9 |
| mean P(seen), "aware" with wrong location | 0.467 | 0.476 |
| wrong-location aware minus unaware-correct | -0.031 [-0.067, 0.004], p 0.12 | -0.022 [-0.070, 0.022], p 0.39 |

"Saw something" reports with the wrong location score no higher than "did not see" trials: the
detector reads stimulus evidence that also supports correct localization, not the report alone.
Consistent with Koivisto et al. 2017 (VAN at detection when detection goes with identification);
note that a wrong-location "aware" trial may also be a false report.

## Q03 unconscious processing (rating 1)

| | large (N 24) | small (N 14) |
|---|---|---|
| correct minus wrong localization | +0.041 [0.026, 0.056], d_z 1.05, p 1e-4, 21/24 | +0.037 [0.014, 0.060], d_z 0.78, p 0.010, 10/14 |
| awareness effect (aware correct - unaware correct) | +0.110 [0.093, 0.127] | +0.111 [0.090, 0.132] |
| ratio Q03 / awareness effect | 0.37 | 0.33 |

On trials reported as unseen, correct localization goes with a higher P(seen): about a third of
the awareness effect. Either residual stimulus processing without awareness (Bola & Doradzińska
2021: PAN "might not be sufficient"), or a conservative criterion (some weak percepts reported as
"did not see"). The data cannot separate the two. Consequence: the detector is not a pure
awareness marker; part of its score follows stimulus strength.
