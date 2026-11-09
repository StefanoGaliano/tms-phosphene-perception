# Q02 seen with wrong content (Eklund; ds006171)

Pre-registered run (§3 of docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)); 2026-11-09; code commit `19521eb`. Holm p (family of seven) for the [P] test: 0.0015 (`../prereg_family/`).

Inputs: results/eklund2018/trial_scores.tsv.gz; loso_predictions.tsv + staged events

## Group tests

| test | role | n | mean | ci_low | ci_high | d_z | p | tail | n_positive |
|---|---|---|---|---|---|---|---|---|---|
| Eklund large: correctness slope (centred duration) > 0 | [P] | 15 | 0.1398 | 0.09987 | 0.1883 | 1.546 | 0.0003 | 1 | 14 |
| Eklund small: same | replication | 9 | 0.1315 | 0.07307 | 0.1925 | 1.334 | 0.003906 | 1 | 8 |
| ds006171: AUC correct vs wrong, contrast-residualized, vs 0.5 | secondary | 31 | 0.5875 | 0.5399 | 0.6345 | 0.6309 | 0.0017 | 2 | 25 |

## Interpretation (rule fixed in the prereg)

both datasets positive: the read-out needs content-bearing evidence (P§3 rule).

## Stated limits

- Wrong-content 'seen' trials can be false reports (P§3).
- Same data as the first look (P§0).

## Deviations

none

Figure: `figure.png`.
