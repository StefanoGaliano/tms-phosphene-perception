# Q01 graded awareness (Eklund & Wiens 2018)

Pre-registered run (§2 of docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)); 2026-11-09; code commit `19521eb`. Holm p (family of seven) for the [P] test: 0.0007 (`../prereg_family/`).

Inputs: results/eklund2018/trial_scores.tsv.gz (14097 critical trials)

## Group tests

| test | role | n | mean | ci_low | ci_high | d_z | p | tail | n_positive |
|---|---|---|---|---|---|---|---|---|---|
| large: primary | [P] | 24 | 0.1028 | 0.08647 | 0.119 | 2.459 | 0.0001 | 1 | 24 |
| large: no_covariate | secondary | 24 | 0.1027 | 0.0866 | 0.1189 | 2.47 | 0.0001 | 1 | 24 |
| large: step_2_minus_1 | secondary | 24 | 0.1072 | 0.09054 | 0.1243 | 2.478 | 0.0001 | 2 | 24 |
| large: step_3_minus_2 | secondary | 11 | 0.05889 | 0.01043 | 0.1059 | 0.6937 | 0.04492 | 2 | 8 |
| small: primary | replication | 14 | 0.1059 | 0.08438 | 0.1279 | 2.475 | 0.0002 | 1 | 14 |
| small: no_covariate | secondary | 14 | 0.1057 | 0.08419 | 0.1277 | 2.476 | 0.0002 | 1 | 14 |
| small: step_2_minus_1 | secondary | 14 | 0.1088 | 0.08796 | 0.1301 | 2.607 | 0.0003 | 2 | 14 |
| small: step_3_minus_2 | secondary | 4 | 0.05842 | 0.01446 | 0.1024 | 1.137 | 0.125 | 2 | 4 |

## Interpretation (rule fixed in the prereg)

graded read-out: the rating slope is positive with the duration covariate (P§2 rule).

## Stated limits

- Rating 3 is sparse (sessions without >= 3 'saw clearly' trials have no step 3-2).
- Same data as the first look (P§0): specified before the rerun, not confirmatory.

## Deviations

none

Figure: `figure.png`.
