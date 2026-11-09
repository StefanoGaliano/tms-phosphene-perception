# Q03 'unseen but correct': guessing-mixture test

Pre-registered run (§4 of docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)); 2026-11-09; code commit `19521eb`. Holm p (family of seven) for the [P] test: 0.0007 (`../prereg_family/`).

Inputs: results/eklund2018/trial_scores.tsv.gz; loso_predictions.tsv + staged events

## Group tests

| test | role | n | mean | ci_low | ci_high | d_z | p | tail | n_positive |
|---|---|---|---|---|---|---|---|---|---|
| large: D_obs - D_pred (two-sided) | [P] | 24 | -0.05105 | -0.06633 | -0.03584 | -1.31 | 0.0001 | 2 | 2 |
| large: TOST ±0.03 on D_obs - D_pred | secondary | nan | nan | nan | nan | nan | 0.9928 | nan | nan |
| large: D_obs | secondary | 24 | 0.04098 | 0.0257 | 0.05659 | 1.054 | 0.0001 | 2 | 21 |
| large: mean D_pred | secondary | nan | 0.09203 | nan | nan | nan | nan | nan | nan |
| small: D_obs - D_pred (two-sided) | replication | 14 | -0.06236 | -0.08434 | -0.04196 | -1.494 | 0.0003 | 2 | 0 |
| small: TOST ±0.03 on D_obs - D_pred | secondary | nan | nan | nan | nan | nan | 0.9938 | nan | nan |
| small: D_obs | secondary | 14 | 0.03674 | 0.01368 | 0.06184 | 0.784 | 0.0094 | 2 | 10 |
| small: mean D_pred | secondary | nan | 0.0991 | nan | nan | nan | nan | nan | nan |
| ds006171: a vs 0.5 | check | 34 | 0.5177 | 0.4935 | 0.5422 | 0.2424 | 0.1679 | 2 | 20 |
| ds006171: D_obs (LOSO score) | check | 34 | 0.09433 | -0.01022 | 0.2032 | 0.2879 | 0.1126 | 2 | 22 |

## Interpretation (rule fixed in the prereg)

below the prediction: informed unseen trials carry less of the signature than weak seen ones (evidence for processing without awareness; P§4 rule).

## Stated limits

- Mixture assumptions: informed trials always correct; guesses uniform.
- Rating-2 correct trials as the reference class for weak percepts (PROJECT CHOICE).
- ds006171: no rating 2, so a, f and D_obs only (answer 5).

## Deviations

none

Figure: `figure.png`.
