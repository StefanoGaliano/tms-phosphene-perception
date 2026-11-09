# Q04 attention, contrast-matched (ds006171 detector)

Pre-registered run (§5 of docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)); 2026-11-09; code commit `19521eb`. Holm p (family of seven) for the [P] test: 0.08575 (`../prereg_family/`).

Inputs: loso_predictions.tsv + staged events (34 subjects)

## Group tests

| test | role | n | mean | p | ci_low | ci_high | d_z | tail | n_positive |
|---|---|---|---|---|---|---|---|---|---|
| informative - nocue: TOST ±0.03 (equivalence p) | [P] | 27 | -0.01375 | 0.08575 | nan | nan | nan | nan | nan |
| noninformative - nocue: TOST ±0.03 | secondary | 27 | 0.01614 | 0.1165 | nan | nan | nan | nan | nan |
| informative - nocue (difference test) | secondary | 27 | -0.01375 | 0.2436 | -0.03629 | 0.008743 | -0.2291 | 2 | 11 |
| noninformative - nocue (difference test) | secondary | 27 | 0.01614 | 0.1643 | -0.00527 | 0.03841 | 0.2738 | 2 | 14 |
| informative - nocue: mean contrast | secondary | 27 | -0.004278 | 0.0385 | -0.008069 | -0.0003189 | -0.4112 | 2 | 9 |
| noninformative - nocue: mean contrast | secondary | 27 | 0.0001968 | 0.8758 | -0.002129 | 0.002273 | 0.03311 | 2 | 17 |
| matched AUC nocue vs 0.5 | descriptive | 30 | 0.7263 | 0.0001 | 0.7023 | 0.7496 | 3.304 | 2 | 30 |
| matched AUC noninformative vs 0.5 | descriptive | 30 | 0.7425 | 0.0001 | 0.7175 | 0.7681 | 3.374 | 2 | 30 |
| matched AUC informative vs 0.5 | descriptive | 31 | 0.7122 | 0.0001 | 0.6938 | 0.7307 | 3.968 | 2 | 31 |

## Interpretation (rule fixed in the prereg)

equivalence within ±0.03 not shown for the informative cue.

## Stated limits

- Deferred: the VAN part (needs the ds006171 epochs, runs/).
- Matching bins contrast within hemifield (Phase 1 rule); noninformative trials form one 'side unknown' group (answer 6).

## Deviations

none

Figure: `figure.png`.
