# Q38 phosphene VAN topography and a wider response-hand control

Pre-registered run (§9 of docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)); 2026-11-16; code commit `de5260e`. Holm p (family of seven) for the [P] test: 0.002 (`../prereg_family/`).

Inputs: derivatives/sv_p3_clean/primary (included recordings); results/phase3/scores_primary.tsv

## Tests

| test | role | n | mean | ci_low | ci_high | d_z | p | tail | n_positive |
|---|---|---|---|---|---|---|---|---|---|
| H1 with P(seen) residualized on the hand index > 0.5 | [P] | 18 | 0.5353 | 0.5184 | 0.5527 | 0.9115 | 0.0005 | 1 | 15 |
| (a) H1 with all 54 lateral channels interpolated | secondary | 18 | 0.5229 | 0.5094 | 0.5364 | nan | 0.0034 | 1 | nan |
| (b) hand index alone, AUC vs 0.5 | secondary | 18 | 0.473 | 0.4496 | 0.4951 | -0.5399 | 0.0356 | 2 | 5 |
| (c) O1: residualized AUC vs 0.5 | secondary | 17 | 0.5262 | 0.5075 | 0.5466 | 0.6098 | 0.0095 | 1 | 14 |
| O1: van_oz_180_220 (µV, phosphene - none) | descriptive (seen) | 17 | -0.5154 | -0.9379 | -0.1343 | -0.5909 | 0.0205 | 2 | 5 |
| O1: van_s6_roi_180_280 (µV, phosphene - none) | descriptive (seen) | 17 | -0.6094 | -1.028 | -0.2389 | -0.7104 | 0.0044 | 2 | 5 |
| O1: lp_pz_300_800 (µV, phosphene - none) | descriptive (seen) | 17 | 0.2905 | -0.08069 | 0.6302 | 0.3765 | 0.1371 | 2 | 14 |
| (c) O2: residualized AUC vs 0.5 | secondary | 17 | 0.5478 | 0.5187 | 0.5769 | 0.7528 | 0.0038 | 1 | 12 |
| O2: van_oz_180_220 (µV, phosphene - none) | descriptive (seen) | 17 | -0.08794 | -0.4788 | 0.318 | -0.1006 | 0.6808 | 2 | 6 |
| O2: van_s6_roi_180_280 (µV, phosphene - none) | descriptive (seen) | 17 | 0.01737 | -0.3012 | 0.3469 | 0.02479 | 0.9174 | 2 | 9 |
| O2: lp_pz_300_800 (µV, phosphene - none) | descriptive (seen) | 17 | 0.5624 | 0.1866 | 0.9287 | 0.7023 | 0.0094 | 2 | 13 |

## Interpretation (rule fixed in the prereg)

H1 survives the hand covariate: the effect is not carried by the lateralized hand signal on any channel pair (P§9 rule).

## Stated limits

- Hand fixed to the label by design: bounded, not removed.
- Residualization is linear (as A1): a strongly nonlinear dependence of P(seen) on the hand index would not be fully removed.
- A null after (a) is ambiguous: interpolating all lateral channels removes posterior signal too.
- VAN rows are descriptive: data seen in the exploration of 2026-11-09.

## Deviations

none

Figure: `figure.png`.
