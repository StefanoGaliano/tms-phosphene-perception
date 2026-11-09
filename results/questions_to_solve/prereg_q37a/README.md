# Q37a trial-level TMS-evoked-response amplitude (GIN)

Pre-registered run (§6 of docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)); 2026-11-09; code commit `19521eb`. Holm p (family of seven) for the [P] test: 0.01 (`../prereg_family/`).

Inputs: derivatives/sv_p3_clean/primary (33 included recordings); results/phase3/scores_primary.tsv

## Group tests

| test | role | n | mean | ci_low | ci_high | d_z | p | tail | n_positive |
|---|---|---|---|---|---|---|---|---|---|
| H1 with P(seen) residualized on TEP amplitude > 0.5 | [P] | 17 | 0.5259 | 0.5096 | 0.5426 | 0.7216 | 0.005 | 1 | 12 |
| H1 frozen (reproduces Phase 3) | check | 17 | 0.5398 | nan | nan | nan | 0.0008 | 1 | nan |
| TEP amplitude alone, AUC vs 0.5 | secondary | 17 | 0.5413 | 0.5185 | 0.5622 | 0.8829 | 0.0035 | 2 | 16 |
| SSP-3 (first look, copied as context) | context | 17 | 0.5254 | nan | nan | nan | 0.0184 | 1 | nan |

## Interpretation (rule fixed in the prereg)

H1 survives: the effect is not the trial-to-trial size of the pulse response. TEP amplitude alone separates the classes: the phosphene modulates the evoked response (Taylor 2010-like), so residualizing removes part of a real effect. (P§6 rule)

## Stated limits

- Same recordings as H1 (P§0 first look: mean subtraction, SSP-3).

## Deviations

none

Figure: `figure.png`.
