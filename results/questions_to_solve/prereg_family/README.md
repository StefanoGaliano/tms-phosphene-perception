# Questions_to_solve: family of primary tests (Holm)

Pre-registered run (§1 of docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)); 2026-11-16; code commit `de5260e`. Holm p for the [P] test: `../prereg_family/`.

Inputs: prereg_q01/primary.json; prereg_q02/primary.json; prereg_q03/primary.json; prereg_q04/primary.json; prereg_q37a/primary.json; prereg_q37b/primary.json; prereg_q38/primary.json

## Family

| question | test | tail | n | mean | ci_low | ci_high | p | p_holm | reject |
|---|---|---|---|---|---|---|---|---|---|
| q01 | large Gabors: rating slope with centred duration > 0 | 1 | 24 | 0.1028 | 0.08647 | 0.119 | 0.0001 | 0.0007 | True |
| q02 | Eklund large: correctness slope > 0 | 1 | 15 | 0.1398 | 0.09987 | 0.1883 | 0.0003 | 0.0015 | True |
| q03 | Eklund large: D_obs - D_pred | 2 | 24 | -0.05105 | -0.06633 | -0.03584 | 0.0001 | 0.0007 | True |
| q04 | informative - nocue matched AUC: TOST equivalence ±0.03 | equivalence | 27 | -0.01375 | nan | nan | 0.08575 | 0.08575 | False |
| q37a | H1 residualized on TEP amplitude | one-sided | 18 | 0.5224 | 0.5056 | 0.5392 | 0.0109 | 0.0218 | True |
| q37b | O2 block AUC at k = 20 | one-sided | 17 | 0.6866 | 0.5812 | 0.7836 | 0.0025 | 0.0075 | True |
| q38 | H1 residualized on the hand index | one-sided | 18 | 0.5353 | 0.5184 | 0.5527 | 0.0005 | 0.002 | True |

## Interpretation (rule fixed in the prereg)

6 of 7 primary tests rejected at Holm alpha 0.05. Q04's p is an equivalence p: rejection = equivalence shown.

## Stated limits

- Specified after the first look (P§0); run once.

## Deviations

none

Figure: `figure.png`.
