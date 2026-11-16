# Q37b pulses needed for a block read-out (GIN)

Pre-registered run (§7 of docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)); 2026-11-16; code commit `de5260e`. Holm p (family of seven) for the [P] test: 0.0075 (`../prereg_family/`).

Inputs: results/phase3/scores_primary.tsv (included recordings)

## Tests

| test | role | n | mean | ci_low | ci_high | d_z | p | tail | n_positive |
|---|---|---|---|---|---|---|---|---|---|
| O2 block AUC at k = 20 > 0.5 (k* not reached on O1) | [P] | 17 | 0.6866 | 0.5812 | 0.7836 | 0.8403 | 0.0025 | 1 | 12 |
| observed - predicted, k = 1 (pooled sites) | secondary | 18 | -0.003215 | -0.006814 | 0.0005982 | -0.3891 | 0.118 | 2 | 6 |
| O1 mean observed / predicted AUC, k = 1 | descriptive | nan | 0.5227 | 0.5276 | nan | nan | nan | nan | nan |
| O2 mean observed / predicted AUC, k = 1 | descriptive | nan | 0.5506 | 0.5523 | nan | nan | nan | nan | nan |
| observed - predicted, k = 5 (pooled sites) | secondary | 18 | -0.0008445 | -0.01034 | 0.008264 | -0.04055 | 0.8721 | 2 | 7 |
| O1 mean observed / predicted AUC, k = 5 | descriptive | nan | 0.5582 | 0.5599 | nan | nan | nan | nan | nan |
| O2 mean observed / predicted AUC, k = 5 | descriptive | nan | 0.606 | 0.6101 | nan | nan | nan | nan | nan |
| observed - predicted, k = 10 (pooled sites) | secondary | 18 | 0.002727 | -0.01137 | 0.01382 | 0.09554 | 0.7408 | 2 | 13 |
| O1 mean observed / predicted AUC, k = 10 | descriptive | nan | 0.5834 | 0.5817 | nan | nan | nan | nan | nan |
| O2 mean observed / predicted AUC, k = 10 | descriptive | nan | 0.6453 | 0.6453 | nan | nan | nan | nan | nan |
| observed - predicted, k = 20 (pooled sites) | secondary | 18 | 0.007792 | -0.009147 | 0.0225 | 0.2197 | 0.3831 | 2 | 12 |
| O1 mean observed / predicted AUC, k = 20 | descriptive | nan | 0.6136 | 0.6082 | nan | nan | nan | nan | nan |
| O2 mean observed / predicted AUC, k = 20 | descriptive | nan | 0.6866 | 0.6825 | nan | nan | nan | nan | nan |

## Interpretation (rule fixed in the prereg)

k* on O1: not reached; O2 block AUC at k = 20: 0.687 (planning estimate, P§7).

## Stated limits

- Subjects with single-pulse AUC < 0.5 are reported, not excluded.
- O1 and O2 are mostly the same subjects: O2 holds out recordings, not people.
- One generator per recording (answer 10): first-look numbers not reproduced exactly.
- Blocks are random draws within class, so dependence between neighbouring pulses cannot enter them: observed - predicted tests the equal-variance Gaussian shape of the single-pulse scores, not independence (P§8 note).

## Deviations

none

Figure: `figure.png`.
