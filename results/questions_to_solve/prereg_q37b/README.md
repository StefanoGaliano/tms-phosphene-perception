# Q37b pulses needed for a block read-out (GIN)

Pre-registered run (§7 of docs/QUESTIONS_PREREG.md (tags questions-prereg, -amend1, -amend2)); 2026-11-09; code commit `19521eb`. Holm p (family of seven) for the [P] test: 0.0087 (`../prereg_family/`).

Inputs: results/phase3/scores_primary.tsv (included recordings)

## Tests

| test | role | n | mean | ci_low | ci_high | d_z | p | tail | n_positive |
|---|---|---|---|---|---|---|---|---|---|
| O2 block AUC at k = 20 > 0.5 (k* not reached on O1) | [P] | 17 | 0.6857 | 0.5802 | 0.7839 | 0.8303 | 0.0029 | 1 | 12 |
| observed - predicted, k = 1 (pooled sites) | secondary | 17 | -0.002056 | -0.005835 | 0.001593 | -0.2546 | 0.3031 | 2 | 7 |
| O1 mean observed / predicted AUC, k = 1 | descriptive | nan | 0.5262 | 0.5286 | nan | nan | nan | nan | nan |
| O2 mean observed / predicted AUC, k = 1 | descriptive | nan | 0.5504 | 0.5523 | nan | nan | nan | nan | nan |
| observed - predicted, k = 5 (pooled sites) | secondary | 17 | -0.003723 | -0.01307 | 0.004567 | -0.1938 | 0.4682 | 2 | 5 |
| O1 mean observed / predicted AUC, k = 5 | descriptive | nan | 0.558 | 0.5621 | nan | nan | nan | nan | nan |
| O2 mean observed / predicted AUC, k = 5 | descriptive | nan | 0.6052 | 0.6101 | nan | nan | nan | nan | nan |
| observed - predicted, k = 10 (pooled sites) | secondary | 17 | -0.0003917 | -0.01488 | 0.01105 | -0.01365 | 0.9672 | 2 | 11 |
| O1 mean observed / predicted AUC, k = 10 | descriptive | nan | 0.5832 | 0.5846 | nan | nan | nan | nan | nan |
| O2 mean observed / predicted AUC, k = 10 | descriptive | nan | 0.6442 | 0.6453 | nan | nan | nan | nan | nan |
| observed - predicted, k = 20 (pooled sites) | secondary | 17 | 0.003276 | -0.01319 | 0.01634 | 0.1015 | 0.7186 | 2 | 11 |
| O1 mean observed / predicted AUC, k = 20 | descriptive | nan | 0.6136 | 0.612 | nan | nan | nan | nan | nan |
| O2 mean observed / predicted AUC, k = 20 | descriptive | nan | 0.6857 | 0.6825 | nan | nan | nan | nan | nan |

## Interpretation (rule fixed in the prereg)

k* on O1: not reached; O2 block AUC at k = 20: 0.686 (planning estimate, P§7).

## Stated limits

- Subjects with single-pulse AUC < 0.5 are reported, not excluded.
- O1 and O2 are mostly the same subjects: O2 holds out recordings, not people.
- One generator per recording (answer 10): first-look numbers not reproduced exactly.
- Blocks are random draws within class, so dependence between neighbouring pulses cannot enter them: observed - predicted tests the equal-variance Gaussian shape of the single-pulse scores, not independence (P§8 note).

## Deviations

none

Figure: `figure.png`.
