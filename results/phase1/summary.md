# Phase 1 results: ds006171, all subjects (Step 10, 2026-11-05)

Pipeline: `sv-stage` -> `sv-configs` -> eegunify 0.9.3 (`runs/<task>/`) -> `sv-analyze`.
Tables in this folder (`group.tsv`, `per_subject.tsv`, `exclusions.tsv`, `clusters.tsv`,
`controls.tsv`, `eegunify_reproduction.tsv`); figures in `figures/`.
Exclusion rules fixed before the group results were seen (`docs/PHASE1_REFERENCES.md` §7).

## Run

- 104 recordings; eegunify completed all four stages (preproc, ica, epochs, erp) on 99.
- The other 5 failed at eegunify's epochs stage, each with more than 50 % of epochs rejected:
  sub-14 nocue 54 %, sub-19 nocue 56 %, sub-27 informative 63 %, sub-32 informative 52 %,
  sub-14 noninformative 66 %. Rule 2 (> 45 %) would exclude them anyway; listed under rule 3.
- Median matched epochs per recording: 119 seen, 121 unseen.

## Exclusions

| Rule | Excluded |
|---|---|
| 1: pooled hit rate 30-70 % and FA <= 20 % (whole subject) | sub-20 (FA 22 %), sub-21 (FA 26 %) |
| 2: > 45 % epochs rejected (recording) | sub-08 nocue 47 %, sub-15 noninformative 48 % |
| 3: < 30 matched epochs / epochs stage failed (recording) | the 5 recordings above |

N per scope: nocue 30, noninformative 30, informative 31, pooled 34.

## Group results (seen minus unseen, µV; d_z with 95 % bootstrap CI)

| Scope | Measure | N | Mean | p (t) | d_z [95 % CI] |
|---|---|---|---|---|---|
| pooled | **VAN contralateral (primary)** | 34 | -1.47 | < 1e-4 | -0.84 [-1.33, -0.50] |
| pooled | VAN bilateral, matched | 34 | -1.00 | < 1e-4 | -0.69 [-1.10, -0.39] |
| pooled | VAN double difference | 34 | -0.86 | < 1e-4 | -0.93 [-1.33, -0.64] |
| pooled | LP bilateral, matched | 34 | +2.63 | < 1e-4 | +2.32 [1.89, 3.03] |
| nocue | VAN contralateral | 30 | -1.12 | 0.003 | -0.59 [-0.99, -0.26] |
| nocue | VAN bilateral, matched | 30 | -0.69 | 0.038 | -0.40 [-0.78, -0.05] |
| nocue | VAN double difference | 30 | -0.74 | < 1e-3 | -0.80 [-1.22, -0.49] |
| nocue | LP | 30 | +2.68 | < 1e-4 | +1.96 [1.49, 2.80] |
| noninformative | VAN bilateral, matched (no lateral form: decision 6) | 30 | -1.08 | < 1e-3 | -0.78 [-1.23, -0.46] |
| noninformative | LP | 30 | +2.79 | < 1e-4 | +1.68 [1.18, 2.56] |
| informative | VAN contralateral | 31 | -1.50 | < 1e-4 | -0.93 [-1.61, -0.54] |
| informative | VAN bilateral, matched | 31 | -1.05 | < 1e-3 | -0.71 [-1.19, -0.36] |
| informative | VAN double difference | 31 | -0.84 | < 1e-4 | -0.82 [-1.21, -0.52] |
| informative | LP | 31 | +2.55 | < 1e-4 | +1.60 [1.27, 2.14] |

Wilcoxon tests agree in sign and significance in every row (`group.tsv`, columns W and p_w).
The VAN is present in every task and in every form. The bilateral VAN in nocue is the weakest
result: its CI comes close to 0.

## Cluster permutation tests (seen_minus_unseen_m, all channels, -100..800 ms)

In every scope, one positive and one negative cluster survive at p = 0.0002, the smallest
p-value the permutations can give. Each spans 92-111 channels from about 70-120 ms to the epoch
end (pooled: positive 102-801 ms, negative 74-801 ms). The topography of the clusters matches
the planned components:

- the VAN window is negative at posterior and occipital sites, with a left-posterior maximum;
- the LP window is positive at centro-parietal sites;
- both have frontal inversions, which is expected with an average reference.

Because the clusters are so broad, they confirm that seen and unseen trials differ, but they do
not give the onset of any component. Use the windowed measures for that.

## Comparison with Eklund & Wiens 2018 (PHASE1_REFERENCES §2, Table 2)

| Measure | Ours (pooled) | Eklund large | Eklund small |
|---|---|---|---|
| VAN | -1.47 µV (contralateral) | -1.58 [-2.37, -0.79] | -1.41 [-2.04, -0.77] |
| LP | +2.63 µV | +1.35 [0.96, 1.74] | +1.00 [0.50, 1.50] |

- **VAN: replicated.** Our mean lies inside both of Eklund's CIs.
- **LP: present, but about twice their size**, and above their CI. A plausible cause: in
  ds006171 every trial asks for a report, so the task and the decision load on the LP. Dembski et
  al. 2021 show the LP is report-sensitive. This explanation is a hypothesis and has not been tested.

## Controls

- **Contrast balance:** the median |SMD| of contrast between seen and unseen trials is 0.25 over
  all trials and 0.015 after matching (max 0.07, across 104 recordings).
- **Hit / FA rates by task (mean):** hit rate 0.47-0.49; FA rate 0.07-0.08. The staircase works.
- **Orientation check:** accuracy is 0.85-0.90 on seen trials and 0.51-0.52 on unseen trials, close
  to chance. Unseen reports therefore mean little orientation information.
- **FA vs CR wave: per task not possible, pooled exploratory and null.** Only 3 of 104
  recordings reach the 10 false alarms the wave needs. Pooled across each subject's tasks, 9
  included subjects keep >= 10 FA and >= 10 CR epochs (rule `MIN_FA` in `report.py`). FA minus
  CR: VAN window +0.66 µV (t = 1.07, p = 0.32, d_z +0.36 [-0.32, 1.26]); LP window -0.30 µV
  (p = 0.76). No awareness-like signature on "seen" reports without a stimulus, but N = 9 gives
  little power, so this is not evidence of absence (`group.tsv`, form `fa_minus_cr`; per-task
  rows with N = 1 carry no test).
- **eegunify reproduction:** our t and d_z equal eegunify's group stage to below 1e-6 in 6 of 6
  rows (N = 33, eegunify's own subject set).

## Figures

- `figures/waves_pooled.png`: seen and unseen grand averages, with the difference wave and its
  95 % CI, for the contralateral VAN, the bilateral VAN and the LP. The contralateral VAN peaks
  near 200 ms at about -2.5 µV. The LP is sustained from about 300 ms onwards.
- `figures/subjects_pooled.png`: per-subject effects. The double difference is negative in most
  subjects, and the LP is positive in all 34.
- `figures/topo_pooled.png`: topographies in the VAN and LP windows.
- The same three figures per task: `waves_<task>.png`, `subjects_<task>.png`, `topo_<task>.png`
  (noninformative has no lateral panel, decision 6).

## Rerun stability (Step 11)

A full `sv-analyze` rerun (2026-11-05) reproduces every one of the 440 per-subject rows exactly
(max |change| 0.0 µV).

## Open

- The LP size difference from Eklund is not explained. The report-on-every-trial hypothesis
  cannot be tested inside ds006171 (every trial has a report). Other differences that could
  contribute, untested: Eklund's aware/unaware classes required a correct localization and
  "aware" pooled weak and clear ratings, while ds006171 has a yes/no report; 128-channel versus
  64-channel + nose average reference; P3/P4 mapped to CPP3/CPP4 (`PHASE1_REFERENCES.md` §4).
  The LP is not a primary measure and has the expected direction.
- The FA vs CR control has low power (N = 9, above).
- Robustness check to run later: saccade-rejection version B (ICA only, no HEOG step rule;
  decision 5), skipped for the v1.0 freeze.
