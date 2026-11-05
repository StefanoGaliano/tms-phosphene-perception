# Phase 1 Step 9: analysis package on 5 subjects x 3 tasks (2026-11-05)

`sv-analyze` (start_vision.analysis: load, pool, lateral, stats, controls, report) on
sub-01..05, every task each has (13 recordings; sub-01 has no informative, sub-02 no
noninformative). Outputs: `per_subject.tsv`, `group.tsv`, `controls.tsv`,
`eegunify_reproduction.tsv`.

## Acceptance

- **stats reproduces eegunify to 1e-6:** 6/6 rows (VAN and LP, seen_minus_unseen_m, each task):
  `stats.paired` on eegunify's own per-subject measures gives eegunify's t and d_z with error 0.0.
- **Pooled results computed:** per subject, conditions pooled across tasks weighted by trial
  count, contrasts recomputed from the pooled conditions (`analysis/pool.py`). The lateral
  waves pool nocue + informative only (decision 6).
- **All controls produced** (`controls.tsv`): hit rate per task 0.36-0.53 (staircase near
  50 %); FA rate 0-0.32; orientation accuracy higher for seen (0.61-1.00) than unseen
  (0.43-0.70) in every recording.
- **Runtime per subject x task** (convert + preproc + ICA + epochs + erp, while another
  pipeline shared the CPU): 49-256 s, median 78 s. Full run, 104 recordings: about 2.5-4 h.

## Notes for Step 10

- With N = 4-5 nothing is significant and there are no d_z CIs (eegunify gives none below 8
  subjects); directions agree with the targets: VAN negative in every scope (pooled
  contralateral -2.40 µV), LP positive (pooled +1.87 µV, t = 3.90, p = 0.018).
- My window means differ slightly from eegunify's for the same evoked (sub-03 nocue LP 1.138
  vs 1.149 µV): the window edges fall on different samples of the 256 Hz grid. Per-task
  group numbers are reported from eegunify's measures; Start_Vision's own means are used for the
  pooled and lateral forms, which eegunify does not compute.
- sub-05 informative: FA rate 0.32 (13/41 catch trials). Decision 2's inclusion rule is a
  pooled FA rate <= 20 %; it is applied in Step 10 over all tasks.
- Figures (`analysis/figures.py`) are deferred to Step 10, where there is a group to plot.
