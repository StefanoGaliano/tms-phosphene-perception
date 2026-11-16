# Questions_to_solve: results

Results for the questions in `../../Question_to_answer/` (the list, sources and ratings). One
subfolder per batch, each with its script in `../../scripts/`. Exploratory: these questions were
not pre-registered; thresholds were fixed in the script before group statistics were computed.

| Batch | Questions | Data | Script | Date |
|---|---|---|---|---|
| `q01_q03_eklund/` | Q01 graded awareness, Q02 detection vs identification, Q03 unconscious processing | Eklund & Wiens 2018, frozen `tms-robust` scores | `scripts/questions_q01_q03_eklund.py` | 2026-11-09 |
| `saved_results/` | Q02 + Q03 on ds006171, Q04 attention, Q17 size, Q27 below-chance | ds006171 LOSO scores + Phase 1 tables; Eklund session AUCs; Phase 2 A2 | `scripts/questions_saved_results.py` | 2026-11-09 |
| `q03_where_eklund/` | Q03 step 1 (where the signal lives) | needs Eklund epochs: **not run**, `runs/` removed on purpose | `scripts/questions_q03_where_eklund.py` | waiting |
| `q37ab_phase3/` | Q37a shared pulse response, Q37b pulses needed | Phase 3 primary cleaned epochs and scores | `scripts/questions_q37ab_phase3.py` | 2026-11-09 |

## Pre-registered run (2026-11-09; Q37a/b, Q38 re-run 2026-11-16 with sub-18 O1; `sv-questions all`, code `19521eb`, tags questions-prereg / -amend1 / -amend2)

| Folder | Question | [P] result | Holm p (family of 7) |
|---|---|---|---|
| `prereg_q01/` | Q01 graded awareness | rating slope +0.103 [0.086, 0.119], 24/24 | 0.0007 |
| `prereg_q02/` | Q02 seen with wrong content | correctness slope +0.140 [0.100, 0.188], 14/15 | 0.0015 |
| `prereg_q03/` | Q03 guessing mixture | D_obs - D_pred -0.051 [-0.066, -0.036] (below prediction) | 0.0007 |
| `prereg_q04/` | Q04 attention, matched | informative - nocue -0.014; equivalence ±0.03 not shown (p 0.086) | 0.086 |
| `prereg_q37a/` | Q37a TEP amplitude | H1 residualized 0.522 [0.506, 0.539] | 0.022 |
| `prereg_q37b/` | Q37b pulses needed | k* not reached on O1; O2 block AUC at k = 20: 0.687 [0.581, 0.784] | 0.0075 |
| `prereg_q38/` | Q38 hand covariate | H1 residualized 0.535 [0.518, 0.553] | 0.0020 |
| `prereg_family/` | Holm table | 6 of 7 rejected | - |

Log: `logs/prereg_run.log` (14 s wall clock, 1.0 GB peak). Known cosmetic flaw: `primary.json`
of Q01-Q03 stores `tail` as a number (1.0/2.0) instead of the word; p values unaffected.
