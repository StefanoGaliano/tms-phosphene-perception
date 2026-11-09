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
