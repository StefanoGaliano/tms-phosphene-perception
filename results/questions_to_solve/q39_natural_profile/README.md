# Q39 follow-up (EXPLORATORY, not pre-registered): time profile of VAN-trained classifiers in natural vision

Run 2026-11-11 on existing Phase 2 A2 matrices (`derivatives/sv_decode_group/a2_generalization.npz`:
within-subject cross-validated temporal generalization, 128 ch, N 34). Compared with the Q39 TMS
matrix (cross-subject transfer, 64 ch, N 17). Not like-for-like (within- vs across-subject, 128 vs 64 ch).

Classifiers trained 0.18-0.28 s, mean AUC by test time:

| Test time | natural (within subject) | TMS (Q39 primary) |
|---|---|---|
| 0.08-0.18 s | 0.529 | 0.515 |
| 0.18-0.28 s | 0.580 | 0.535 |
| 0.28-0.38 s | 0.584 | 0.538 |
| peak (group) | **0.27 s** (median subject 0.28 s) | **0.48 s** |
| 0.5 s | 0.538 | 0.539 |

Early minus late: natural -0.055, p 0.0001 (later also stronger there). Profile correlation 0.68.

Reading: in natural vision the VAN-window code peaks just after its training window and decays by
0.5 s; in the TMS data it keeps rising to ~0.48 s. The TMS peak is ~200 ms later. Exploratory only:
the comparison is not like-for-like, and the late TMS window overlaps the response-locked hand
lateralization (R3) and the LP. Hypothesis for a future pre-registered test, not a claim.
