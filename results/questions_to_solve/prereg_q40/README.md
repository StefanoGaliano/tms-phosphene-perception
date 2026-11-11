# Q40 reverse transfer (pre-registered, docs/QUESTIONS_PREREG.md §11, tag questions-prereg-amend4)

Run 2026-11-11, `scripts/questions_q40_reverse.py`. Per-time-point decoder trained on all included TMS
recordings (phosphene vs none, 64 ch), tested on the 34 ds006171 subjects (seen vs unseen).

| Test | primary | paperlike |
|---|---|---|
| **[P] 0.15-0.60 s, AUC > 0.5** | **0.514 [0.504, 0.523], d_z 0.47, p 0.0042, 23/34** | 0.516 [0.507, 0.524], p 0.0008, 25/34 |
| VAN 0.18-0.28 s | 0.496, p 0.82 | 0.502, p 0.31 |
| LP 0.35-0.55 s | 0.521 [0.507, 0.535], p 0.0025 | 0.523, p 0.0018 |
| prestimulus -0.5-0 s | 0.501, p 0.25 | 0.500, p 0.56 |
| cluster over time | 0.49-0.99 s, p 0.0004 | 0.30-0.99 s, p 0.0001 |

**Interpretation (rule fixed in the prereg):** [P] significant -> the phosphene-vs-none pattern carries
information about natural seeing that cannot come from the response hand (yes/no positions randomized
in ds006171); the shared signal is two-way. It is late (LP window, cluster from ~0.3-0.5 s), not in the
VAN window. Limit: a late report/decision signal common to both tasks (both report "seen") is not
excluded by this test; only the hand is.
