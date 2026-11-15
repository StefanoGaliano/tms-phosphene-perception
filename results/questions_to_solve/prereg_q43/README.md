# Q43 label-free domain alignment (37h; pre-registered, docs/QUESTIONS_PREREG.md §14, tag questions-prereg-amend7)

Run 2026-11-15, `scripts/questions_q43_align.py`; Q42 models; each training subject and TMS recording
centred and Ledoit-Wolf whitened on its own trials without labels (M3: re-centred tangent vectors).

| Model | TMS raw | TMS aligned | Natural LOSO aligned (raw, Q42) |
|---|---|---|---|
| M0 50 ms bins | 0.540 (14/17) | 0.551 (14/17) | 0.720 (0.750) |
| M1 window mean | 0.555 | 0.557 (16/17) | 0.647 (0.636) |
| M3 covariance | 0.545 | 0.545 | 0.625 (0.621) |

**[P] aligned M0 - M0:** +0.011 [0.006, 0.017], p 0.0002 one-sided, 15/17 -> significant (paperlike
+0.010, p 0.0068). Secondary: M1 +0.002, p 0.67; M3 +0.000, p 0.87.

**Interpretation (prereg rule):** part of the gap is a label-free domain shift, correctable per
recording without labels. But the gain appears only for the time-binned features and brings M0 to
about where the window mean already was (0.551 vs 0.555); aligning the window mean adds nothing. So
alignment and latency tolerance fix overlapping parts of the gap, not two separate ones. Alignment
costs 0.03 natural AUC for M0.
