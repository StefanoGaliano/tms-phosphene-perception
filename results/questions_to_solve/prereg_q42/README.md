# Q42 latency-tolerant features (37e; pre-registered, docs/QUESTIONS_PREREG.md §13, tag questions-prereg-amend6)

Run 2026-11-15, `scripts/questions_q42_latency.py`; trained on 34 ds006171 subjects (tms-robust 64 ch,
C = 0.01), tested on 33 TMS recordings / 17 subjects. A first run was stopped during M3 (features
recomputed in every LOSO fold, slow) and rerun with features cached per subject; M0-M2 identical
(`run_stopped.log`).

| Model | TMS AUC primary (n > 0.5) | TMS AUC paperlike | Natural LOSO AUC |
|---|---|---|---|
| M0 50 ms bins 0.15-0.60 s | 0.540 (14/17) | 0.540 | 0.750 |
| M1 one window mean | 0.555 (15/17) | 0.552 | 0.636 |
| M2 M0 shifted +0.20 s (descriptive, circular) | 0.550 (15/17) | 0.553 | - |
| M3 log-Euclidean covariance | 0.545 (17/17) | 0.543 | 0.621 |

**[P] M1 - M0 > 0:** +0.015 [0.001, 0.030], d_z 0.48, p 0.033 (one-sided), 11/17 positive -> significant.
Secondary: paperlike M1 - M0 +0.013, p 0.094; M2 - M0 +0.010, p 0.51; M3 - M0 +0.005, p 0.57.

**Interpretation (prereg rule):** part of the transfer loss is latency. The gain is small and does
not reach significance in the paperlike pipeline. Collapsing time costs 0.11 AUC in natural vision
(0.750 -> 0.636) yet gains on TMS: the fine time course that helps in natural vision does not carry
over to phosphenes. The shifted detector (M2) does not beat M0, so the gain is from timing tolerance,
not from one fixed lag.
