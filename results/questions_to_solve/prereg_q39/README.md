# Q39 time-shifted transfer (pre-registered, docs/QUESTIONS_PREREG.md §10, tag questions-prereg-amend3)

Run 2026-11-11, `scripts/questions_q39_tg.py`; GeneralizingEstimator trained on 34 ds006171 subjects
(28 066 trials, tms-robust 64 ch), tested on 33 TMS recordings / 17 subjects; times 0-1.0 s.

| Test | primary | paperlike |
|---|---|---|
| [P] train VAN 0.18-0.28 s, test 0.08-0.18 s, AUC > 0.5 | 0.515 [0.504, 0.526], d_z 0.62, p 0.0073, 12/17 | 0.509 [0.501, 0.518], p 0.029, 11/17 |
| (a) early minus late (test 0.28-0.38 s) | -0.024, p 0.0003, 2/17 positive | -0.026, p 0.0002 |
| VAN diagonal / VAN train, late test | 0.535 / 0.538 | 0.530 / 0.535 |
| (b) train LP 0.35-0.55, test 0.25-0.35 | 0.521 [0.510, 0.532], p 0.0014 | 0.520, p 0.0036 |
| (c) test time of peak, VAN-trained (group / median subject) | 0.477 s / 0.477 s | 0.477 s / 0.508 s |
| (d) 2-D cluster | one cluster, p 0.0003, mean AUC 0.543 | p 0.0002, 0.541 |

**Interpretation (rule fixed in the prereg):** [P] significant but (a) not positive -> broad
generalization, not a specific earlier shift. The natural VAN-window code transfers at earlier and
later test times, more strongly later (peak ~0.48 s). No evidence that the signature arrives earlier
after TMS; the paper keeps "partial match" without a time-shift claim.
