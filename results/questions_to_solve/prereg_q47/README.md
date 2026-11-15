# Q47 within-recording decodability, parietal vs occipital (pre-registered, docs/QUESTIONS_PREREG.md §18, tag questions-prereg-amend11)

Run 2026-11-15, `scripts/questions_q47_within_ips.py`; S8 code unchanged on 30 IPS recordings / 17
subjects; occipital from the saved S8 curves (17 subjects).

| | IPS | occipital | diff [95 % CI] | p (IPS lower, one-sided) |
|---|---|---|---|---|
| **[P] within-recording AUC, 0.15-0.60 s** | 0.557 | 0.590 | -0.034 [-0.065, -0.003] | **0.023** |
| (b) 0.07-0.98 s | 0.591 | 0.617 | -0.026 [-0.055, 0.003] | 0.046 |
| (a) IPS cluster vs 0.5 | 258-984 ms, p 0.0001, mean 0.608 | (S8: 70-984 ms) | | |
| (c) share carried by the frozen detector, (transfer-0.5)/(within-0.5) | 0.69 | 0.44 | | descriptive |
| (d) P3 / P4 | 0.562 / 0.555 | | | |

**Interpretation (prereg rule):** significant. Parietal phosphene pulses are less separable in the EEG
than occipital ones, consistent with the ceiling reading of Q46. The frozen detector already captures
about 69 % of the parietal within-recording separability (44 % occipital), leaving little room for a
better recipe there. Separable activity also starts later after parietal pulses (258 vs 70 ms). Not
proof: within-recording decoding includes report- and hand-related activity.
