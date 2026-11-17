# Q47 within-recording decodability, parietal vs occipital (pre-registered, docs/QUESTIONS_PREREG.md §18, tag questions-prereg-amend11)

Run 2026-11-15, `scripts/questions_q47_within_ips.py`; S8 code unchanged on 30 IPS recordings / 17
subjects; occipital from the saved S8 curves. Re-run 2026-11-16 with sub-18 O1 (occipital N 18).
Numbers below updated 2026-11-17 from `q47.json`; `c_share_carried` recomputed with the current
frozen-detector transfer (the script had the occipital value of the 17-subject run, 0.5399,
hard-coded; now read from `results/phase3/t2_tests_primary.tsv` and `results/ips/summary.json`).

| | IPS | occipital | diff [95 % CI] | p (IPS lower, one-sided) |
|---|---|---|---|---|
| **[P] within-recording AUC, 0.15-0.60 s** | 0.557 | 0.584 | -0.028 [-0.060, 0.003] | **0.056** |
| (b) 0.07-0.98 s | 0.591 | 0.612 | -0.021 [-0.050, 0.008] | 0.095 |
| (a) IPS cluster vs 0.5 | 258-984 ms, p 0.0001, mean 0.608 | (S8: 78-984 ms) | | |
| (c) share carried by the frozen detector, (transfer-0.5)/(within-0.5) | 0.69 | 0.45 | | descriptive |
| (d) P3 / P4 | 0.562 / 0.555 | | | |

**Interpretation (prereg rule):** not significant (p 0.056; it was 0.023 on 17 occipital subjects).
Parietal pulses tend to be less separable than occipital ones, but the difference is not
established. Separable activity starts later after parietal pulses (258 vs 78 ms), and the frozen
detector captures a larger share of the within-recording separability on parietal pulses (69 % vs
45 %). Not proof of a ceiling: within-recording decoding includes report- and hand-related
activity (the response hand is tied to the report within every recording).
