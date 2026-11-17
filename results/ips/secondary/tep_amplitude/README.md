# Q41 secondary (d): H1 residualized on the trial-wise TEP amplitude (IPS, primary pipeline)

Pre-registered in docs/QUESTIONS_PREREG.md §12 (amendment 5): "(d) the occipital controls unchanged:
... Q37a (TEP amplitude residualization)". `scripts/q41_secondary.py` ran the Q37a first-look variants
(mean_sub, ssp3) but not this test; it was run on 2026-11-17 with `scripts/q41_tep_amplitude.py`
(same code as the occipital Q37a: amplitude = projection of each trial on the recording's label-free
mean response, 0-0.6 s, 64 contract channels; P(seen) residualized within recording). Reported in the
paper as a deviation (run after the other parietal tests).

| Test | N | AUC [95 % CI] | d_z | p |
|---|---|---|---|---|
| H1 residualized on TEP amplitude (one-sided) | 17 | 0.532 [0.513, 0.551] | 0.78 | 0.0033 |
| H1 frozen (check, = Q41 primary) | 17 | 0.539 [0.519, 0.559] | 0.90 | 0.0014 |
| TEP amplitude alone vs 0.5 (two-sided) | 17 | 0.521 [0.500, 0.543] | 0.45 | 0.079 |

Reading: unlike the occipital data (amplitude alone 0.545, residualized 0.522), the pulse-response
amplitude explains little of the parietal effect, although intensity was adjusted during the IPS runs.
