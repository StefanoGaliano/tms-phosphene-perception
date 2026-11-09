# Q02, Q03 (ds006171), Q04, Q17, Q27 from saved results (2026-11-09, exploratory)

Script `scripts/questions_saved_results.py`; no epochs used (`runs/` removed on purpose). Group
tables: `group.tsv`; per subject: `q02_q03_ds006171.tsv`, `q04_attention.tsv`, `q17_size.tsv`;
`q27_below_chance.tsv`. Two-sided sign-flip (10 000), bootstrap 95 % CI, d_z.

## Q02 / Q03 on ds006171 (held-out LOSO scores, orientation-question trials, tasks pooled)

| | AUC correct vs wrong orientation [95 % CI] | contrast regressed out | behaviour: orientation accuracy |
|---|---|---|---|
| Q02 seen trials (N 31) | 0.590 [0.537, 0.642], d_z 0.60, p 0.003, 24/31 | 0.588 [0.538, 0.633], p 0.001 | 0.878 |
| Q03 unseen trials (N 34) | 0.521 [0.492, 0.550], p 0.18 | 0.521 [0.495, 0.547], p 0.14 | 0.518 (chance 0.5, p 0.17) |

Contrast does not differ between correct and wrong trials (seen p 0.30, unseen p 0.61).
- Q02 replicates Eklund on the training dataset: among seen trials, correct identification goes
  with a higher score, not explained by contrast.
- Q03 does **not** appear in ds006171, and neither does unconscious behaviour: unseen orientation is
  at chance. In Eklund, unseen localization was well above chance (about 50 % vs 25 %) and the
  detector followed it. Reading across both datasets: the detector follows the stimulus evidence
  that behaviour can use; where unseen trials carry none (ds006171), it gives none.

## Q04 attention (ds006171 tasks; subjects with both tasks)

| Measure | informative (or noninf.) minus nocue | N | p |
|---|---|---|---|
| VAN contralateral (µV) | -0.21 [-0.65, 0.22] | 27 | 0.36 |
| VAN bilateral, informative | -0.38 [-0.92, 0.20] | 27 | 0.20 |
| VAN bilateral, noninformative | -0.16 [-0.74, 0.44] | 27 | 0.61 |
| LOSO AUC, informative | -0.017 [-0.040, 0.006]; TOST ±0.03 p 0.13 | 27 | 0.16 |
| LOSO AUC, noninformative | +0.005 [-0.015, 0.026]; TOST ±0.03 p 0.015 | 27 | 0.63 |

LOSO AUC per task: nocue 0.737, noninformative 0.741, informative 0.720 (all > 0.5, every subject).
No attention effect on the VAN or on the detector in either direction. The noninformative cue is
equivalent to no cue within ±0.03; the informative cue is not shown equivalent (CI reaches -0.040).
Against Bola 2021: here attention does not change the awareness signal measurably, though the
VAN difference CIs are wide (about ±0.5 µV).

## Q17 size (Eklund, frozen detector, sessions with both sizes, N 12)
AUC large minus small +0.002 [-0.028, 0.030], p 0.89; TOST ±0.03 p 0.053 (just short of equivalence).
Consistent with Eklund's inconclusive size result.

## Q27 below-chance generalization (Phase 2 A2, post-stimulus, N 34)
No cluster of AUC < 0.5 (smallest p 0.35; lowest group-mean AUC 0.488, train 63 ms / test 602 ms).
No reversal like King 2014's auditory one.
