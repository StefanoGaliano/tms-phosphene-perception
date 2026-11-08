# Phase 3 step 9: cleaning QA on sub-03 and sub-04, site O1 (2026-11-08)

`sv-p3-clean -s 03 -s 04 --site O1` (primary pipeline), then `python -I scripts/phase3_clean_qa.py`.
Figures `qa/F1_sub-0{3,4}_site-O1.png`, `qa/F2_artifact_end.png`; tables `qa/clean_qa.tsv`,
`qa/muscle_q5.tsv`. Both recordings were already used in the go/no-go, so nothing new is unblinded.

## Run
- 3 min for both recordings (about 85 s each), peak memory 16.5 GB. Rerun with the same key is
  a no-op (`cached`).
- Pulse artifact end (C1): sub-03 9.6 ms (cut stays -2..10 ms), sub-04 34.4 ms (cut extended to
  the 20 ms cap). The C1 re-measure after the decay ICA is 20-22 ms in both; the measure is crude
  (it also picks up the early TMS-evoked response), use the figures, not that number.
- Decay ICA (TESA rule): 1 component removed of 31 (sub-03) and 34 (sub-04). In F1 the large
  early decay (0-100 ms) is gone; peaks near 100 ms and 300 ms remain (TMS-evoked response, and
  late activity that includes the response), no step or ringing at the cut edges.
- eegunify ocular ICA: 3 and 2 components removed; pyprep bad channels: TP7 (sub-03), none (sub-04).
- Trial rejection (§6): sub-03 kept 357/360 (0 ptp, 2 HEOG); **sub-04 kept 215/360** (115 ptp,
  45 HEOG step). The noisiest channels in sub-04 are O1, PO7 and the interpolated Iz/P9, under
  or next to the coil. S100 share after rejection 0.60 (0.58 before).
- GMFP 150-600 ms is 8.7x / 10.5x the prestimulus GMFP: the detector window holds a large
  pulse-evoked response common to both classes (same pulse), as expected; the detector reads
  the seen-vs-unseen difference after its per-recording normalization.

## Q5 measure (ongoing muscle, 20-35 Hz, lateral edge channels)

| data | power (µV²/Hz) | log10 vs ds006171 median |
|---|---|---|
| GIN sub-03 O1 | 0.096 | -0.56 |
| GIN sub-04 O1 | 0.798 | +0.36 |
| ds006171 sub-03 nocue | 1.064 | +0.49 |
| ds006171 sub-04 nocue | 0.231 | -0.18 |
| ds006171 sub-05 nocue | 0.402 | +0.06 |
| ds006171 sub-06 nocue | 0.293 | -0.07 |

Both GIN recordings fall inside the range of the training recordings. No sign of more ongoing
muscle than in training, so on this measure the training-like primary (option A) fits.

## For the user (stop point)
1. Q5: primary with or without the paper's muscle ICA round.
2. Approve the four project-choice parameters in `docs/PHASE3_REFERENCES.md` (k = 3, 2 ms,
   20 ms cap, 1000 Hz).
3. sub-04: 32 % of trials rejected on peak-to-peak near the coil. Keep the §6 rule as is
   (recommended: it is the training rule, and 215 trials pass the 40/30 minimums) or exclude
   coil-adjacent channels from the ptp test.
