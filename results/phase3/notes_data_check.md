# Phase 3 data check (2026-11-08)

Data: GIN `DB_123/Phosphenes_occipital` (CC0), Siviero et al. 2023 Sensors 23:8833
(`paper_for_info/`). Facts and sources in `docs/DATASETS.md` (Phase 3). On disk: O1 site,
subjects 03-11 (9 of 16); O2 not yet downloaded. No analysis plan yet: this is a data check.

## 1. Which code is "phosphene present"? S100 (9/9 subjects)

The authors do not define S100/S200. Present = 'm' key, right hand; absent = 'z', left hand
(paper, 2.3). Right-hand preparation makes C3 more negative than C4. Test per subject:
C3-C4, 0.1-10 Hz, pulse-locked, baseline -0.2..0 s, mean 0.3-1.2 s, S100 minus S200,
trial-label permutation (5000). Script kept outside the repo (scratchpad `lrp_stat.py`).

| Subject | n S100 | n S200 | S100 share | C3-C4 S100 (uV) | S200 (uV) | diff | p |
|---|---|---|---|---|---|---|---|
| 03_RS | 253 | 106 | 0.70 | -1.72 | +0.55 | -2.27 | <0.001 |
| 04_DP | 209 | 151 | 0.58 | -0.85 | +1.91 | -2.76 | <0.001 |
| 05_MC | 217 | 138 | 0.61 | -2.72 | -0.70 | -2.01 | <0.001 |
| 06_AV | 86 | 270 | 0.24 | -3.75 | -0.01 | -3.74 | <0.001 |
| 07_NZ | 126 | 196 | 0.39 | -1.41 | -0.77 | -0.64 | 0.33 |
| 08_MZ | 116 | 232 | 0.33 | -1.29 | +0.55 | -1.84 | <0.001 |
| 09_GT | 228 | 132 | 0.63 | -0.68 | -0.17 | -0.51 | 0.27 |
| 10_JS | 176 | 182 | 0.49 | -0.67 | +0.33 | -1.00 | 0.059 |
| 11_GP | 146 | 212 | 0.41 | +0.09 | +3.93 | -3.84 | <0.001 |

Direction negative in 9/9 subjects (two-sided sign test p = 0.004; the subject is the unit).
**S100 = phosphene present, S200 = absent.** Still an inference: ask the authors to confirm.
Caveat: the window is pulse-locked (RTs are not in the data), so it mixes response timing.

## 2. Phosphene rate

S100 share 0.24-0.70 (median 0.49), close to the 50 % the threshold procedure targets.
With the Phase 1 rule (30-70 %), 06_AV (0.24) would fail; 03_RS (0.70) is on the edge. No
catch/sham trials exist, so the false-alarm half of that rule cannot be applied.

## 3. Fit to the frozen detector `sv-perception-tms-robust-v1.0.0`

- Channels: 59 scalp + HEOGL/HEOGR/VEOGUP/VEOGDOWN + LM (online ref RM not recorded).
  Contract channels missing: P9, P10, Iz, Fpz, AFz (5; AFz was the ground). `FP1`/`FP2` only
  differ in case. The contract allows at most 4 missing. **Decision 10 (user, 2026-11-08):
  interpolate all 5** before the detector (`start_vision.phosphenes`, tests
  `test_phosphenes.py`); tried on 03_RS (60 s): 64 channels out, interpolated SDs in the range
  of their neighbours.
- 5000 Hz continuous, raw (no TMS artifact removal). Needs pulse cut/interpolation
  (paper: -2..10 ms, cubic) and TMS-decay/muscle ICA before resampling to 128 Hz.
- Contract prestimulus z-score window -0.5..-0.1 s: each pulse is preceded by a 1300 ms ITI
  and a 700-1000 ms jitter after the previous answer, so the window holds no keypress.
- 360 trials per site, so the 40-trial minimum is met.

## 4. Quick go/no-go

See `gonogo/README.md`: detector AUC median 0.526 (8/9 > 0.5, p = 0.008), 0.520 without
motor channels, no trial-wise tracking of C3-C4. Decision: go (small effect). Email to the
authors (codes, RTs) sent by the user 2026-11-08 (to D. Bonfanti, cc I. Siviero); no answer yet.

## Open
- Download O2 and O1 subjects 13-18; rerun section 1 on all.
- Confirm the code with the authors.
- Missing in the design: no sham/off-target pulses (Phase 3 control) - the left vs right
  site and the contralateral field of the phosphene are the only internal controls.
