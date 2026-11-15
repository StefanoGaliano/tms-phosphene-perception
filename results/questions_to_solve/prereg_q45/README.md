# Q45 prestimulus alpha phase and phosphenes (Q09; pre-registered, docs/QUESTIONS_PREREG.md §16, tag questions-prereg-amend9)

Run 2026-11-15, `scripts/questions_q45_phase.py`; GIN occipital, 33 recordings / 17 subjects; posterior
ROI (A1 channels), 8-13 Hz, segment -0.60..-0.02 s cut before filtering, mirror-padded.

| Test | primary | paperlike |
|---|---|---|
| [P] POS z at -0.02 s, > 0 one-sided | -0.32 [-0.64, 0.03], d_z -0.44, p 0.95, 5/17 | -0.18, p 0.85, 6/17 |
| (a) time-resolved POS, -0.40..-0.05 s | no cluster | no cluster |
| (b) phosphene rate over 6 phase bins | 0.443-0.485, depth 0.042 | depth 0.037 |
| (c) detector AUC, alpha peak half vs trough half | 0.541 vs 0.539, diff +0.003, p 0.82 | 0.540 vs 0.540, p 0.97 |
| (d) POS z of P(seen) median split | -0.02, p 0.55 | -0.02, p 0.56 |

**Interpretation (prereg rule):** null. No evidence that prestimulus alpha phase predicts phosphene
report here; Samaha 2017's phase effect is not reproduced with this pipeline (cleaning, 128 Hz, fixed
band, phase at -0.02 s from a cut segment) - not proof of absence. Together with A1 (alpha power null),
the phosphene reports in this dataset are not tied to the measured prestimulus alpha state.
Practical result for a stimulation study: the read-out works equally at the alpha peak and trough
(c), so phase-locked stimulation would not need a phase-specific detector.
