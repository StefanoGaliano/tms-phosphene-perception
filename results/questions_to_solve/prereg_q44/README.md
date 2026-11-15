# Q44 stimulus-general training, ds006171 + Eklund 2018 (37g; pre-registered, docs/QUESTIONS_PREREG.md §15, tag questions-prereg-amend8)

Run 2026-11-15, `scripts/questions_q44_pooled.py`; training 34 ds006171 subjects + 38 Eklund sessions
(epochs rebuilt 2026-11-15, 38/39 ok, small 06 failed as in the first build); 64 channels; equal
total weight per dataset; C = 0.01. Tested on 33 TMS recordings / 17 subjects.

| Training | TMS AUC M0 (bins) | TMS AUC M1 (window mean) |
|---|---|---|
| ds006171 only | 0.540 (14/17) | 0.555 (15/17) |
| Eklund only | 0.547 (16/17) | **0.585** (15/17) |
| pooled | 0.550 (15/17) | 0.575 (15/17) |
| pooled, left/right-symmetric | 0.528 (13/17) | - |

**[P] M0 pooled - ds006171 only:** +0.010 [0.003, 0.017], p 0.0079 one-sided, 12/17 -> significant
(paperlike +0.009, p 0.0037, 15/17).
Secondary: M1 pooled - ds only +0.020 [0.010, 0.029], p 0.0009, 16/17 (paperlike +0.019, p 0.0007);
symmetric - pooled -0.022, p 0.0001, 0/17 (symmetry hurts, as mirroring did in Phase 2 A9).
Cross-dataset (natural): ds006171 -> Eklund 0.653 (M0) / 0.585 (M1); Eklund -> ds006171 0.618 / 0.622.

**Interpretation (prereg rule):** a stimulus-general training transfers better, so part of the gap is
stimulus-/dataset-specific content. Effects add up with Q42: pooled window-mean 0.575 vs the frozen
recipe 0.540. Eklund-only training transfers best (0.585, exploratory): Eklund's contrast (aware
correct vs unaware correct, ratings) may be closer to the phosphene report than ds006171's yes/no.
Lateralized information does not help on TMS but symmetrizing removes signal: the phosphene
signature keeps hemisphere-specific structure.
