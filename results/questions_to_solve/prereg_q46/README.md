# Q46 v2 recipe tested once on IPS phosphenes (pre-registered, docs/QUESTIONS_PREREG.md §17, tag questions-prereg-amend10)

Run 2026-11-15, `scripts/questions_q46_v2_ips.py`; v2 = ds006171 (34) + Eklund (38), equal dataset weight,
window mean 0.15-0.60 s, C = 0.01. IPS primary: 30 recordings / 17 subjects, 64 channels.

| | primary | paperlike |
|---|---|---|
| frozen detector (Q41) | 0.539 | 0.535 |
| v2 | 0.543 (16/17 > 0.5, p 0.0012) | 0.535 (p 0.017) |
| **[P] v2 - frozen** | **+0.004 [-0.024, 0.032], p 0.40, 10/17** | +0.000, p 0.49 |
| Eklund only (exploratory best on occipital) | 0.540 | 0.532 |
| P3 site v2 / frozen | 0.551 / 0.525 (diff p 0.089) | 0.539 / 0.525 |
| P4 site v2 / frozen | 0.544 / 0.544 | 0.538 / 0.541 |
| hand groups (right n 8 / left n 3) | 0.546 / 0.490, p 0.28 | 0.545 / 0.450, p 0.19 |

**Interpretation (prereg rule):** null. The improvements found on the occipital data (0.540 -> 0.575)
do not generalize to held-out parietal phosphenes; the frozen detector remains the reference. v2 still
reads IPS phosphenes above chance at the frozen level (0.543), so nothing is lost, but nothing gained.
The occipital gain was likely partly specific to that dataset (choices made on it) or to occipital
stimulation. Eklund-only, the exploratory best on occipital (0.585), gives 0.540 here.
