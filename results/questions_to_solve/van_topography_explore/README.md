# Exploratory: where is the phosphene VAN? (2026-11-09, not pre-registered)

Script `scripts/explore_phase3_van_topography.py`. Phase 3 primary, included recordings; S100 - S200
mean in 180-280 ms (Phase 1 VAN window), baseline -100..0 ms; per site, sign flip over recordings
(10 000), **uncorrected** over 64 channels x 2 sites. Files: `group_by_site_channel.tsv`,
`per_recording_channel.tsv`.

| VAN window, phosphene - none (µV) | O1 site (N 16) | O2 site (N 17) |
|---|---|---|
| Stimulated-side ROI (S6) | O1 -0.60 (p 0.004), PO7 -0.58 (0.025) | O2 +0.08 (0.72), PO8 -0.04 (0.79) |
| Other hemisphere, posterior | O2 -0.45 (0.08), PO8 -0.50 (0.04), P8 -0.45 (0.03) | PO7 -0.43 (0.01), P7 -0.51 (0.006), TP7 -0.47 (0.003) |
| Left central | C3 -0.04, C5 -0.12, CP5 -0.17 (n.s.) | C3 -0.31 (0.004), C5 -0.38 (0.003), CP5 -0.41 (0.002) |
| Right central | C4 +0.26, CP4 +0.19 (n.s.) | C4 +0.42 (0.024), CP4 +0.36 (0.018) |
| Pulse-evoked response at the ROI (all trials) | O1 -7.2 µV | O2 -8.5 µV |
| Trial SD at the ROI | about 5.3 µV | about 5.6 µV |

Reading:
- O1 site: a posterior negativity on both hemispheres, central channels flat: the better VAN candidate.
- O2 site: nothing on the stimulated side; a left lateral-posterior negativity together with left
  central negativity and right central positivity, the pattern of a prepared right-hand response
  (phosphene = right hand in this dataset). Likely response-hand activity, already present at
  180-280 ms and reaching lateral posterior channels (P7, TP7, PO7) that R1 did not remove.
- The pooled S6 ROI averages a negativity (O1 site) with nothing (O2 site): -0.25 µV.
- The effect (<= 0.6 µV) is small against the common pulse response (-7 to -8.5 µV) and the trial
  SD (about 5 µV) at channels under the coil.
