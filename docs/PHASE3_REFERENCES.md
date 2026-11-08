# Phase 3 references: sourced parameter values

Every number used by the Phase 3 cleaning chain (`plans/PHASE3_BLUEPRINT.md` §7) with its source.
Project rule: no value typed from memory. "Project choice" = no readable source; it needs the
user's approval (Decision 11, Q11) and is listed as such.

## Sourced

| Parameter (blueprint step) | Value | Source |
|---|---|---|
| Pulse cut window (C2) | -2..10 ms, cubic interpolation | Siviero et al. 2023, Sensors 23:8833, §2.4 (read in full, `paper_for_info/`) |
| Pulse window replaced by a constant before each ICA, re-interpolated after (C4) | — | Siviero 2023 §2.4 |
| Downsampling in the paper | 500 Hz (we use 1000 Hz, project choice C3) | Siviero 2023 §2.4 |
| Paper round-1 ICA purpose | remove TMS artifacts (`runica`), then 0.1-100 Hz Butterworth order 4 + 49-51 Hz band-stop; round 2 blinks and muscle; REST reference; 40 Hz low-pass; TBT rejection | Siviero 2023 §2.4 |
| TMS-evoked muscle component rule (C4) | ratio = mean of \|trial-averaged component\| in the target window / mean of \|trial-averaged component\| over the whole epoch; reject if ratio >= 8 | TESA `tesa_compselect.m` (github.com/nigelrogasch/TESA, master, last commit 95ef97e, 2026-04-12), lines 78-97 (doc) and 292-296, 536-540 (defaults and code) |
| Target window (C4) | 11..30 ms | same file, default `tmsMuscleWin = [11,30]` |

Note: these are TESA's software defaults, which its own documentation says "can be adapted by
the user and may differ" between data (line 26). The paper cites TESA (Rogasch et al. 2017)
but does not give the values it used. Rogasch et al. 2017 (NeuroImage 147:934-951) was not read
in full; the values above come from the code, not the paper.

## Project choices (pending the user's approval at the Step 9 stop)

| Parameter | Proposed | Why |
|---|---|---|
| Artifact end (C1): threshold `k` x prestimulus SD (-500..-10 ms) on the 95th percentile over channels of the pulse-locked median \|x\| | k = 3 | QA measure only; it decides whether the cut is extended |
| Artifact end (C1): must stay below for `m` | 2 ms | idem |
| Cut extension cap (C2) | 20 ms | keeps the cut well before the 150 ms feature window; longer artifacts are left to round-1 ICA |
| Resample before ICA (C3) | 1000 Hz | keeps decay resolution, makes ICA tractable |
