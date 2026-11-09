"""Exploratory (not pre-registered): where is the phosphene VAN? Group S100 - S200 map in the VAN
window (180-280 ms, Phase 1 reference) per stimulation site, Phase 3 primary included recordings.
Baseline -100..0 ms as S6. Output: results/questions_to_solve/van_topography_explore/."""
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from start_vision.decoding import stats

root = Path.cwd()
out = root / "results/questions_to_solve/van_topography_explore"
inc = pd.read_csv(root / "results/phase3/inclusion_primary.tsv", sep="\t", dtype={"subject": str})
inc = inc[inc.included]
rows = []
for rid in inc.recording:
    ep = mne.read_epochs(root / f"derivatives/sv_p3_clean/primary/{rid}_desc-detector_epo.fif",
                         verbose="error").pick("eeg")
    t = ep.times
    x = ep.get_data() * 1e6
    x -= x[..., (t >= -0.1) & (t <= 0)].mean(-1, keepdims=True)
    y = ep.metadata.seen.to_numpy().astype(bool)
    w = x[..., (t >= 0.18) & (t <= 0.28)].mean(-1)
    tep = x[..., (t >= 0.18) & (t <= 0.28)].mean(-1).mean(0)
    sd = w.std(0)
    for c, d, a, s in zip(ep.ch_names, w[y].mean(0) - w[~y].mean(0), tep, sd, strict=True):
        rows.append({"recording": rid, "site": rid.split("-")[-1], "channel": c,
                     "diff_uv": d, "tep_uv": a, "trial_sd_uv": s})
df = pd.DataFrame(rows)
df.to_csv(out / "per_recording_channel.tsv", sep="\t", index=False, float_format="%.4f")
res = []
for (site, ch), g in df.groupby(["site", "channel"]):
    v = g.diff_uv.to_numpy()
    res.append({"site": site, "channel": ch, "n": len(v), "mean_diff_uv": v.mean(),
                "p_two_sided": stats._signflip_p(v, 10000, 0, tail=0),
                "mean_tep_uv": g.tep_uv.mean(), "mean_trial_sd_uv": g.trial_sd_uv.mean()})
r = pd.DataFrame(res)
r.to_csv(out / "group_by_site_channel.tsv", sep="\t", index=False, float_format="%.4f")
for site in ("O1", "O2"):
    s = r[r.site == site].sort_values("mean_diff_uv")
    print(f"--- site {site}: most negative S100 - S200 (180-280 ms)")
    print(s.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    roi = ["O1", "PO7"] if site == "O1" else ["O2", "PO8"]
    print("ROI rows:"); print(s[s.channel.isin(roi + ["P7", "P8", "TP7", "TP8", "P9", "P10"])].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
