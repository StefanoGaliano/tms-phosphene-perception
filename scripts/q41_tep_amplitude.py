"""Q41 secondary (d), Q37a as pre-registered: H1 with P(seen) residualized on the trial-wise
TMS-evoked-response amplitude, on the IPS primary pipeline (QUESTIONS_PREREG §12, amendment 5:
"the occipital controls unchanged: ... Q37a (TEP amplitude residualization)").

scripts/q41_secondary.py ran the Q37a first-look variants (mean_sub, ssp3) but not this test;
added 2026-11-17. Same code as the occipital Q37a (start_vision.questions.q37a): amplitude = least-
squares projection of each trial on the recording's label-free mean template, 0-0.6 s, on the
detector input (64 contract channels); P(seen) residualized on it within each recording.
Included recordings only (results/ips/inclusion_primary.tsv); subject = mean over sites;
one-sided sign flip for the residualized AUC, two-sided for amplitude alone (seed 20261109).

Run from Start_Vision/: ~/miniconda3/envs/eegunify/bin/python -I scripts/q41_tep_amplitude.py
Output: results/ips/secondary/tep_amplitude/{per_recording.tsv, group.tsv, results.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mne
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from start_vision.decoding.detector import Detector  # noqa: E402
from start_vision.phase3.analyses import group_one_sided, subject_values  # noqa: E402
from start_vision.phase3.score import detector_input  # noqa: E402
from start_vision.questions.q37a import recording_values, tep_amplitude  # noqa: E402
from start_vision.questions.stats import one_sample  # noqa: E402

RES = ROOT / "results" / "ips"
OUT = RES / "secondary" / "tep_amplitude"
EPO = ROOT / "derivatives" / "sv_ips_clean" / "primary"
SEED = 20261109


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    det = Detector.load(ROOT / "detectors" / "sv-perception-tms-robust-v1.0.0")
    inc = pd.read_csv(RES / "inclusion_primary.tsv", sep="\t", dtype={"subject": str})
    inc = inc[inc["included"]]
    rows = []
    for rid, sub, site in inc[["recording", "subject", "site"]].itertuples(index=False):
        ep = mne.read_epochs(EPO / f"{rid}_desc-detector_epo.fif", verbose="error")
        e = detector_input(ep, det)
        g = pd.DataFrame({"seen": ep.metadata["seen"].to_numpy().astype(int),
                          "p_seen": det.from_epochs(e),
                          "amp": tep_amplitude(e.get_data() * 1e6, e.times)})
        r = {"recording": rid, "subject": sub, "site": site, **recording_values(g)}
        rows.append(r)
        print(rid, {k: round(v, 3) for k, v in r.items() if k.startswith("auc")}, flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_recording.tsv", sep="\t", index=False, float_format="%.5f")
    res = {
        "H1 residualized on TEP amplitude (one-sided)":
            group_one_sided(subject_values(per, "auc_res"), seed=SEED),
        "H1 frozen (check = q41 primary)":
            group_one_sided(subject_values(per, "auc_frozen"), seed=SEED),
        "TEP amplitude alone, AUC vs 0.5 (two-sided)":
            one_sample(subject_values(per, "auc_amp"), 0.5, tail=2, seed=SEED),
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    t = pd.DataFrame([{"test": k, **v} for k, v in res.items()])
    t.to_csv(OUT / "group.tsv", sep="\t", index=False, float_format="%.5f")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
