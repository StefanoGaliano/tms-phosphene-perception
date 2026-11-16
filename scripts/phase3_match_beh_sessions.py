"""Match E-Prime sessions (beh_eprime_18ss.xlsx) to GIN O1/O2 recordings by response sequence.

EEG: S100 -> 'm' (phosphene, right hand), S200 -> 'z'. Behaviour: vresp.RESP per Subject x Session
in trial order (Block, Trial); no-response trials kept as '-'. Similarity = difflib ratio.
Writes results/phase3/beh_session_match.tsv.
Usage: python -I scripts/phase3_match_beh_sessions.py
"""
from difflib import SequenceMatcher
from pathlib import Path

import mne
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sourcedata" / "phosphenes_occipital"

beh = pd.read_csv(SRC / "beh_eprime_18ss.tsv", sep="\t")  # tsv copy of the xlsx (no openpyxl in env)
beh["r"] = beh["vresp.RESP"].fillna("-")
rows = []
for site in ("O1", "O2"):
    for f in sorted((SRC / site).glob(f"*_{site}_SICL.set")):
        sub = int(f.name[:2])
        try:
            ann = mne.io.read_raw_eeglab(f, preload=False, verbose=False).annotations.description
        except Exception as e:  # noqa: BLE001
            rows.append({"subject": sub, "site": site, "error": str(e)[:80]}); continue
        eeg = "".join("m" if d == "S100" else "z" for d in ann if d in ("S100", "S200"))
        for ses, g in beh[beh.Subject == sub].groupby("Session"):
            b = "".join(g.sort_values(["Block", "Trial"])["r"])
            bn = b.replace("-", "")
            rows.append({"subject": sub, "site": site, "session": ses, "n_eeg": len(eeg),
                         "n_beh": len(b), "n_beh_resp": len(bn),
                         "ratio": round(SequenceMatcher(None, eeg, b, autojunk=False).ratio(), 3),
                         "ratio_resp_only": round(SequenceMatcher(None, eeg, bn, autojunk=False).ratio(), 3),
                         "exact_resp_only": eeg == bn})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "phase3" / "beh_session_match.tsv", sep="\t", index=False)
print(out.to_string(index=False))
