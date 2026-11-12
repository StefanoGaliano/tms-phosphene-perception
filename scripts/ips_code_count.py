"""Count S100/S200 events per IPS recording (GIN DB_123/IPS_parietal_phosphenes).

Reads only the EEGLAB .set headers (events), not the EEG data. Tests whether S100/S200
encode the answer (S100 the minority in every recording, near the paper's 33-37 %
phosphene rate) or the key (majority flips in subjects with the other key mapping;
keys were counterbalanced across participants, Bonfanti et al. 2025, p. 133).
Writes results/ips/code_count.tsv.
"""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sourcedata" / "ips_parietal_phosphenes"
OUT = ROOT / "results" / "ips" / "code_count.tsv"


def main() -> int:
    rows = []
    for f in sorted(SRC.glob("*/*.set")):
        m = sio.loadmat(f, squeeze_me=True, struct_as_record=False,
                        variable_names=["event", "srate", "nbchan"])
        c = collections.Counter(str(e.type).strip() for e in m["event"])
        n100, n200 = c.get("S100", 0), c.get("S200", 0)
        other = {k: v for k, v in c.items() if k not in ("S100", "S200")}
        rows.append({"site": f.parent.name, "file": f.name, "subject": f.name.split("_")[0],
                     "S100": n100, "S200": n200,
                     "S100_share": round(n100 / max(n100 + n200, 1), 3),
                     "other_events": ";".join(f"{k}:{v}" for k, v in sorted(other.items()))})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("\t".join(str(v) for v in r.values()))
    shares = [r["S100_share"] for r in rows]
    print(f"\nn={len(rows)}  S100 share min {min(shares)} max {max(shares)} "
          f"mean {sum(shares)/len(shares):.3f}  S100 majority in {sum(s > 0.5 for s in shares)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
