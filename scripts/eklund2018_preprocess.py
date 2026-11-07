"""Eklund & Wiens 2018 preprocessing (A12 issue #1): eegunify convert + erp per session
(4 in parallel), then the group stage per Gabor size.

Run from the repo root: python scripts/eklund2018_preprocess.py
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path.cwd()
B = Path(sys.executable).parent


def one(gabor: str, sub: str) -> str:
    d = ROOT / "runs" / "eklund2018" / gabor
    cfg = str(ROOT / "configs" / "generated" / f"eklund2018_{gabor}.yaml")
    with open(d / f"convert_{sub}.log", "w") as f:
        subprocess.run([B / "eegunify", "convert", cfg, "-s", sub, "--root", "."], cwd=d,
                       stdout=f, stderr=subprocess.STDOUT)
    with open(d / f"erp_{sub}.log", "w") as f:
        subprocess.run([B / "eegunify", "erp", "run", cfg, "-s", sub, "--to", "erp",
                        "--root", "."], cwd=d, stdout=f, stderr=subprocess.STDOUT)
    stages = [ln.split() for ln in (d / f"erp_{sub}.log").read_text().splitlines()
              if ln.split()[:2][1:] == [sub]]
    msg = f"{gabor} {sub}: " + " ".join(f"{s[0]}={s[2]}" for s in stages)
    print(msg, flush=True)
    return msg


jobs = []
for g in ("large", "small"):
    (ROOT / "runs" / "eklund2018" / g).mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs" / "generated" / f"eklund2018_{g}.yaml").read_text())
    jobs += [(g, s) for s in cfg["recordings"]["subjects"]]
with ThreadPoolExecutor(4) as ex:
    list(ex.map(lambda j: one(*j), jobs))
for g in ("large", "small"):
    d = ROOT / "runs" / "eklund2018" / g
    with open(d / "group.log", "w") as f:
        cfg = str(ROOT / "configs" / "generated" / f"eklund2018_{g}.yaml")
        subprocess.run([B / "eegunify", "erp", "run", cfg, "--root", "."],
                       cwd=d, stdout=f, stderr=subprocess.STDOUT)
    print(g, "group:", (d / "group.log").read_text().strip().splitlines()[-1], flush=True)
