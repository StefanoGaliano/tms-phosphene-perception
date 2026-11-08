from pathlib import Path

import numpy as np
import scipy.io

from start_vision.phase3 import inventory as inv


def _set(path: Path, datfile: str, nbchan: int, pnts: int, codes: list[str]) -> None:
    ev = np.zeros(len(codes), dtype=[("type", object), ("latency", float)])
    for i, c in enumerate(codes):
        ev[i] = (c, 1000.0 + i * 15000.0)
    scipy.io.savemat(path, {"datfile": datfile, "nbchan": nbchan, "pnts": pnts,
                            "srate": 5000.0, "event": ev})


def _tree(tmp: Path) -> tuple[Path, Path]:
    index, data = tmp / "index", tmp / "data"
    for site in ("O1", "O2"):
        (index / site).mkdir(parents=True)
        (data / site).mkdir(parents=True)
    return index, data


def _add(index: Path, data: Path, site: str, stem: str, fdt: str, nb: int, pnts: int,
         codes: list[str], fdt_bytes: int | None = None, write: bool = True) -> None:
    sp = data / site / f"{stem}.set"
    _set(sp, fdt, nb, pnts, codes)
    fbytes = nb * pnts * 4 if fdt_bytes is None else fdt_bytes
    (index / site / f"{stem}.set").write_text(
        f"/annex/objects/MD5-s{sp.stat().st_size}--{inv.md5_file(sp)}\n")
    (index / site / fdt).write_text(f"/annex/objects/MD5-s{nb * pnts * 4}--{'0' * 32}\n")
    if write:
        (data / site / fdt).write_bytes(b"\0" * fbytes)
    else:
        sp.unlink()


def test_phase3_inventory_statuses(tmp_path: Path) -> None:
    index, data = _tree(tmp_path)
    codes = ["S100", "S200"] * 5 + ["boundary"]
    _add(index, data, "O1", "03_RS_O1_SICL", "03_RS_O1_SICL.fdt", 4, 100, codes)
    _add(index, data, "O1", "04_DP_O1_SICL", "04_DP_O1_SICL.fdt", 4, 100, codes, fdt_bytes=10)
    _add(index, data, "O2", "03_RS_O2_SICL", "03_RS_O2_SICL.fdt", 4, 100, codes, write=False)
    df, _ = inv.run(index, data, tmp_path / "inv.tsv")
    st = {(r.site, r.subject): r.status for r in df.itertuples()}
    assert st == {("O1", "03"): "ok", ("O1", "04"): "size_mismatch", ("O2", "03"): "missing"}
    row = df[(df.site == "O1") & (df.subject == "03")].iloc[0]
    assert row.n_S100 == 5 and row.n_S200 == 5 and row.n_boundary == 1
    assert abs(row.median_isi_s - 3.0) < 1e-9


def test_phase3_inventory_mismatched_pair_18(tmp_path: Path) -> None:
    index, data = _tree(tmp_path)
    codes = ["S100", "S200"] * 5
    # the O1 .set names an O2-stem .fdt stored in O1/ (as on GIN); O2 has its own identical file
    _add(index, data, "O1", "18_ET_O1_SICL", "18_ET_O2_SICL.fdt", 4, 100, codes)
    _add(index, data, "O2", "18_ET_O2_SICL", "18_ET_O2_SICL.fdt", 4, 100, codes)
    df, note = inv.run(index, data, tmp_path / "inv.tsv")
    o1 = df[(df.site == "O1") & (df.subject == "18")].iloc[0]
    assert o1.fdt_file == "18_ET_O2_SICL.fdt"
    assert o1.status == "pairing_excluded" and "identical" in note


def test_phase3_inventory_hash(tmp_path: Path) -> None:
    index, data = _tree(tmp_path)
    _add(index, data, "O1", "05_MC_O1_SICL", "05_MC_O1_SICL.fdt", 2, 10, ["S100"] * 3)
    fdt = data / "O1" / "05_MC_O1_SICL.fdt"
    (index / "O1" / fdt.name).write_text(
        f"/annex/objects/MD5-s{fdt.stat().st_size}--{inv.md5_file(fdt)}\n")
    df, _ = inv.run(index, data, tmp_path / "inv.tsv", do_hash=True)
    assert df.iloc[0].hash_ok and df.iloc[0].status == "ok"
