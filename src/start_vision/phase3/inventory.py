"""Step 1: inventory and download check for GIN DB_123/Phosphenes_occipital (blueprint §5).

The annex index is a content-less clone (``sourcedata/phosphenes_occipital_index/``): each
``.set``/``.fdt`` there is a small pointer file holding the git-annex key
``/annex/objects/MD5-s<bytes>--<md5>`` (MD5 keys, not SHA256E as the blueprint guessed).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

SITES = ("O1", "O2")
_KEY = re.compile(r"MD5E?-s(\d+)--([0-9a-f]{32})")
_NAME = re.compile(r"^(\d{2})_([A-Z]{2})_(O[12])_SICL\.(set|fdt)$")


@dataclass(frozen=True)
class AnnexKey:
    size: int
    md5: str


def read_key(pointer: Path) -> AnnexKey | None:
    """Annex key of a pointer file or symlink; None if the path holds real content."""
    target = str(pointer.readlink()) if pointer.is_symlink() else None
    if target is None:
        if pointer.stat().st_size > 1024:
            return None
        target = pointer.read_text(errors="replace")
    m = _KEY.search(target)
    return AnnexKey(int(m.group(1)), m.group(2)) if m else None


def md5_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@dataclass
class SetHeader:
    datfile: str
    nbchan: int
    pnts: int
    srate: float
    n_S100: int
    n_S200: int
    n_boundary: int
    n_other: int
    median_isi_s: float


def read_header(set_path: Path) -> SetHeader:
    """EEGLAB header fields and event counts, without loading the .fdt."""
    m = scipy.io.loadmat(set_path, squeeze_me=True, struct_as_record=False,
                         variable_names=["datfile", "data", "nbchan", "pnts", "srate", "event"])
    dat = m.get("datfile")
    datfile = str(dat) if isinstance(dat, str) and dat else str(m.get("data", ""))
    ev = np.atleast_1d(m["event"])
    types = [str(e.type).strip() for e in ev]
    lat = np.array([float(e.latency) for e, t in zip(ev, types, strict=True)
                    if t in ("S100", "S200")])
    srate = float(m["srate"])
    isi = float(np.median(np.diff(np.sort(lat))) / srate) if len(lat) > 1 else float("nan")
    return SetHeader(datfile=datfile, nbchan=int(m["nbchan"]), pnts=int(m["pnts"]),
                     srate=srate, n_S100=types.count("S100"), n_S200=types.count("S200"),
                     n_boundary=types.count("boundary"),
                     n_other=sum(t not in ("S100", "S200", "boundary") for t in types),
                     median_isi_s=isi)


def _recordings(index: Path) -> list[tuple[str, str, str]]:
    """(site, subject, initials) from the annex index file names."""
    out = set()
    for site in SITES:
        for f in (index / site).glob("*_SICL.*"):
            m = _NAME.match(f.name)
            if m:
                out.add((site, m.group(1), m.group(2)))
    return sorted(out, key=lambda r: (r[0], r[1]))


def _key(index: Path, site: str, name: str) -> AnnexKey | None:
    p = index / site / name
    return read_key(p) if p.exists() or p.is_symlink() else None


def check_recording(index: Path, data: Path, site: str, sub: str, ini: str,
                    do_hash: bool = False) -> dict[str, object]:
    stem = f"{sub}_{ini}_{site}_SICL"
    set_name = f"{stem}.set"
    # the .fdt this recording uses is whatever its index .set names; default = same stem
    row: dict[str, object] = {"site": site, "subject": sub, "initials": ini,
                              "set_file": set_name}
    kset = _key(index, site, set_name)
    lset = data / site / set_name
    row["set_bytes_key"] = kset.size if kset else None
    row["set_bytes_local"] = lset.stat().st_size if lset.exists() else None
    status = "ok"
    hdr: SetHeader | None = None
    if not lset.exists():
        status = "missing"
    elif kset and lset.stat().st_size != kset.size:
        status = "size_mismatch"
    else:
        hdr = read_header(lset)
    fdt_name = hdr.datfile if hdr and hdr.datfile else f"{stem}.fdt"
    row["fdt_file"] = fdt_name
    kfdt = _key(index, site, fdt_name)
    lfdt = data / site / fdt_name
    row["fdt_bytes_key"] = kfdt.size if kfdt else None
    row["fdt_bytes_local"] = lfdt.stat().st_size if lfdt.exists() else None
    if status == "ok":
        if not lfdt.exists():
            status = "missing"
        elif kfdt and lfdt.stat().st_size != kfdt.size:
            status = "size_mismatch"
    row["datfile_in_set"] = hdr.datfile if hdr else None
    if hdr:
        row.update({k: v for k, v in asdict(hdr).items() if k != "datfile"})
        expect = hdr.nbchan * hdr.pnts * 4
        row["fdt_bytes_expected"] = expect
        if status == "ok" and lfdt.exists() and lfdt.stat().st_size != expect:
            status = "size_mismatch"
        if hdr.n_other:
            status = "unknown_codes" if status == "ok" else status
    hash_ok: bool | None = None
    if do_hash and status == "ok" and kset and kfdt:
        hash_ok = md5_file(lset) == kset.md5 and md5_file(lfdt) == kfdt.md5
        if not hash_ok:
            status = "hash_mismatch"
    row["hash_ok"] = hash_ok
    row["status"] = status
    return row


def pairing_rule_18(df: pd.DataFrame, data: Path) -> tuple[pd.DataFrame, str]:
    """Blueprint §5.4: decide sub-18 site-O1 (its .set points to an O2-named .fdt)."""
    sel = (df["site"] == "O1") & (df["subject"] == "18")
    if not sel.any():
        return df, "sub-18 site-O1 not in the index"
    r = df[sel].iloc[0]
    # GIN's O1 folder holds 18_ET_O1_SICL.set + 18_ET_O2_SICL.fdt (no O1-named .fdt). Decide from
    # the annex keys first: an O1-folder .fdt with the same key as the O2 one is the O2 data.
    index = data.parent / f"{data.name}_index"
    k_o1 = _key(index, "O1", "18_ET_O2_SICL.fdt") if index.exists() else None
    k_o2 = _key(index, "O2", "18_ET_O2_SICL.fdt") if index.exists() else None
    if k_o1 is not None and k_o1 == k_o2:
        exp = r.get("fdt_bytes_expected")
        df.loc[sel, "status"] = "pairing_excluded"
        return df, (f"sub-18 O1: the only .fdt in O1/ (18_ET_O2_SICL.fdt) has the same annex key "
                    f"as O2/18_ET_O2_SICL.fdt ({k_o1.size} B; the O1 header needs {exp} B) "
                    f"-> O1 data not on GIN, sub-18 site-O1 excluded")
    if r["status"] != "ok":
        return df, f"sub-18 site-O1 not checkable yet (status {r['status']})"
    o2 = df[(df["site"] == "O2") & (df["subject"] == "18")]
    o2_fdt = data / "O2" / str(o2.iloc[0]["fdt_file"]) if len(o2) else None
    o1_fdt = data / "O1" / str(r["fdt_file"])
    note = f"sub-18 O1 .set -> {r['fdt_file']}"
    if o2_fdt is not None and o2_fdt.exists() and o1_fdt.exists():
        same = o1_fdt.stat().st_size == o2_fdt.stat().st_size and (
            md5_file(o1_fdt) == md5_file(o2_fdt))
        if same:
            df.loc[sel, "status"] = "pairing_excluded"
            return df, note + ": identical to the O2 data file -> excluded"
        corr = _oz_corr(data / "O1" / str(r["set_file"]),
                        data / "O2" / str(o2.iloc[0]["set_file"]))
        if corr >= 0.99:
            df.loc[sel, "status"] = "pairing_excluded"
            return df, note + f": Oz first 60 s r = {corr:.3f} with O2 -> excluded"
        note += f"; Oz first 60 s r with O2 = {corr:.3f}"
    isi = float(r.get("median_isi_s", np.nan))
    if not (2.0 <= isi <= 4.0):
        df.loc[sel, "status"] = "pairing_excluded"
        return df, note + f": implausible pulse interval {isi:.2f} s -> excluded"
    df.loc[sel, "status"] = "ok_pairing_deviation"
    return df, note + ": sizes and events fit -> used, recorded as a deviation"


def _oz_corr(set_a: Path, set_b: Path) -> float:
    import mne

    xs = []
    for p in (set_a, set_b):
        raw = mne.io.read_raw_eeglab(p, preload=False, verbose="error")
        xs.append(raw.get_data(picks=["Oz"], start=0, stop=int(60 * raw.info["sfreq"]))[0])
    n = min(len(xs[0]), len(xs[1]))
    return float(np.corrcoef(xs[0][:n], xs[1][:n])[0, 1])


def run(index: Path, data: Path, out: Path, do_hash: bool = False) -> tuple[pd.DataFrame, str]:
    recs = _recordings(index)
    if not recs:
        raise FileNotFoundError(f"no annex index under {index}: git clone "
                                "https://gin.g-node.org/DB_123/Phosphenes_occipital.git there")
    rows = [check_recording(index, data, s, sub, ini, do_hash) for s, sub, ini in recs]
    df = pd.DataFrame(rows)
    df, note = pairing_rule_18(df, data)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    return df, note
