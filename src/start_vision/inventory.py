"""Subjects x tasks of ds006171, from the public OpenNeuro S3 mirror or from disk.

The S3 ListObjectsV2 API returns at most 1000 keys per page, so the listing follows
``continuation-token`` until ``IsTruncated`` is false (PHASE1_BLUEPRINT §7).
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

S3 = "https://s3.amazonaws.com/openneuro.org"
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_EEG = re.compile(r"sub-(?P<sub>\d+)/eeg/sub-\d+_task-(?P<task>[a-z]+)_eeg\.bdf$")
TASKS = ("nocue", "noninformative", "informative")


def iter_s3_keys(prefix: str) -> Iterator[tuple[str, int]]:
    """Yield (key, size in bytes) for every object under ``prefix``, across all pages."""
    token: str | None = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        url = f"{S3}?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            root = ET.fromstring(resp.read())
        for item in root.findall("s3:Contents", _NS):
            key = item.findtext("s3:Key", default="", namespaces=_NS)
            size = int(item.findtext("s3:Size", default="0", namespaces=_NS))
            yield key, size
        if root.findtext("s3:IsTruncated", namespaces=_NS) != "true":
            return
        token = root.findtext("s3:NextContinuationToken", namespaces=_NS)


def _table(rows: list[tuple[str, int]]) -> pd.DataFrame:
    recs = []
    for key, size in rows:
        m = _EEG.search(key)
        if m:
            recs.append({"subject": m["sub"], "task": m["task"], "bytes": size})
    df = pd.DataFrame(recs, columns=["subject", "task", "bytes"])
    wide = df.pivot_table(index="subject", columns="task", values="bytes", aggfunc="sum")
    wide = wide.reindex(columns=list(TASKS))
    out = pd.DataFrame(index=wide.index)
    for t in TASKS:
        out[t] = wide[t].notna()
    out["n_tasks"] = out[list(TASKS)].sum(axis=1)
    out["bytes"] = wide.fillna(0).sum(axis=1).astype("int64")
    return out.reset_index().sort_values("subject").reset_index(drop=True)


def list_remote(dataset: str = "ds006171") -> pd.DataFrame:
    """One row per subject: task presence (bool per task), number of tasks, BDF bytes."""
    return _table(list(iter_s3_keys(f"{dataset}/")))


def list_local(root: Path) -> pd.DataFrame:
    """Same table for BDFs already downloaded under ``root`` (e.g. sourcedata/ds006171)."""
    rows = [(str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*_eeg.bdf")]
    return _table(rows)
