"""inventory._table: task presence and sizes from S3-style keys."""

from __future__ import annotations

from start_vision.inventory import _table


def test_ds006171_missing_informative_task() -> None:
    rows = [
        ("ds006171/sub-01/eeg/sub-01_task-nocue_eeg.bdf", 10),
        ("ds006171/sub-01/eeg/sub-01_task-noninformative_eeg.bdf", 20),
        ("ds006171/sub-01/eeg/sub-01_task-nocue_events.tsv", 1),
        ("ds006171/sub-02/eeg/sub-02_task-informative_eeg.bdf", 5),
    ]
    df = _table(rows).set_index("subject")
    assert not df.loc["01", "informative"] and df.loc["01", "n_tasks"] == 2
    assert df.loc["01", "bytes"] == 30
    assert df.loc["02", "informative"] and df.loc["02", "n_tasks"] == 1
