from tests.synthetic.tms import ABSENT, contract_channels, simulate


def test_tms_synthetic_shapes() -> None:
    s = simulate(n_pulses=20, sfreq=1000.0, seed=1)
    assert len(s.events) == 20 and set(s.labels) <= {0, 1}
    assert s.raw.info["sfreq"] == 1000.0
    assert len([c for c in s.raw.ch_names if c not in ("HEOGL", "HEOGR", "VEOGUP", "VEOGDOWN",
                                                       "LM")]) == len(contract_channels()) - 5
    assert not set(ABSENT) & set(s.raw.ch_names)
    assert s.meta["spike_end_ms"] == 5.0
    assert [a["description"] for a in s.raw.annotations][:3] == [
        "S100" if lab else "S200" for lab in s.labels[:3]]
