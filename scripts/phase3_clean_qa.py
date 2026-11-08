"""Phase 3 step 9 QA: figures F1/F2 and the ongoing-muscle measure for Q5 (option C).

Run from Start_Vision/ after `sv-p3-clean -s 03 -s 04 --site O1`:
    python -I scripts/phase3_clean_qa.py
Writes results/phase3/qa/ (F1_*.png, F2_artifact_end.png, muscle_q5.tsv).

Muscle measure: ongoing (not pulse-locked) 20-35 Hz power at the lateral edge channels
FT7 T7 TP7 FT8 T8 TP8, where scalp/jaw muscle shows, with the same simple processing on both
datasets (1 Hz high-pass, average reference over the recorded EEG, 40 Hz low-pass, 128 Hz):
GIN = the 0.9 s before each pulse (-1.0..-0.1 s), ds006171 = 2-s segments over the first 600 s
of the nocue recording (training data). Compared as log10 power ratio GIN / ds006171.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from start_vision.phase3 import config, stages  # noqa: E402

EDGE = ["FT7", "T7", "TP7", "FT8", "T8", "TP8"]
BAND = (20.0, 35.0)
OUT = ROOT / "results" / "phase3" / "qa"
mne.set_log_level("ERROR")


def _band_power(x: np.ndarray, sf: float) -> float:
    f, p = _welch(x, sf)
    m = (f >= BAND[0]) & (f <= BAND[1])
    return float(np.median(p[..., m].mean(axis=-1)))


def _welch(x: np.ndarray, sf: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    return welch(x, fs=sf, nperseg=min(x.shape[-1], int(sf)), axis=-1)


def _simple(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.filter(1.0, 40.0, picks="eeg")
    raw.set_eeg_reference("average", projection=False)
    return raw.resample(128.0)


def gin_muscle(cfg: config.Phase3Config, sub: str, site: str) -> float:
    rec = stages.clean.read_recording(stages.set_path(cfg, ROOT, sub, site), cfg)
    raw = rec.raw.pick("eeg")
    raw.rename_channels({c: c.capitalize() for c in raw.ch_names if c in ("FP1", "FP2")})
    sf0 = raw.info["sfreq"]
    raw = _simple(raw)
    p = np.round(rec.pulses * 128.0 / sf0).astype(int)
    x = raw.get_data(picks=EDGE) * 1e6
    seg = np.stack([x[:, q - 128:q - 13] for q in p if q - 128 >= 0])  # -1.0..-0.1 s
    return _band_power(seg, 128.0)


def ds006171_muscle(sub: str) -> float:
    tpl = yaml.safe_load((ROOT / "configs" / "template_ds006171.yaml").read_text())
    inv = {v: k for k, v in tpl["channels"]["rename"].items()}
    bdf = ROOT / "sourcedata" / "ds006171" / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-nocue_eeg.bdf"
    raw = mne.io.read_raw_bdf(bdf, preload=False)
    eeg = [c for c in raw.ch_names if c[:1] in "ABCD" and c[1:].isdigit()]
    raw.pick(eeg).crop(0, 600).load_data()
    raw = _simple(raw)
    x = raw.get_data(picks=[inv[c] for c in EDGE]) * 1e6
    n = x.shape[1] // 256
    seg = x[:, : n * 256].reshape(len(EDGE), n, 256).transpose(1, 0, 2)
    return _band_power(seg, 128.0)


def figures(cfg: config.Phase3Config, rid: str) -> dict[str, object]:
    out = stages.out_paths(cfg, ROOT, "primary", rid)
    erp = np.load(out["erp"])
    qa = json.loads(out["qa"].read_text())
    steps = [s for s in ("raw", "cut", "decay_ica", "ocular_ica") if f"{s}_avg" in erp]
    fig, axes = plt.subplots(2, len(steps), figsize=(4 * len(steps), 6), sharex=True)
    for j, s in enumerate(steps):
        t = erp[f"{s}_times"] * 1000
        a = erp[f"{s}_avg"] * 1e6
        gmfp = a.std(axis=0)
        axes[0, j].plot(t, a.T, lw=0.4, color="0.4")
        axes[0, j].set_title(s)
        lim = np.percentile(np.abs(a[:, t > 30]), 99.5) * 1.5 if s != "raw" else None
        if lim:
            axes[0, j].set_ylim(-lim, lim)
        axes[1, j].plot(t, gmfp, color="C0")
        axes[1, j].axvspan(150, 600, color="C1", alpha=0.15, label="detector window")
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("ms from pulse")
    axes[0, 0].set_ylabel("µV (butterfly)")
    axes[1, 0].set_ylabel("GMFP µV (log)")
    axes[1, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"{rid}: pulse-locked average of all pulses after each cleaning step (F1)")
    fig.tight_layout()
    fig.savefig(OUT / f"F1_{rid}.png", dpi=110)
    plt.close(fig)
    t = erp["ocular_ica_times"]
    g = (erp["ocular_ica_avg"] * 1e6).std(axis=0)
    pre = g[t < -0.01]
    return {"recording": rid, "artifact_end_ms": qa["artifact_end_ms"],
            "cut_ms": qa["cut_ms"], "artifact_end_after_ica_ms": qa.get("artifact_end_after_ica_ms"),
            "decay_ica_excluded": len(qa["decay_ica"]["excluded"]),
            "decay_ica_n": qa["decay_ica"]["n_components"],
            "ocular_ica_excluded": len(qa["ocular_ica"]["excluded"]),
            "bad_channels": ",".join(qa["bad_channels"]) or "-",
            "kept": qa["rejection"]["kept"], "rej_ptp": qa["rejection"]["ptp"],
            "rej_heog": qa["rejection"]["heog_step"],
            "gmfp_150_600_over_pre": float(g[(t >= 0.15) & (t <= 0.6)].mean() / pre.mean()),
            "gmfp_50_150_over_pre": float(g[(t >= 0.05) & (t < 0.15)].mean() / pre.mean()),
            "time_total_s": round(qa["timings_s"]["total"]),
            }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load(ROOT / "configs" / "phase3.yaml")
    rows = [figures(cfg, stages.rec_id(s, "O1")) for s in ("03", "04")]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "clean_qa.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(df["recording"], df["artifact_end_ms"], label="before")
    ax.bar(df["recording"], df["artifact_end_after_ica_ms"], width=0.4, label="after decay ICA")
    ax.set_ylabel("artifact end (ms)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "F2_artifact_end.png", dpi=110)
    mus = [{"data": f"GIN sub-{s} O1", "power_uv2_hz": gin_muscle(cfg, s, "O1")} for s in ("03", "04")]
    mus += [{"data": f"ds006171 sub-{s} nocue", "power_uv2_hz": ds006171_muscle(s)}
            for s in ("03", "04", "05", "06")]
    m = pd.DataFrame(mus)
    ref = m[m["data"].str.startswith("ds")]["power_uv2_hz"].median()
    m["log10_ratio_vs_ds006171_median"] = np.log10(m["power_uv2_hz"] / ref)
    m.to_csv(OUT / "muscle_q5.tsv", sep="\t", index=False)
    print(df.to_string(index=False))
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
