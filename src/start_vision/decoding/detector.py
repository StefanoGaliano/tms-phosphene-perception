"""The exported perception detector (PHASE2 §9; build step 10).

Recipe, per subject (no labels needed at use time):
1. epochs time-locked to stimulus/pulse onset, contract channels, contract sfreq, already
   filtered as the contract says (Phase 2 data: 40 Hz low-pass, 128 Hz; ``data.prepare``);
2. per-trial baseline on ``baseline_s`` (prestimulus only, ending >= 10 ms before onset, A10);
3. per-channel z-score with the subject's own prestimulus statistics (``norm_window_s``);
4. window features (50 ms bins in ``window_s``) -> StandardScaler -> balanced LR;
5. Platt calibration fitted on LOSO held-out scores, class-balanced so the output is
   P(seen) at equal priors;
6. decision: percept present if P(seen) >= threshold (rule recorded in the contract).

Files in ``detectors/sv-perception-<variant>-v<semver>/``: model.skops, contract.json,
channels.tsv, patterns.npz, calibration.tsv, performance.json, README.md, MANIFEST.sha256.
``Detector.load`` verifies the manifest and loads skops with an explicit trusted-type list.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from start_vision import __version__ as sv_version
from start_vision.decoding import models
from start_vision.decoding.config import DecodingConfig
from start_vision.decoding.data import TrialSet
from start_vision.decoding.features import baseline, prestim_normalize

DETECTOR_BASELINE_S = (-0.2, -0.01)  # A10: prestimulus only, ends >= 10 ms before onset
FILES = ("model.skops", "contract.json", "channels.tsv", "patterns.npz", "calibration.tsv",
         "performance.json", "README.md")
TRUSTED = {
    "start_vision.decoding.models.WindowFeatures",
    "mne.decoding.base.LinearModel",
}


class ContractError(ValueError):
    """Input does not match the detector contract."""


def prepare_subject(x: np.ndarray, times: np.ndarray, baseline_s: tuple[float, float],
                    norm_window_s: tuple[float, float]) -> np.ndarray:
    """Steps 2-3 of the recipe for one subject's trials (label-free)."""
    return prestim_normalize(baseline(x, times, baseline_s), times, norm_window_s)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold on calibrated P(seen) that maximizes balanced accuracy (ties: closest to 0.5)."""
    cand = np.unique(np.concatenate([p, [0.5]]))
    acc = np.array([balanced_accuracy_score(y, p >= c) for c in cand])
    best = cand[acc >= acc.max() - 1e-12]
    return float(best[np.argmin(np.abs(best - 0.5))])


def threshold_for_fa(p_catch: np.ndarray, target_fa: float) -> float:
    """Threshold whose false-alarm rate on catch trials is closest to ``target_fa``."""
    cand = np.unique(np.concatenate([p_catch, [1.0]]))
    fa = np.array([(p_catch >= c).mean() for c in cand])
    return float(cand[np.argmin(np.abs(fa - target_fa))])


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = [{"bin_low": edges[k], "bin_high": edges[k + 1], "n": int((b == k).sum()),
             "mean_p": float(p[b == k].mean()) if (b == k).any() else np.nan,
             "observed": float(y[b == k].mean()) if (b == k).any() else np.nan}
            for k in range(n_bins)]
    return pd.DataFrame(rows)


@dataclass
class Detector:
    pipeline: Any
    calibrator: LogisticRegression
    contract: dict[str, Any]
    performance: dict[str, Any] = field(default_factory=dict)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    patterns: np.ndarray = field(default_factory=lambda: np.zeros(0))
    held_out: dict[str, np.ndarray] = field(default_factory=dict)  # LOSO P(seen), for F10
    channel_table: pd.DataFrame | None = None  # target, cap, distance_mm, exact (+ x/y/z)

    # ---------- training ----------
    @classmethod
    def fit(cls, sets: Sequence[TrialSet], cfg: DecodingConfig, window_s: tuple[float, float],
            variant: str, version: str = "1.0.0", C: float | None = None,
            trained_on: dict[str, Any] | None = None,
            threshold_rule: str = "max_balanced_accuracy",
            catch: Sequence[TrialSet | None] | None = None,
            channel_table: pd.DataFrame | None = None) -> Detector:
        """LOSO held-out scores -> calibrator + threshold; final model on all subjects.

        ``catch``: per subject (same order as ``sets``) its stimulus-absent trials or None.
        They are scored by that subject's LOSO fold model (never trained on), and take part in
        the subject's label-free normalization, as at use time ("all trials of the subject").
        """
        if window_s[0] < DETECTOR_BASELINE_S[1]:
            raise ValueError("feature window must start after the baseline")
        ch = sets[0].ch_names
        if any(s.ch_names != ch for s in sets):
            raise ValueError("all training sets need the same channels (harmonize first)")
        catch = list(catch) if catch is not None else [None] * len(sets)
        times = sets[0].times
        feat = models.WindowFeatures(times, window_s, cfg.feature_bin_s)
        fs, fc = [], []
        for s, c in zip(sets, catch, strict=True):
            x = s.X if c is None or not len(c) else np.concatenate([s.X, c.X])
            f = feat.transform(prepare_subject(x, times, DETECTOR_BASELINE_S,
                                               cfg.norm_window_s))
            fs.append(f[:len(s)])
            fc.append(f[len(s):] if c is not None and len(c) else None)
        ys = [s.labels("seen") for s in sets]

        def make() -> Any:  # window features already computed: scaler + LR
            pipe = models.window_detector(cfg, times, window_s, C)
            pipe.steps[0] = (pipe.steps[0][0], "passthrough")
            return pipe

        held, per_sub, catch_s, catch_y = [], [], [], []
        for i in range(len(sets)):
            ftr = np.concatenate([f for j, f in enumerate(fs) if j != i])
            ytr = np.concatenate([y for j, y in enumerate(ys) if j != i])
            m = make().fit(ftr, ytr)
            s = m.decision_function(fs[i])
            held.append(s)
            sub = str(sets[i].meta["subject"].iloc[0])
            per_sub.append({"subject": sub, "n": len(s), "auc": float(roc_auc_score(ys[i], s))})
            if fc[i] is not None:
                catch_s.append(m.decision_function(fc[i]))
                catch_y.append(catch[i].labels("seen"))  # type: ignore[union-attr]
        s_all, y_all = np.concatenate(held), np.concatenate(ys)
        cal = LogisticRegression(class_weight="balanced").fit(s_all[:, None], y_all)
        p_all = cal.predict_proba(s_all[:, None])[:, 1]
        theta = best_threshold(y_all, p_all)
        final = make().fit(np.concatenate(fs), y_all)
        final.steps[0] = (final.steps[0][0], feat)  # stateless: same model, applies to epochs
        # Haufe patterns per feature (§9): undo the scaler, then channels x 50 ms bins
        lin, scaler = final[-1], final[1]
        pats = (np.ravel(lin.patterns_) * scaler.scale_).reshape(len(ch), -1)
        aucs = np.array([r["auc"] for r in per_sub])
        perf: dict[str, Any] = {
            "loso_auc_per_subject": per_sub,
            "loso_auc_mean": float(aucs.mean()),
            "threshold": theta,
            "balanced_accuracy_at_threshold": float(balanced_accuracy_score(y_all,
                                                                            p_all >= theta)),
            "brier": float(brier_score_loss(y_all, p_all)),
            "n_trials": len(y_all), "n_subjects": len(sets),
        }
        held_p = {"seen": p_all[y_all == 1], "unseen": p_all[y_all == 0]}
        if catch_s:
            pc = cal.predict_proba(np.concatenate(catch_s)[:, None])[:, 1]
            yc = np.concatenate(catch_y).astype(bool)
            behav_fa = float(yc.mean())
            th_fa = threshold_for_fa(pc, behav_fa)
            perf["catch"] = {
                "n_catch": len(pc), "n_false_alarms": int(yc.sum()),
                "behavioural_fa_rate": behav_fa,
                "detector_fa_rate_at_threshold": float((pc >= theta).mean()),
                "threshold_at_behavioural_fa": th_fa,
                "hit_rate_at_behavioural_fa_threshold": float((p_all[y_all == 1] >= th_fa)
                                                              .mean()),
                "auc_fa_vs_cr_pooled": (float(roc_auc_score(yc, pc))
                                        if 0 < yc.sum() < len(yc) else None),
            }
            held_p |= {"catch FA": pc[yc], "catch CR": pc[~yc]}
        contract = {
            "detector_version": version, "variant": variant,
            "trained_on": trained_on or {},
            "start_vision_version": sv_version, "eegunify_version": _version("eegunify"),
            "mne_version": mne.__version__, "sklearn_version": sklearn.__version__,
            "channels": list(ch), "reference": "average over the recording's EEG channels",
            "sfreq": float(sets[0].sfreq),
            "filter": {"l_freq": 0.1, "h_freq": cfg.lowpass_hz, "method": "iir zero-phase"},
            "epoch_window_s": [float(times[0]), float(times[-1])],
            "baseline_window_s": list(DETECTOR_BASELINE_S),
            "normalization": {"type": "per-subject prestimulus z-score",
                              "window_s": list(cfg.norm_window_s)},
            "feature_window_s": list(window_s), "feature_bin_s": cfg.feature_bin_s,
            "C": cfg.C if C is None else C,
            "threshold": theta, "threshold_rule": threshold_rule,
            "alternative_rules": {
                "behavioural_fa": perf.get("catch", {}).get("threshold_at_behavioural_fa"),
                "subject_adaptive": "theta_subject = median P(seen) of that subject's trials"},
            "calibration": "Platt (class-balanced) on LOSO held-out scores",
            "min_trials_for_normalization": cfg.min_trials_for_normalization,
            "max_missing_channels": cfg.max_missing_channels,
            "ica_ocular_removed": True, "seed": cfg.seed,
            "stage_key": "",
        }
        det = cls(final, cal, contract, perf, reliability(y_all, p_all), pats)
        det.held_out = held_p
        det.channel_table = channel_table
        return det

    # ---------- use ----------
    def check(self, ch_names: Sequence[str], sfreq: float, times: np.ndarray) -> None:
        c = self.contract
        if abs(sfreq - c["sfreq"]) > 1e-6:
            raise ContractError(f"sfreq {sfreq} != contract {c['sfreq']} (resample upstream)")
        if list(ch_names) != c["channels"]:
            missing = [x for x in c["channels"] if x not in ch_names]
            raise ContractError(f"channels differ from the contract (missing {missing[:8]}"
                                f"{'...' if len(missing) > 8 else ''}); use from_epochs to "
                                "reorder or interpolate")
        lo, hi = c["epoch_window_s"]
        if times[0] > lo + 1e-6 or times[-1] < hi - 1e-6:
            raise ContractError(f"epochs {times[0]:.3f}..{times[-1]:.3f} s do not cover the "
                                f"contract window {lo}..{hi} s")

    def decision_scores(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        c = self.contract
        if len(x) < c["min_trials_for_normalization"]:
            raise ContractError(f"{len(x)} trials < {c['min_trials_for_normalization']} needed "
                                "for the per-subject normalization")
        lo, hi = c["epoch_window_s"]
        m = (times >= lo - 1e-9) & (times <= hi + 1e-9)
        xp = prepare_subject(x[..., m], times[m], tuple(c["baseline_window_s"]),
                             tuple(c["normalization"]["window_s"]))
        return np.asarray(self.pipeline.decision_function(xp))

    def predict_proba(self, ts: TrialSet) -> np.ndarray:
        """Calibrated P(seen) per trial for one subject's trials (µV, contract layout)."""
        self.check(ts.ch_names, ts.sfreq, ts.times)
        s = self.decision_scores(ts.X, ts.times)
        return np.asarray(self.calibrator.predict_proba(s[:, None])[:, 1])

    def predict(self, ts: TrialSet, threshold: float | None = None) -> np.ndarray:
        return self.predict_proba(ts) >= (self.contract["threshold"] if threshold is None
                                          else threshold)

    def from_epochs(self, epochs: mne.BaseEpochs, allow_resample: bool = False) -> np.ndarray:
        """Calibrated P(seen) from MNE epochs: picks/reorders contract channels, interpolates
        at most ``max_missing_channels`` if positions exist, resamples only if allowed."""
        c = self.contract
        e = epochs.copy().load_data()
        if abs(e.info["sfreq"] - c["sfreq"]) > 1e-6:
            if not allow_resample:
                raise ContractError(f"sfreq {e.info['sfreq']} != {c['sfreq']}; pass "
                                    "allow_resample=True to resample")
            e.resample(c["sfreq"], verbose="error")
        missing = [x for x in c["channels"] if x not in e.ch_names]
        if missing:
            if len(missing) > c["max_missing_channels"]:
                raise ContractError(f"{len(missing)} contract channels missing: {missing}")
            mont = e.get_montage()
            if mont is None:
                raise ContractError("missing channels and no montage to interpolate them")
            from start_vision.decoding.channels import positions

            pos = dict(mont.get_positions()["ch_pos"])
            pos.update(positions(missing))  # template positions for the absent channels only
            zeros = mne.EpochsArray(np.zeros((len(e), len(missing), len(e.times))),
                                    mne.create_info(missing, e.info["sfreq"], "eeg"),
                                    tmin=e.tmin, verbose="error")
            e.add_channels([zeros], force_update_info=True)
            e.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head"),
                          on_missing="ignore", verbose="error")
            e.info["bads"] = missing
            e.interpolate_bads(reset_bads=True, verbose="error")
        e.pick(c["channels"])
        x = (e.get_data() * 1e6).astype(np.float32)
        self.check(e.ch_names, e.info["sfreq"], e.times)
        s = self.decision_scores(x, e.times)
        return np.asarray(self.calibrator.predict_proba(s[:, None])[:, 1])

    # ---------- files ----------
    def save(self, folder: Path, overwrite: bool = False) -> Path:
        import skops.io as sio

        if folder.exists() and any(folder.iterdir()) and not overwrite:
            raise FileExistsError(f"{folder} exists; exported detectors are immutable")
        folder.mkdir(parents=True, exist_ok=True)
        sio.dump({"pipeline": self.pipeline, "calibrator": self.calibrator},
                 folder / "model.skops")
        (folder / "contract.json").write_text(json.dumps(self.contract, indent=2) + "\n")
        self._channels_tsv().to_csv(folder / "channels.tsv", sep="\t", index=False)
        np.savez(folder / "patterns.npz", patterns=self.patterns)
        self.calibration.to_csv(folder / "calibration.tsv", sep="\t", index=False)
        (folder / "performance.json").write_text(json.dumps(self.performance, indent=2) + "\n")
        (folder / "README.md").write_text(_readme(self.contract))
        lines = [f"{_sha256(folder / f)}  {f}" for f in FILES]
        (folder / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
        return folder

    def _channels_tsv(self) -> pd.DataFrame:
        """name, 10-10 label (= name), source cap electrode, distance, template x/y/z (m)."""
        from start_vision.decoding.channels import positions

        names = self.contract["channels"]
        t = (self.channel_table.set_index("target").loc[names].reset_index()
             if self.channel_table is not None else
             pd.DataFrame({"target": names, "cap": names, "distance_mm": 0.0, "exact": True}))
        t = t.rename(columns={"target": "name", "cap": "source_electrode"})
        t.insert(1, "label_10_10", t["name"])
        if "x" in t:
            return t
        try:
            pos = positions(list(names))
            xyz = np.array([pos[n] for n in names])
            t[["x", "y", "z"]] = xyz
        except Exception:  # names outside the template (e.g. synthetic) get no positions
            pass
        return t

    @classmethod
    def load(cls, folder: Path) -> Detector:
        import skops.io as sio

        for line in (folder / "MANIFEST.sha256").read_text().splitlines():
            digest, name = line.split("  ", 1)
            if _sha256(folder / name) != digest:
                raise ContractError(f"{name} does not match MANIFEST.sha256")
        path = folder / "model.skops"
        untrusted = set(sio.get_untrusted_types(file=path))
        extra = untrusted - TRUSTED
        if extra:
            raise ContractError(f"model.skops contains unexpected types: {sorted(extra)}")
        obj = sio.load(path, trusted=sorted(untrusted))
        pats = np.load(folder / "patterns.npz")["patterns"]
        return cls(obj["pipeline"], obj["calibrator"],
                   json.loads((folder / "contract.json").read_text()),
                   json.loads((folder / "performance.json").read_text()),
                   pd.read_csv(folder / "calibration.tsv", sep="\t"), pats)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _version(pkg: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def _readme(c: dict[str, Any]) -> str:
    return f"""# sv-perception-{c['variant']}-v{c['detector_version']}

Seen-vs-unseen EEG detector (Start_Vision Phase 2). Immutable: a new recipe is a new major version.

```python
from start_vision.decoding.detector import Detector
det = Detector.load(Path("detectors/sv-perception-{c['variant']}-v{c['detector_version']}"))
p_seen = det.from_epochs(epochs)        # MNE epochs of ONE subject, onset at 0 s
present = p_seen >= det.contract["threshold"]
```

Input: {len(c['channels'])} channels (channels.tsv), {c['sfreq']:g} Hz, filtered
{c['filter']['l_freq']}-{c['filter']['h_freq']} Hz, average reference, window
{c['epoch_window_s'][0]:.3f}..{c['epoch_window_s'][1]:.3f} s, at least
{c['min_trials_for_normalization']} trials (per-subject normalization).
Features {c['feature_window_s'][0]}-{c['feature_window_s'][1]} s; baseline
{c['baseline_window_s']} s. Threshold {c['threshold']:.3f} ({c['threshold_rule']}).
Details: contract.json; held-out performance: performance.json.
"""
