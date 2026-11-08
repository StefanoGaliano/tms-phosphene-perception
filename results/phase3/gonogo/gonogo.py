"""Phase 3 quick go/no-go: frozen tms-robust detector on GIN phosphene data (rough cleaning)."""
import sys, json, numpy as np, mne
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
root = Path(sys.argv[1]); sys.path.insert(0, str(root / "src"))
from start_vision.decoding.detector import Detector
from start_vision.phosphenes import complete_contract_channels
mne.set_log_level("ERROR")
det = Detector.load(root / "detectors/sv-perception-tms-robust-v1.0.0"); C = det.contract
MOTOR = ["FC5","FC3","FC1","C5","C3","C1","CP5","CP3","CP1","FC6","FC4","FC2","C6","C4","C2","CP6","CP4","CP2"]
f = Path(sys.argv[2])
raw = mne.io.read_raw_eeglab(f, preload=True)
raw.pick([c for c in raw.ch_names if c not in ("HEOGL","HEOGR","VEOGUP","VEOGDOWN","LM")])
ev, eid = mne.events_from_annotations(raw)
ev = ev[np.isin(ev[:, 2], [eid["S100"], eid["S200"]])]
sf = raw.info["sfreq"]; a, b = int(-0.002 * sf), int(0.015 * sf)
d = raw._data
for s in ev[:, 0]:                       # linear interpolation over the pulse, -2..15 ms
    i0, i1 = s + a - 1, s + b + 1
    w = np.linspace(0, 1, i1 - i0 + 1)[None, 1:-1]
    d[:, i0 + 1:i1] = d[:, [i0]] * (1 - w) + d[:, [i1]] * w
raw.filter(0.1, 40, method="iir", iir_params=dict(order=4, ftype="butter"))
raw, ev = raw.resample(C["sfreq"], events=ev)
raw, miss = complete_contract_channels(raw, C["channels"])
raw.set_eeg_reference("average")
y = (ev[:, 2] == eid["S100"]).astype(int)
def score(r):
    ep = mne.Epochs(r, ev, tmin=C["epoch_window_s"][0], tmax=C["epoch_window_s"][1], baseline=None, preload=True)
    assert len(ep) == len(ev)
    return det.from_epochs(ep), ep
p, ep = score(raw)
rm = raw.copy(); rm.info["bads"] = MOTOR; rm.interpolate_bads(reset_bads=True)
p_nm, _ = score(rm)
t = ep.times; m = (t >= 0.15) & (t <= 0.6)
x = ep.get_data(picks=["C3", "C4"]); lat = (x[:, 0] - x[:, 1])[:, m].mean(1)
rho = np.nanmean([spearmanr(p[y == k], lat[y == k])[0] for k in (0, 1)])
res = dict(file=f.name, n_seen=int(y.sum()), n_unseen=int((1 - y).sum()),
           auc=roc_auc_score(y, p), auc_no_motor=roc_auc_score(y, p_nm),
           auc_c3c4_alone=roc_auc_score(y, -lat), rho_p_vs_c3c4_within=float(rho),
           mean_p_seen=float(p[y == 1].mean()), mean_p_unseen=float(p[y == 0].mean()))
print(json.dumps(res))
