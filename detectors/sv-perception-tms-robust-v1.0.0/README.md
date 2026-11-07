# sv-perception-tms-robust-v1.0.0

Seen-vs-unseen EEG detector (Start_Vision Phase 2). Immutable: a new recipe is a new major version.

```python
from start_vision.decoding.detector import Detector
det = Detector.load(Path("detectors/sv-perception-tms-robust-v1.0.0"))
p_seen = det.from_epochs(epochs)        # MNE epochs of ONE subject, onset at 0 s
present = p_seen >= det.contract["threshold"]
```

Input: 64 channels (channels.tsv), 128 Hz, filtered
0.1-40.0 Hz, average reference, window
-0.500..0.992 s, at least
40 trials (per-subject normalization).
Features 0.15-0.6 s; baseline
[-0.2, -0.01] s. Threshold 0.521 (max_balanced_accuracy).
Details: contract.json; held-out performance: performance.json.
