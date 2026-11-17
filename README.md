# tms-phosphene-perception

Code, pre-registration, frozen detectors and results for the paper *TMS-evoked phosphenes carry the
EEG signature of natural visual awareness* (S. Galiano). An EEG detector of natural near-threshold
seeing is trained on OpenNeuro ds006171, frozen, checked on an independent natural-vision dataset
(Eklund & Wiens, 2018) and applied without retraining to TMS-evoked phosphenes over occipital
(GIN `DB_123/Phosphenes_occipital`) and parietal cortex (GIN `DB_123/IPS_parietal_phosphenes`).

This repository is the paper's code companion, split out of the working project with its history
filtered to the files below. Commit dates and all tags are preserved, so every pre-registration can
be compared with the analyses that followed it:

| Tag | Meaning |
|---|---|
| `phase1-v1.0` | Step 1: VAN and late positivity, seen minus unseen (ds006171) |
| `phase2-detectors-v1.0.0` | Step 2: exported frozen detectors (`detectors/`) |
| `phase3-prereg` | Step 3 pre-registration (`docs/PHASE3_PREREG.md`), before any phosphene data were scored |
| `phase3-v1.0` … `phase3-v1.2` | Step 3 results |
| `questions-prereg`, `questions-prereg-amend1` … `-amend11` | Follow-up questions pre-registration (`docs/QUESTIONS_PREREG.md`) and its amendments, each tagged before the analysis it describes was run |

## Layout

- `docs/`: the pre-registrations (`PHASE3_PREREG.md`, `QUESTIONS_PREREG.md`), the reference values
  taken from the literature (`PHASE1_REFERENCES.md`, `PHASE3_REFERENCES.md`) and the dataset checks
  (`DATASETS.md`).
- `src/start_vision/`: the package (staging, natural-vision analysis, `decoding/`, `phase3/`, `questions/`).
- `configs/`: preprocessing, component windows, decoding and TMS settings.
- `detectors/`: frozen detectors `sv-perception-{tms-robust,full}-v1.0.0` (immutable; see their
  README and `MANIFEST.sha256`).
- `scripts/`: the analyses reported in the paper (S7/R4b, Q39-Q47, the parietal dataset Q41).
- `results/`, `reports/`: tables and reports behind every number in the paper.
- `tests/`: synthetic tests (`pytest`; real-data tests are marked `real` and skipped without data).

## Install and run

```bash
pip install -e .[dev]          # needs eegunify 0.9.3: github.com/StefanoGaliano/eeg-erp-pipeline
pytest
```

Data are not included: ds006171 (CC0, doi:10.18112/openneuro.ds006171.v1.0.0) from OpenNeuro; the
phosphene datasets (CC0) from GIN (`gin.g-node.org/DB_123`); the independent natural-vision data
(CC BY 4.0, doi:10.17045/sthlmuni.5418967) from figshare. Command-line entry points: `sv-inventory`,
`sv-stage`, `sv-configs`, `sv-analyze`, `sv-decode`, `sv-export`, `sv-eklund`, `sv-p3-*`,
`sv-questions`.

## License and citation

Code: MIT (`LICENSE`). Documentation, results, reports and the exported detectors: CC BY 4.0
(`LICENSE-CC-BY-4.0.txt`). Please cite the paper and this archive (`CITATION.cff`).
