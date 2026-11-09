# Automatika-CNN-WNN-WBiLSTM-Attention

**Article:** Advanced health state intelligent diagnosis of lithium-ion batteries based on CNN-WNN-WBiLSTM model with attention mechanism  
**Journal:** Automatika (2025)  
**DOI:** https://doi.org/10.1080/00051144.2025.2461821

## Status: executable research *reconstruction*, not verified original code

The modules implement the named architecture family (`cnn_wavelet_bilstm`), chronological data handling, training, testing, and diagnostic outputs. This project was newly prepared using the PDF, not extracted from the original authors' source. **Never present its outputs as the paper's exact published results** until all hyperparameters, feature engineering, source datasets, protocol, loss details and random seeds are independently audited.

The paper identifies **none** optimization and **none** residual modeling. Both are exposed in the code, where applicable. `cbo` is currently an explicitly illustrative chaotic-search placeholder, *not* a validated transcription of the full published optimizer. `informer` is a compact sampled-key sparse attention prototype rather than a validated full Informer implementation.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python train.py --smoke --epochs 2 --out outputs/smoke
```

## Real data

```bash
python train.py --csv data/your_data.csv --target target --time-col time --epochs 40 --out outputs/real_trial
```

Optional small optimization sanity check:

```bash
python train.py --smoke --epochs 2 --search paper --trials 3 --out outputs/search_smoke
```

**Important:** the default `--search none` avoids falsely implying the global optimizer has run. For full published-protocol reproduction you must first validate `config.json` and the manuscript's original search space, objective function, budget, and seed procedures.

Outputs: checkpoint, training history, independent chronological test predictions, JSON metrics, plot. Neural residuals are calibrated on a separate chronological slice *inside* the train period; independent test data is never used to train the residual learner.

Read `REPRODUCIBILITY.md` and `data/README.md` before claiming reproduction.


## Real NASA PCoE and CALCE CS2 datasets (2026-11-09 integration)

The original NASA ZIP and four CALCE CS2 ZIP archives have been inspected. Use the new offline converter in [`scripts/prepare_battery_data.py`](scripts/prepare_battery_data.py) and see [`docs/REAL_DATA_SETUP.md`](docs/REAL_DATA_SETUP.md) for commands, source links, quality controls, target definitions and limitations. Source archives and derived CSVs are deliberately ignored by Git. This integration validates the **data import interface**, not the published paper metrics.


## Real NASA and CALCE datasets now provided

This repository includes derived, cycle-level CSV data in [data/processed/](data/processed/) for 12 NASA cells and four CALCE CS2 cells, obtained from original experimental archives. See [the dataset manifest](data/processed/manifest.json) and [setup guide](docs/REAL_DATA_SETUP.md). Raw archive files are not included. Existing smoke tests remain distinct from article-level replication.
