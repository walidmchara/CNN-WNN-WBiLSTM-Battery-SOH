# CNN–WNN–WBiLSTM with Attention for Battery SOH

**Publication:** Advanced health state intelligent diagnosis of lithium-ion batteries based on CNN-WNN-WBiLSTM model with attention mechanism  
**Journal:** Automatika (2025)  
**DOI:** https://doi.org/10.1080/00051144.2025.2461821

## Simple neural SOH baselines — NASA and CALCE

A **separate, reproducible SOH forecasting benchmark** is available under [`simple_soh/`](simple_soh/README.md). It complements the original CNN–WNN–WBiLSTM study without changing the associated paper implementation.

| Component | Configuration |
|---|---|
| Networks | MLP, LSTM, GRU and CNN-LSTM (PyTorch) |
| Forecasting target | SOH of the **next cycle**, from 12 preceding measured cycles |
| Features | Five shared discharge measurements; capacity and SOH labels excluded |
| NASA | Train B0005/B0006; validate B0007; test B0018 |
| CALCE | Train CS2_35/CS2_36; validate CS2_37; test CS2_38 |
| Preprocessing | Training-battery-only StandardScaler |
| Checkpoints | Complete model, optimizer, RNG and validation history per epoch |
| Evaluation | MAE, RMSE, MSE, MAPE, R² and signed residuals on unseen test batteries |

Data files and fixed splits are already versioned. The four model classes are implemented in [`simple_soh/models.py`](simple_soh/models.py); the training entry point is [`simple_soh/run.py`](simple_soh/run.py).

```bash
python -m simple_soh.run --config configs/simple_soh_nasa.json --model gru --output-root outputs/simple_soh --checkpoint-root outputs/simple_soh_checkpoints
python -m simple_soh.run --config configs/simple_soh_calce.json --model lstm --output-root outputs/simple_soh --checkpoint-root outputs/simple_soh_checkpoints
```

For persistent Colab runs, direct both output roots to Google Drive. A re-run after interruption uses `--resume`. See the [standalone training and publication instructions](simple_soh/README.md).

**Experimental integrity:** No full-run comparison metrics are reported until the 30-epoch experiments have actually been executed. These benchmarks estimate next-cycle SOH; they do **not** claim validated remaining useful life (RUL), cross-dataset transfer, or exact reproduction of the CNN–WNN–WBiLSTM paper.

---

## Research method

The processing chain combines convolutional feature extraction, a wavelet-neural representation, bidirectional recurrent modeling and an attention-based regression head. Input windows encode past discharge-cycle measurements; the prediction target is a capacity-based SOH indicator.

The project covers data preparation, model fitting, chronological validation, and prediction diagnostics. Configuration parameters are defined in [config.json](config.json), while the main training entry point is [train.py](train.py).

## Battery data

Cycle-level experimental records are included for **12 NASA PCoE cells** and **four CALCE CS2 cells**. The available data, column definitions and provenance hashes are documented in [data/README.md](data/README.md) and [data/processed/manifest.json](data/processed/manifest.json).

The training command below runs against an existing NASA dataset:

```bash
pip install -r requirements.txt
python train.py --csv data/processed/nasa/B0005.csv --target target --time-col time --features voltage_mean,voltage_std,current_mean,current_std,temperature_mean,discharge_duration_s --epochs 2 --out outputs/nasa_trial
```

For a quick software check:

```bash
pytest -q
python train.py --smoke --epochs 2 --out outputs/smoke
```

Results are written under `outputs/` as model checkpoints, epoch-level training records, holdout predictions and evaluation metrics.

## Experimental validity

The current configuration uses a compact implementation of the reported component families. Architectural details, battery-level splits and tuning parameters require alignment with the published protocol before comparative accuracy claims.

The one-cell command demonstrates data compatibility; cross-battery results must use explicitly separated training and test cells. Reported paper metrics are not implied by a successful execution. Full notes on data provenance, split definitions and known limitations are recorded in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [docs/REAL_DATA_SETUP.md](docs/REAL_DATA_SETUP.md).
