# CNN–WNN–WBiLSTM with Attention for Battery SOH

## Rebuilding the published CNN–WNN–WBiLSTM–AM network

The [full 2025 *Automatika* manuscript](https://doi.org/10.1080/00051144.2025.2461821) supplies the network component equations and Table 2 hyperparameters. The original source code is unavailable; this repository now contains a separately versioned **paper-grounded reconstruction**, not a claim of bitwise-identical source reproduction.

**[Paper architecture and experimental methodology](paper_reproduction/README.md)** · **[Google Colab 58-epoch NASA/CALCE launcher](paper_reproduction/COLAB.md)** · [Model code](paper_reproduction/models.py) · [Original NASA/CALCE raw-source parser](paper_reproduction/data.py).

The implemented sequence is **CNN(64, kernel 3) → trainable Morlet WNN(127) → bidirectional Morlet LSTM(94) → Attention(188) → ReLU SOH output**. Training uses RMSprop(0.001), MSE, batch size 32, and 58 epochs. The target is **measured capacity / rated capacity** (2.0 Ah NASA; 1.1 Ah CALCE), matching Eq. (1). Original discharge measurements V/I/T/t are used rather than cycle-summary feature substitutes.

Three separate NASA temperature-group networks evaluate B0018, B0030 and B0056 on their original MATLAB source files; a fourth CALCE CS2 experiment reads source Excel workbooks. Validation uses development cycles only. The original file hashes and any exclusions are versioned as provenance; no experimental performance is fabricated.

**[Automated architecture and source-data tests](.github/workflows/automatika-paper-reconstruction.yml)** check model dimensions, Morlet gates, attention weighting, original NASA MATLAB source extraction, and raw CALCE workbook parsing. A one-epoch CI exercise is a code test, **not** the completed 58-epoch reproduction.

The earlier `simple_soh/` GRU/LSTM work is preserved as a **separate exploratory benchmark**. It used different targets, inputs, networks and data splits, so its metrics are not a replication of the published article.

---
## Published article: reproduction reference

The associated [Automatika (2025) paper](https://doi.org/10.1080/00051144.2025.2461821) is the **author-reported numerical reference**, not the new GRU or LSTM experiments. Its CNN–WNN–WBiLSTM–AM architecture and original-data protocol must be reconstructed before claiming reproduction of the publication.

| Published test cell | MAE | RMSE |
|---|---:|---:|
| NASA B0018 | 0.0036 | 0.0041 |
| NASA B0030 | 0.0032 | 0.0039 |
| NASA B0056 | 0.0029 | 0.0046 |
| CALCE CS2-38 | 0.0088 | Under verification |

**[Read the paper-to-code protocol audit and unresolved fidelity issues](PAPER_REPRODUCTION_REFERENCE.md)** · [Machine-readable published reference](results/published_automatika_2025_reference.json).

The source paper reports time-sampled voltage/current/temperature/time inputs, Min–Max scaling and RMSprop; our independent `simple_soh/` baseline uses different features, scaling, architecture, splits and optimizer. **These results are not directly comparable.**

---
## NASA original MATLAB source: all 12 batteries

The new NASA baseline reads the original SHA-256-verified NASA PCoE **`.mat` files** for B0005, B0006, B0007, B0018, B0029, B0030, B0031, B0032, B0053, B0054, B0055 and B0056, with battery-disjoint train/validation/test splits. It does **not** train on the NASA CSV summaries. Two terminal zero-capacity raw discharge events are tracked and excluded, yielding 1,157 usable cycles.

**[NASA 12-cell original-MAT Colab code](simple_soh/COLAB_NASA_MAT12.md)** · [Scientific protocol and original source audit](simple_soh/NASA_MAT12_PROTOCOL.md).

The previous NASA four-cell CSV trial and the separate CALCE CSV run are retained; their results must not be mixed with the new source-and-split protocol. No full training result is claimed for the 12-cell MATLAB experiment yet.

---

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
| NASA (original 12 MATLAB cells) | Train B0005/B0006/B0029/B0032/B0053/B0054; validate B0007/B0031/B0055; test B0018/B0030/B0056 |
| CALCE | Train CS2_35/CS2_36; validate CS2_37; test CS2_38 |
| Preprocessing | Training-battery-only StandardScaler |
| Checkpoints | Complete model, optimizer, RNG and validation history per epoch |
| Evaluation | MAE, RMSE, MSE, MAPE, R² and signed residuals on unseen test batteries |

Data files and fixed splits are already versioned. The four model classes are implemented in [`simple_soh/models.py`](simple_soh/models.py); the training entry point is [`simple_soh/run.py`](simple_soh/run.py).

```bash
python -m simple_soh.run --config configs/simple_soh_nasa_mat12.json --model gru --nasa-mat-dir /content/drive/MyDrive/NASA_original_source/mat --output-root /content/drive/MyDrive/Battery_SOH_Baselines/results --checkpoint-root /content/drive/MyDrive/Battery_SOH_Baselines/checkpoints
python -m simple_soh.run --config configs/simple_soh_calce.json --model lstm --output-root outputs/simple_soh --checkpoint-root outputs/simple_soh_checkpoints
```

For the NASA command, restore the original 12 MATLAB files first; the [Colab instructions](simple_soh/COLAB_NASA_MAT12.md) perform this source verification and save checkpoints on Google Drive. A re-run after interruption uses `--resume` when a matching checkpoint exists. See the [standalone methodology](simple_soh/README.md).

**Experimental integrity:** No full-run comparison metrics are reported until the 30-epoch experiments have actually been executed. These benchmarks estimate next-cycle SOH; they do **not** claim validated remaining useful life (RUL), cross-dataset transfer, or exact reproduction of the CNN–WNN–WBiLSTM paper.

---

## Research method

The processing chain combines convolutional feature extraction, a wavelet-neural representation, bidirectional recurrent modeling and an attention-based regression head. Input windows encode past discharge-cycle measurements; the prediction target is a capacity-based SOH indicator.

The project covers data preparation, model fitting, chronological validation, and prediction diagnostics. Configuration parameters are defined in [config.json](config.json), while the main training entry point is [train.py](train.py).

## Battery data

Cycle-level experimental records are included for **12 NASA PCoE cells** and **four CALCE CS2 cells**. The available data, column definitions and provenance hashes are documented in [data/README.md](data/README.md) and [data/processed/manifest.json](data/processed/manifest.json).

The following command belongs to the original repository's earlier CSV-based prototype; it does not use the twelve-cell MATLAB pipeline:

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
