# Automatika 2025: published reference versus reconstructed implementation

**Publication:** W. Mchara, M. A. Khalfa and L. Manai, “Advanced health state intelligent diagnosis of lithium-ion batteries based on CNN-WNN-WBiLSTM model with attention mechanism,” *Automatika*, 66(2), 154–173 (2025), DOI [10.1080/00051144.2025.2461821](https://doi.org/10.1080/00051144.2025.2461821).

**Status:** A completely new, tested, **paper-grounded reconstruction** is implemented under [`paper_reproduction/`](paper_reproduction/README.md). The original Keras/TensorFlow source and exact random seeds are not available. This work must not be described as a fully established numerical reproduction until training and artifact comparison are complete.

## Published numerical reference — original Table 3

| Battery | MAPE (paper table entry) | MAE | RMSE |
|---|---:|---:|---:|
| B0018 (NASA 24°C) | 0.0042 | **0.0036** | **0.0041** |
| B0030 (NASA 43°C) | 0.0031 | **0.0032** | **0.0039** |
| B0056 (NASA 4°C) | 0.0039 | **0.0029** | **0.0046** |
| CS2-38 (CALCE) | 0.0098 | **0.0088** | **0.0097** |

The published MAPE numerical entries are reproduced literally. Equation (26) multiplies percentage error by 100, so the unit convention of those entries requires careful reconciliation. New calculated metrics are reported unambiguously with a `mape_percent` field.

## Published method — verified directly from the original manuscript

| Item | Published location | Explicit content | Reconstruction |
|---|---|---|---|
| SOH | Equation (1) | Capacity divided by rated capacity | NASA nominal 2 Ah, CALCE nominal 1.1 Ah |
| Measurement inputs | Equation (22) | Per-cycle V, I, T and t matrix | Resampled original time-stamped traces |
| Scaling | Equation (23) | Min–Max normalization | Train-only normalization |
| CNN | Table 2 | Filters 64, kernel 3, ReLU, pooling 2, dropout .3 | Implemented |
| WNN | Equations (3)–(6), Table 2 | Morlet, trainable dilation/translation, 127 units | Implemented |
| Wavelet recurrent | Equations (13)–(17), Table 2 | Morlet-gated wavelet BiLSTM, 94 units | Implemented |
| Attention | Equations (19)–(21), Table 2 | 188 units, sigmoid, temporal weighted sum | Implemented |
| Optimization | Sections 3.2–4.5, Table 2 | RMSprop, lr 0.001, batch 32, MSE, 58 epochs | Implemented |
| Output | Table 2 | ReLU | Implemented |

The original paper does not fully identify sequence padding/resampling, layer connectivity and weight initialization, validation folds, random seeds or the exact scaler fitting scope. This repository explicitly documents every such choice; it does not silently assert exact implementation identity.

## Paper's separate original-source battery groups

| Group | Development | Test |
|---|---|---|
| NASA 24°C | B0005 / B0006 / B0007 | B0018 |
| NASA 43°C | B0029 / B0031 / B0032 | B0030 |
| NASA 4°C | B0053 / B0054 / B0055 | B0056 |
| CALCE | CS2_35 / CS2_36 / CS2_37 | CS2_38 |

The published test cells and NASA 24°C/43°C development groups are explicit. In the NASA 4°C and CALCE families, the selection of the three remaining group cells as development batteries is a **documented reconstruction assumption**, not a recovered exact validation fold.

## Distinguish the three experimental levels

1. **Author-reported scores:** the historical Table 3 numbers above, retained in [`results/published_automatika_2025_reference.json`](results/published_automatika_2025_reference.json).
2. **Architecture and data-source validation:** GitHub Actions tests model dimensions and gradients and exercises an actual one-epoch model fit against original NASA MATLAB traces. This does not establish accuracy replication.
3. **Completed 58-epoch reconstruction:** pending actual training in Colab. Runs have separately named output folders and independent `metrics.json` / prediction CSV / original-file checksum provenance.

**Previous GRU/LSTM runs** under [`simple_soh/`](simple_soh/README.md) are independent exploratory experiments: they use cycle-summary features, first-observed-capacity SOH and other group splits. They cannot serve as reproduction scores for the CNN–WNN–WBiLSTM–AM publication.

**Scientific limitation:** CALCE original workbooks inspected to date have no temperature channel in their Arbin `Channel_` sheets. The reconstruction uses an explicit zero-filled temperature channel, documented as a departure that can influence comparability. Potential incomplete workbook-terminal discharges are retained and flagged for sensitivity analysis, not silently removed to optimize metrics.

**Launch instructions:** [Three original NASA groups and CALCE from source archives](paper_reproduction/COLAB.md).
