# Reconstructing the Automatika 2025 CNN–WNN–WBiLSTM–AM experiment

**Original publication:** W. Mchara, M. A. Khalfa, L. Manai, “Advanced health state intelligent diagnosis of lithium-ion batteries based on CNN-WNN-WBiLSTM model with attention mechanism,” *Automatika* **66**(2), 154–173 (2025). DOI [10.1080/00051144.2025.2461821](https://doi.org/10.1080/00051144.2025.2461821).

This package is a clean-room **reconstruction of the paper equations and disclosed hyperparameters**. The original TensorFlow/Keras source is not available. Accordingly, it does **not** claim bitwise equivalence or numerical replication of Table 3. Actual experimental metrics must be generated from completed runs before comparison with author-reported numbers.

## 1. Architecture reconstructed from the original article

| Stage | Paper source | Implemented method |
|---|---|---|
| Input | Section 4.2, Eq. (22) | Per-discharge time-sampled **V, I, T, t** input matrix, four channels |
| CNN | Table 2 | Conv1D(64 filters, kernel 3, ReLU), MaxPooling1D(2), Dropout(0.30) |
| Wavelet neural network | Section 2.4, Eqs. (3)–(6), Table 2 | 127 trainable Morlet units with learned dilation and translation |
| Wavelet BiLSTM | Section 3.1, Eqs. (13)–(17), Table 2 | 94 hidden units **per direction** using bidirectional Morlet-activated recurrent cells |
| Attention | Section 3.3, Eqs. (19)–(21), Table 2 | Sigmoid 188-unit score network, temporal softmax and weighted context |
| SOH output | Table 2 | Dense(1), ReLU |
| Optimization | Section 3.2 and Table 2 | MSE, RMSprop, learning rate **0.001**, 58 epochs, batch size 32 |

**Morlet mother wavelet (Eq. 6):** `M(x) = cos(1.75x) exp(-x²/2)`.

The wavelet recurrent cell implements Eqs. (13)–(17) as printed in the publication; these equations replace ordinary LSTM gate sigmoid/tanh activations with Morlet. This is distinct from an ordinary BiLSTM placed after a fixed wavelet filter bank.

### Reconstruction choices not fully specified in the paper

- **Input length:** Each original cycle is resampled by linear interpolation to 64 equally spaced positions. The paper does not disclose a universal sample length or its exact sequence padding.
- **Validation partition:** The last 15% of the chronological cycles within each *development battery* are reserved for model selection; each target test battery remains excluded from development.
- **Framework:** PyTorch is used for transparent implementation of custom Morlet recurrent gates; the paper reports Keras/TensorFlow. This is an algorithmic reconstruction, not the original executable code.
- **WNN and attention topology:** The trainable dilation/translation follows Eqs. (3)–(6); the 188-unit sigmoid attention scorer and temporal softmax reconcile Table 2 with Eqs. (19)–(21). The original layer connectivity and initialization are not fully recoverable.
- **Normalization scope:** Min–Max scalers are fitted on *development-training* cycles only, then applied unchanged to validation and held-out test cycles. The paper specifies Min–Max scaling but does not unambiguously disclose the fitting scope. This choice prevents test-set preprocessing leakage.
- **Reported task:** The model predicts SOH using sensor measurements acquired **during the evaluated discharge cycle**, not using only preceding cycles. It is a same-cycle post-discharge SOH estimator. Therefore it is not operational one-cycle-ahead forecasting or RUL prediction.

The original article lists regression **accuracy** among Keras metrics. This implementation reports scientifically interpretable MAE, RMSE, MAPE and R², while optimizing MSE.

## 2. Four independent original-source experiments

The original battery groups are **not mixed together**. The protocol is one separately trained network for each experiment.

| Group | Train batteries | Held-out test battery | Nominal capacity |
|---|---|---|---|
| NASA 24 °C | B0005, B0006, B0007 | **B0018** | 2.0 Ah |
| NASA 43 °C | B0029, B0031, B0032 | **B0030** | 2.0 Ah |
| NASA 4 °C | B0053, B0054, B0055 | **B0056** | 2.0 Ah |
| CALCE CS2 | CS2_35, CS2_36, CS2_37 | **CS2_38** | 1.1 Ah |

The B0053–56 and CALCE test allocations reproduce the test-cell choices displayed in Figure 8/Table 3. The precise original allocation of training/validation cycles in these families is not fully documented, so the stated train/validation rules are **reconstruction choices**, not recovered historical fact.

**Target definition (paper Eq. 1):** `SOH = measured_discharge_capacity / rated_capacity`. NASA uses the paper's **2 Ah nominal capacity** and CALCE the paper's **1.1 Ah nominal capacity**. The previous first-discharge normalization `C_t/C_1` is not used. The two zero-capacity terminal NASA records (B0053 and B0054) are excluded with their source indices and source-file hashes retained.

**Raw input sources:** NASA reads verified original `Bxxxx.mat` discharge traces. CALCE reads the original `CS2_xx.zip` archives containing chronologically ordered Excel workbooks and samples from their `Channel_` worksheets. CALCE has **no recorded temperature channel in the inspected raw source**, so its temperature input is zero-filled and flagged in provenance, not invented.

### Published Table 3 reference (not reproduced scores)

| Test battery | MAPE — paper table | MAE — paper table | RMSE — paper table |
|---|---:|---:|---:|
| NASA B0018 | 0.0042 | **0.0036** | **0.0041** |
| NASA B0030 | 0.0031 | **0.0032** | **0.0039** |
| NASA B0056 | 0.0039 | **0.0029** | **0.0046** |
| CALCE CS2-38 | 0.0098 | **0.0088** | **0.0097** |

The MAPE definition in Eq. (26) multiplies by 100%; the table entries require unit-convention reconciliation. New code stores `mape_percent` explicitly as a percentage. Do not claim agreement of MAPE values without reconciling their scales.

## 3. Training and scientific results

For Google Colab, **[use the full executable launcher](COLAB.md)**. The launcher securely clones the private repository, mounts Drive, uses verified original-source files, executes all three NASA training groups independently, saves full PyTorch optimizer/checkpoint state per epoch, and displays numerical results and figures only after training.

Direct runner example for NASA 24 °C:

```bash
python -m paper_reproduction.run \
  --group nasa_24c \
  --source-root /content/drive/MyDrive/NASA_original_source/mat \
  --cache-root /content/drive/MyDrive/Automatika_2025_Reproduction/cache \
  --output-root /content/drive/MyDrive/Automatika_2025_Reproduction/results \
  --checkpoint-root /content/drive/MyDrive/Automatika_2025_Reproduction/checkpoints
```

Change the group to `nasa_43c`, `nasa_4c`, or `calce_cs2` (where `--source-root` points to a directory with `CS2_35.zip` through `CS2_38.zip`). The default is 58 epochs, 64 time samples, batch size 32 and seed 42. Use `--resume` after an interruption when a matching checkpoint exists. **Do not use `--resume` with an unverified or untrusted checkpoint.**

Run IDs are separate from the older GRU/LSTM baseline, e.g. `automatika2025_nasa_24c_p64_e58_seed42`. No previous experimental score is overwritten.

After completion, each run saves `metrics.json`, `provenance.json`, `test_predictions.csv`, `training_history.csv`, four diagnostic PNGs, and a separate `best_model.pt` weight file. It records the published Table 3 values **and** the newly measured results in distinct fields; the model scores are never synthesized.

## 4. Before public release

- Validate architecture and training against Table 2, Eqs. (3)–(21), and the 58-epoch budget. Confirm real-data unit tests and at least one completed training run.
- Preserve the original source hashes and the known B0053/B0054 exclusions. Flag CALCE short workbook-terminal discharges rather than removing them post hoc.
- Compare source-level per-cycle target distributions and inspect anomalous values; do not adjust preprocessing only to improve test scores.
- Compare both reproduced and alternative baseline methods on the *same target definition, evaluation cycles and input availability*.
- Evaluate random-seed variability; do not claim exact reproduction when implementation choices differ from undisclosed original settings.
- Keep source dataset archives and model weights out of public GitHub history; only data-derived metrics/plots with provenance should be committed.

The original manuscript PDF is a research source and is not redistributed in the repository.

[Paper reference audit](../PAPER_REPRODUCTION_REFERENCE.md) · [Original source restoration](../simple_soh/restore_nasa_mat.py) · [Automated source and model tests](../.github/workflows/automatika-paper-reconstruction.yml)
