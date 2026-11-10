# NASA and CALCE: simple neural SOH benchmark

## NASA source update: twelve original MATLAB batteries

The primary NASA experiment now loads the original **B0005, B0006, B0007, B0018, B0029, B0030, B0031, B0032, B0053, B0054, B0055 and B0056** `.mat` files, **not the versioned NASA CSV tables**. The official NASA ZIP is SHA-256-verified, and the original source files are verified individually before processing. This dataset contains 1,159 raw discharge events, of which two terminal zero-capacity events are explicitly excluded, leaving 1,157 usable cycles.

**NASA MAT12 experimental split:** train B0005/B0006/B0029/B0032/B0053/B0054; validation B0007/B0031/B0055; test B0018/B0030/B0056. These are disjoint batteries. The MATLAB run uses its own persistent identifier, `nasa_mat12_gru_lb12_e30_seed42`. The earlier four-battery CSV NASA run has not been overwritten and is not directly comparable to this new protocol. CALCE retains its original CSV source.

[**Colab: run NASA from original MATLAB files**](COLAB_NASA_MAT12.md) · [MAT12 source audit, counts and splits](NASA_MAT12_PROTOCOL.md) · [separate CALCE/legacy workflow](COLAB.md)

```bash
python -m simple_soh.restore_nasa_mat --archive /content/drive/MyDrive/NASA_original_source/5.+Battery+Data+Set.zip --dest /content/drive/MyDrive/NASA_original_source/mat --manifest-out /content/drive/MyDrive/NASA_original_source/mat12_source_audit.json
python -m simple_soh.run --config configs/simple_soh_nasa_mat12.json --model gru --nasa-mat-dir /content/drive/MyDrive/NASA_original_source/mat --output-root /content/drive/MyDrive/Battery_SOH_Baselines/results --checkpoint-root /content/drive/MyDrive/Battery_SOH_Baselines/checkpoints
```

The new 30-epoch MATLAB training has **not** been run in Colab yet; no new numerical test metrics are asserted.

---

This package evaluates four lightweight PyTorch regressors on original-data-derived cycle-level measurements: **MLP, LSTM, GRU, and CNN-LSTM**. It is separate from the CNN–WNN–WBiLSTM architecture implemented at the repository root. No numerical result from the wavelet-model publication is attributed to these new baseline experiments.

## Experimental task

The network forecasts the SOH of discharge cycle *t* using **only measurements from the 12 preceding discharge cycles**, t−12 through t−1. No features from the predicted cycle are supplied. SOH is defined as observed cycle capacity divided by the first observed capacity **of that battery**.

Five features common to both dataset families are used:

- Mean and standard deviation of discharge voltage
- Mean and standard deviation of discharge current
- Duration of the preceding discharge

Same-cycle capacity, SOH labels, any direct target transformation, the dataset-end cycle count, and future observations are excluded from the predictor matrix. The target SOH is used **only as the supervised label**.

The processed CALCE CS2 data do not supply temperature, so the models do not assume a CALCE temperature channel.

## Pre-registered battery splits

| Dataset | Train cells | Validation cell | Test cell | Predicted targets |
|---|---|---|---|---|
| NASA PCoE | B0005, B0006 | B0007 | B0018 | 120 B0018 cycles (13–132) |
| CALCE CS2 | CS2_35, CS2_36 | CS2_37 | CS2_38 | 1,016 CS2_38 cycles (13–1028) |

The split is fixed in `configs/simple_soh_nasa.json` and `configs/simple_soh_calce.json`. **No battery appears in two partitions.** Feature normalization is fitted only to training batteries and applied unchanged to validation and test batteries. Model checkpoint selection uses validation MSE; test observations are not used to select hyperparameters.

These are **separate within-source experiments**, not a demonstrated NASA↔CALCE transfer test. The full inventories of 12 NASA and four CALCE cells remain available for later group and leave-one-cell-out protocols.

For a private GitHub/Google Colab workflow, see the [copy-and-run Colab cell](COLAB.md). Source suitability checks are in [DATA_QA.md](DATA_QA.md) and the pre-release research criteria in [RELEASE_REVIEW.md](RELEASE_REVIEW.md).

## Installation

From the repository root, in an environment with Python, PyTorch and the dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

### NASA: GRU baseline

```bash
python -m simple_soh.run \
  --config configs/simple_soh_nasa.json \
  --model gru \
  --output-root /content/drive/MyDrive/Battery_SOH_Baselines/results \
  --checkpoint-root /content/drive/MyDrive/Battery_SOH_Baselines/checkpoints
```

### CALCE: LSTM baseline

```bash
python -m simple_soh.run \
  --config configs/simple_soh_calce.json \
  --model lstm \
  --output-root /content/drive/MyDrive/Battery_SOH_Baselines/results \
  --checkpoint-root /content/drive/MyDrive/Battery_SOH_Baselines/checkpoints
```

The same commands can use `--model mlp` or `--model cnn_lstm`. They require no Jupyter notebook. Each model has a separate run identifier, so experiments do not overwrite one another.

## Recovery after Colab interruption

After each completed epoch the package saves a persistent PyTorch model/optimizer checkpoint, the best-validation model state, its epoch, the training history, and RNG states in a single atomically replaced `latest.pt` file. Re-execute the **same command** with `--resume` after a runtime interruption. An incomplete epoch is retrained; completed epochs are preserved. The checkpoint fingerprint covers the training protocol and SHA-256 hashes of the input CSV files. A changed model, split, hyperparameters or source file prevents unsafe resume.

The default training budget is 30 epochs with seed 42, learning rate 0.001, batch size 32, and 32 hidden units. The entire budget is executed; the lowest-validation-MSE epoch is used for test inference. The use of a random seed is not a guarantee of bitwise identical outcomes across CPU/GPU versions.

## Outputs

The results folder contains:

| File | Content |
|---|---|
| `metrics.json` | Test MAE, RMSE, MSE, MAPE, R², bias, best epoch and sample counts |
| `provenance.json` | Split, feature list, architecture, parameters, source file SHA-256 |
| `training_history.csv` | Training and validation MSE per epoch |
| `test_predictions.csv` | Test battery, target cycle, last observed cycle, actual and predicted SOH |
| `soh_trajectory.png` | Observed and predicted test-cell SOH |
| `learning_curve.png` | Training and validation loss |
| `parity.png` | Measured versus predicted SOH |
| `residuals.png` | Signed error by test-cell cycle |
| `best_model.pt` | Model state and scaler statistics; stored on Drive only |

The persistent checkpoint is separate from the final model. The source CSVs, weights, and recovery snapshots are not published to GitHub by the result exporter.

## Publishing completed results

Once a complete 30-epoch run is verified, publication is optional. The `simple_soh.publish` command uses the GitHub Contents API, **without a local git rebase**, to store only eight report/figure files and a checksummed manifest under `results/simple_soh/<run-id>/`. Set `GITHUB_TOKEN` securely in the execution environment; do not print it.

```bash
python -m simple_soh.publish \
  --run-dir "/content/drive/MyDrive/Battery_SOH_Baselines/results/nasa_gru_lb12_e30_seed42"
```

The uploader refuses to write to a public repository, refuses incomplete experiments, and refuses to replace different already published data under the same run ID. All published results must reflect actually executed training sessions.

## Limitations

- This is **SOH next-cycle forecasting**, not a validated RUL prediction benchmark. The `remaining_observed_cycles` column is the number of remaining entries in a recorded file, **not** a verified end-of-life RUL label. Do not report it as ground-truth RUL.
- CALCE and NASA have different measurement domains, chemistry and operating profiles. Separate-model results cannot establish cross-dataset transfer.
- The label's first-observation capacity is assumed available to define the evaluation reference for a battery; it is never a model input.
- Features are cycle-complete summaries **of past cycles**, not online measurements from a future cycle. The model's operational forecast horizon is one discharge cycle.
- Only one test battery per source is held out. Additional independent batteries, multi-seed experiments, baseline fairness and uncertainty assessment are required before publication-quality comparative claims.
- The package is a clean baseline experiment, **not an exact software reproduction** of the published CNN–WNN–WBiLSTM paper. The root-level original implementation and its documentation remain independent.

Data attribution and provenance are documented in [data/README.md](../data/README.md) and [data/processed/manifest.json](../data/processed/manifest.json). The source files must be cited to NASA PCoE and CALCE/University of Maryland, respectively.
