# Lightweight neural networks for battery SOH prediction

This package evaluates four compact PyTorch regressors — **MLP, LSTM, GRU and CNN-LSTM** — on independently held-out lithium-ion batteries. It is separate from the repository's published CNN–WNN–WBiLSTM architecture and must not be described as an exact reproduction of that publication.

## Data sources and experiment definitions

| Source | Input data | Training batteries | Validation batteries | Test batteries |
|---|---|---|---|---|
| NASA PCoE | **12 original `.mat` files**, directly extracted and SHA-256-verified | B0005, B0006, B0029, B0032, B0053, B0054 | B0007, B0031, B0055 | B0018, B0030, B0056 |
| CALCE CS2 | Four cycle-level CSV tables derived from the source XLSX workbooks | CS2_35, CS2_36 | CS2_37 | CS2_38 |

**NASA is now trained from MATLAB files, not preprocessed NASA CSVs.** The complete NASA source archive was verified: 1,159 raw discharge events across twelve files; two zero-capacity terminal events are excluded with explicit source provenance, leaving **1,157 usable cycles**. [NASA source and split audit](NASA_MAT12_PROTOCOL.md).

CALCE remains unchanged, with 3,921 versioned cycle-level records derived from the source workbooks. [CALCE source details](../data/README.md) and [data quality observations](DATA_QA.md).

## Forecasting protocol

Each model takes measurements from **twelve previous discharge cycles** and estimates the SOH of the **following** cycle. The five shared features are discharge-voltage mean/std, discharge-current mean/std, and discharge duration. SOH is each battery's measured capacity divided by its first observed capacity.

Same-cycle capacity, same-cycle SOH, target-derived fields, end-of-dataset cycle counts and features from the predicted cycle are excluded from the network inputs. Feature standardization is fitted only on training batteries. The best model is selected by held-out validation MSE; test batteries are not used for tuning.

| Experiment | Usable train cycles | Validation cycles | Test cycles | Test windows with lookback 12 |
|---|---:|---:|---:|---:|
| NASA MAT12 | 573 | 310 | 274 | 238 |
| CALCE CS2 | 1,855 | 1,038 | 1,028 | 1,016 |

Configuration files: [NASA original MATLAB](../configs/simple_soh_nasa_mat12.json) and [CALCE](../configs/simple_soh_calce.json).

## Google Colab: run the original NASA experiment

**[Copy the full NASA MAT12 Colab cell](COLAB_NASA_MAT12.md).** It clones the private repository using the Colab `GithubToken` secret, mounts Google Drive, restores and verifies all twelve original NASA MATLAB files, trains the GRU for 30 epochs and resumes automatically from persistent checkpoints after interruption.

The official 210 MB NASA ZIP is cached on Drive; if it already exists from a previous upload, it is reused. The original archive is not part of GitHub. The experiment stores model weights and checkpoint state exclusively in Google Drive.

From a configured environment, the equivalent command is:

```bash
python -m simple_soh.restore_nasa_mat \
  --archive /content/drive/MyDrive/NASA_original_source/5.+Battery+Data+Set.zip \
  --dest /content/drive/MyDrive/NASA_original_source/mat \
  --manifest-out /content/drive/MyDrive/NASA_original_source/mat12_source_audit.json

python -m simple_soh.run \
  --config configs/simple_soh_nasa_mat12.json \
  --model gru \
  --nasa-mat-dir /content/drive/MyDrive/NASA_original_source/mat \
  --output-root /content/drive/MyDrive/Battery_SOH_Baselines/results \
  --checkpoint-root /content/drive/MyDrive/Battery_SOH_Baselines/checkpoints
```

Its run identifier is `nasa_mat12_gru_lb12_e30_seed42`. The previous **four-battery NASA CSV trial**, `nasa_gru_lb12_e30_seed42`, remains unchanged on Drive; that older experiment is archived and must not be conflated with the MAT12 study. Its original command is retained in [legacy Colab instructions](COLAB.md).

For CALCE, the independent existing experiment remains:

```bash
python -m simple_soh.run \
  --config configs/simple_soh_calce.json --model lstm \
  --output-root /content/drive/MyDrive/Battery_SOH_Baselines/results \
  --checkpoint-root /content/drive/MyDrive/Battery_SOH_Baselines/checkpoints
```

For either experiment, repeat the same command with `--resume` after a runtime interruption when a matching completed-epoch `latest.pt` checkpoint is present.

## Outputs and reproducibility

The implementation is in [`models.py`](models.py), [`data.py`](data.py), [`nasa_mat.py`](nasa_mat.py), [`restore_nasa_mat.py`](restore_nasa_mat.py) and [`run.py`](run.py).

For each completed run the pipeline writes `metrics.json`, `provenance.json` (including original MATLAB checksums and any excluded discharge events), `training_history.csv`, `test_predictions.csv`, four PNG figures (SOH trajectories, learning curves, parity, residuals), and a `best_model.pt` PyTorch model. Checkpoints include the optimizer and RNG states, training history, and best-validation model; saved snapshots are validated against the input hashes and configuration.

For NASA MAT12, `metrics.json` contains **both pooled and per-test-battery** MAE/RMSE/MSE/MAPE/R². Only genuinely executed 30-epoch runs are eligible for [optional GitHub result publication](publish.py); raw MATLAB data and model weights are never uploaded.

### Verification

GitHub Actions has verified all twelve original MATLAB files, checked their raw and usable discharge counts, verified the source checksums, and successfully executed a **one-epoch** real-data PyTorch run with checkpoint restoration. That technical check is not a completed scientific benchmark or a substitute for 30 epochs in Colab.

[12-file original-source CI audit](../.github/workflows/nasa-mat12-protocol.yml) · [Source dataset checks](DATA_QA.md) · [Research release checklist](RELEASE_REVIEW.md)

## Interpretation limits

The MAT12 source upgrade and additional training cells do not guarantee improved SOH accuracy. Different NASA cell families may have substantial distribution shifts, even under a disjoint split. Repeat-seed studies, cross-cell folds, and fair alternative baselines are needed before generalization claims.

The `remaining_observed_cycles` column counts remaining records before a data file ends. It is **not** a scientifically validated end-of-life RUL target. The present models perform one-cycle-ahead SOH estimation, not established remaining-useful-life prediction.
