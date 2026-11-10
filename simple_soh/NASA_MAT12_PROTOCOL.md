# NASA PCoE: original MATLAB sources and 12-cell SOH protocol

The owner's original NASA archive `5.+Battery+Data+Set.zip` was inspected directly. Its 209,708,670 bytes have SHA-256 `82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e`. It contains six nested ZIPs and all twelve requested original `Bxxxx.mat` files. Exact individual file hashes are specified in [`nasa_mat.py`](nasa_mat.py). Raw binaries are never added to Git.

## Source discharge inventory

| NASA battery | Raw discharge events | Usable cycles | Split |
|---|---:|---:|---|
| B0005 | 168 | 168 | Train |
| B0006 | 168 | 168 | Train |
| B0007 | 168 | 168 | Validation |
| B0018 | 132 | 132 | Test |
| B0029 | 40 | 40 | Train |
| B0030 | 40 | 40 | Test |
| B0031 | 40 | 40 | Validation |
| B0032 | 40 | 40 | Train |
| B0053 | 56 | 55 | Train |
| B0054 | 103 | 102 | Train |
| B0055 | 102 | 102 | Validation |
| B0056 | 102 | 102 | Test |
| **Total** | **1,159** | **1,157** | |

The final raw discharge events of B0053 (source index 56) and B0054 (index 103) each have zero measured capacity and three time samples. The original `.mat` extraction explicitly records these exclusions, including source indices and reasons, in a persistent audit manifest. All remaining discharge cycles are included. The usable counts coincide with the earlier versioned CSV manifest, but **training now loads the original MATLAB files directly**, never the derived NASA CSVs.

## Predictive protocol

The parser uses SciPy's `loadmat` to extract measured voltage, current, duration and capacity from each NASA discharge trace. Five predictor features summarize completed discharges: voltage mean and standard deviation, current mean and standard deviation, and discharge duration. Features from the **twelve preceding cycles** predict the SOH of the following discharge cycle.

Measured capacity is used only to construct the supervised SOH label, as `capacity_ah / first_observed_capacity_ah`; capacity, target columns and same-cycle features of the predicted cycle are excluded from the model input. The feature scaler is fit only on training batteries, and validation is battery-held-out.

| Partition | NASA batteries | Usable cycles | Prediction windows |
|---|---|---:|---:|
| Train | B0005, B0006, B0029, B0032, B0053, B0054 | 573 | 501 |
| Validation | B0007, B0031, B0055 | 310 | 274 |
| Test | B0018, B0030, B0056 | 274 | 238 |

The split is frozen in [`simple_soh_nasa_mat12.json`](../configs/simple_soh_nasa_mat12.json). Each test battery is reported separately as well as in the overall metrics. The MAT experiment uses run ID `nasa_mat12_gru_lb12_e30_seed42`, leaving the former four-cell CSV run `nasa_gru_lb12_e30_seed42` untouched.

## Persistent source recovery

```bash
python -m simple_soh.restore_nasa_mat \
  --archive /content/drive/MyDrive/NASA_original_source/5.+Battery+Data+Set.zip \
  --dest /content/drive/MyDrive/NASA_original_source/mat \
  --manifest-out /content/drive/MyDrive/NASA_original_source/mat12_source_audit.json
```

Existing MATLAB files are validated and reused; missing ones are extracted from the SHA-256-verified archive, which is automatically downloaded from the official NASA PCoE source if needed. A changed ZIP or source file causes a validation error. No MATLAB license is required.

[Colab execution](COLAB_NASA_MAT12.md) · [source parser](nasa_mat.py) · [restoration code](restore_nasa_mat.py) · [twelve-file CI audit](../.github/workflows/nasa-mat12-protocol.yml)

## Scientific limitations

This experiment is not a rerun of the original sample-level CNN–BiLSTM notebooks; it is a separate battery-disjoint one-cycle-ahead SOH prediction task. Training on original `.mat` files does not guarantee improved accuracy. The SOH denominator comes from each cell's first measured capacity and does not establish manufacturer-certified SOH. The recorded end of a dataset is not a validated RUL event. The first short CI training checks are technical only; complete metrics require actual 30-epoch Colab training and further multi-seed testing.
