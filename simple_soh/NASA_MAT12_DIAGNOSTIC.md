# Diagnostic note — NASA twelve-cell GRU pilot, 30 epochs

**Protocol:** `nasa_mat12_gru_lb12_e30_seed42`. Original NASA MATLAB source files; six training cells, three validation cells and three test cells; twelve prior cycles per forecast; seed 42; 30 epochs. These scores were supplied in the completed Colab log, not yet committed as a versioned `metrics.json`/`test_predictions.csv` bundle. Do not treat them as GitHub-artifact-verified until the original output files are uploaded.

## Reported held-out test results

| Test battery | Windows | MAE | RMSE | R² | Mean signed error |
|---|---:|---:|---:|---:|---:|
| B0018 | 120 | 0.076790 | 0.089860 | −0.522004 | +0.049851 |
| B0030 | 28 | 0.054734 | 0.055813 | −3.488452 | −0.054734 |
| B0056 | 90 | 0.686630 | 0.693478 | −144.048191 | −0.686630 |
| **Pooled** | **238** | **0.304807** | **0.431619** | **−0.838742** | **−0.240954** |

Best model selection: validation MSE `0.0684424` at epoch 29; training MSE at epoch 30 `0.006422`. This gap, together with the negative test R², does not support generalization claims. B0056 accounts for about **97.6%** of pooled test squared error. The outputs are retained on Google Drive; they are not to be altered or superseded without a separately identified experiment.

## Source data check: initial capacity reference

The original source files were checked directly, independent of predictions. The original target is `SOH_t = C_t / C_1`. In the B005x family, the first recorded discharge capacity is markedly lower than the second:

| Battery | First discharge (Ah) | Second discharge (Ah) | Median discharge capacity, cycles 2–12 (Ah) | Maximum current first-cycle-normalized SOH |
|---|---:|---:|---:|---:|
| B0054 | 0.739935 | 1.166544 | 1.056837 | 1.57655 |
| B0055 | 0.799000 | 1.319892 | 1.200961 | 1.65193 |
| B0056 | 0.785278 | 1.344356 | 1.258950 | 1.71195 |

The first-cycle denominator causes the label range to exceed 1.5 for these cells. This is a **target-definition and cross-cell calibration concern**, not evidence that the raw NASA files are corrupt or proof that the first discharge was partial. The phenomenon is present in training (B0054), validation (B0055), and test (B0056). Avoid editing source files or silently dropping samples.

## Transparent reference baseline

A one-cycle persistence rule (`predicted SOH at t = measured SOH at t−1`) computed from the original MATLAB capacities, for cycles 13 onward, yields:

| Battery | MAE (persistence) | RMSE (persistence) | R² (persistence) |
|---|---:|---:|---:|
| B0018 | 0.007917 | 0.012638 | 0.969893 |
| B0030 | 0.007747 | 0.009770 | 0.862465 |
| B0056 | 0.035676 | 0.046553 | 0.346341 |
| **Pooled** | **0.018394** | **0.030188** | **0.991005** |

Persistence **uses a different information set**, because it requires knowledge of the preceding cycle's measured SOH/capacity, whereas the GRU receives five past-cycle electrical summaries. It is an informative operational reference, not a perfectly matched input-feature comparison.

## Follow-up analysis, not a replacement for the recorded run

1. Preserve the present 30-epoch experiment, configuration and test predictions unchanged.
2. Define an **exploratory** target-sensitivity protocol using a reference derived from a fixed early observation window (e.g., the median capacity of cycles 2–12). These values are already available before cycle-13 forecasting; apply the same definition to every train, validation and test battery. Do not choose a denominator separately for B0056.
3. Use a **distinct experiment ID**, and repeat training; original model scores cannot simply be rescaled and claimed as new trained-model performance.
4. Quantify distribution shift and performance by battery family (B0005–18, B0029–32, B0053–56); compare MLP/LSTM/GRU/CNN-LSTM using identical split, seed, budget and target.
5. Because the target sensitivity was motivated by examining the B0056 test results, any new evaluation on the same held-out battery is **post-hoc exploratory**. A fresh held-out protocol or cross-validation is required for confirmatory claims.

The data-loading code deliberately excludes same-cycle capacity as a network input and fits feature normalization on training batteries only. This note identifies a target reference issue, **not same-cycle feature leakage**.
