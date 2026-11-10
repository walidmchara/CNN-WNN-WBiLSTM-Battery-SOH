# Source-data suitability for the simple SOH benchmark

> **Scope note (NASA):** The NASA four-cell CSV checks below describe the **earlier CSV run**. The new primary NASA protocol reads 12 original MATLAB files directly and is documented in [NASA_MAT12_PROTOCOL.md](NASA_MAT12_PROTOCOL.md). Its split is not the earlier four-cell split. CALCE quality observations below remain applicable.

The input files under `data/processed/` were examined before defining the benchmark splits. The checks below apply to the eight selected cells; the other available NASA series are not part of the current fixed protocol.

## Cell coverage and target verification

| Source | Cell | Cycles | Required-feature NaNs | SOH ratio consistent with capacity | Role |
|---|---|---:|---:|---|---|
| NASA | B0005 | 168 | 0 | Yes | Train |
| NASA | B0006 | 168 | 0 | Yes | Train |
| NASA | B0007 | 168 | 0 | Yes | Validation |
| NASA | B0018 | 132 | 0 | Yes | Test |
| CALCE | CS2_35 | 882 | 0 | Yes | Train |
| CALCE | CS2_36 | 973 | 0 | Yes | Train |
| CALCE | CS2_37 | 1,038 | 0 | Yes | Validation |
| CALCE | CS2_38 | 1,028 | 0 | Yes | Test |

Each cell has strictly increasing discharge-cycle identifiers. All five selected predictors are numeric and present. The `soh_first_observed` column equals `capacity_ah / capacity_ah[0]` in the inspected CSVs. Automated tests repeat these checks for the versioned data at runtime.

The common input features are `voltage_mean`, `voltage_std`, `current_mean`, `current_std`, and `discharge_duration_s`.

## Exclusions and rationale

- `capacity_ah`, `soh_first_observed`, `soh_nominal` and `target` are excluded because they include capacity-derived label information.
- `remaining_observed_cycles` is excluded because it encodes the number of observations remaining before the file ends, not a verified end-of-life event.
- `temperature_mean` is excluded because CALCE temperature measurements are not supplied in these processed files.
- The predicted discharge cycle's own features are excluded. The forecast uses only the preceding `lookback=12` cycles.
- Training/validation/test cells are distinct, and the feature scaler is fitted exclusively on the training cells.

The discharge duration of past cycles may be correlated with their capacities, but is not the contemporaneous target capacity; its use and relevance must still be assessed by future ablation.

## Label and task interpretation

SOH is defined relative to the first observed measured capacity of each battery, not the manufacturer's nominal capacity or an independent ground-truth degradation state. The model predicts SOH at the next recorded cycle after a history of twelve measured discharge cycles. The first twelve cycles of each test cell supply context and are **not** test target rows.

A source file containing 1,028 CALCE cycles therefore yields 1,016 evaluable forecast targets at a lookback of twelve. No interpolation or synthetic test observations are introduced by this pipeline.

**RUL is not included:** the processed datasets do not provide a universally verified physical end-of-life event or a consistent nominal-capacity definition across both sources. Reporting `remaining_observed_cycles` as actual RUL would be methodologically misleading.

Source provenance: [NASA PCoE and CALCE manifest](../data/processed/manifest.json). Experiment definitions: [NASA config](../configs/simple_soh_nasa.json) and [CALCE config](../configs/simple_soh_calce.json).

## Follow-up: exploratory checks after the first 30-epoch Colab trials

These are **post-hoc diagnostics**, not a revised train/test protocol. The original test metrics and CSV files remain unchanged. The underlying source Excel workbooks have **not** yet been inspected independently.

### CALCE CS2_38: apparent incomplete discharge records

In the committed CSV, discharge cycles 96 and 787 each contain only 8 and 7 discharge samples, respectively, with durations approximately 210 and 180 s and estimated capacity approximately 0.06417 Ah. Neighboring discharge cycles are roughly 90–116 samples and 2,640–3,430 s. Both suspect cycles are the **last recorded cycle of their respective Excel source workbook**. The reported SOH ratio at each is approximately 0.0563 and returns to approximately 0.914 or 0.742 on the next cycle.

Corresponding short end-of-workbook records also occur in the development data: CS2_36 cycles 97 and 255, and CS2_37 cycle 98. No corresponding record with both under 20 samples and under 600 seconds was found in CS2_35 during this preliminary check.

**Caution:** a global discharge-duration threshold such as `duration < 600 s` would also remove many plausible late-life cycles from CS2_36 and CS2_37. This is **not** an acceptable blanket filter without domain-specific justification. The raw Excel files and conversion code must be checked to establish whether short workbook-terminal records represent deliberately partial discharges or artifacts. Any quality-controlled reanalysis must be designated exploratory/sensitivity analysis, preserve the unfiltered benchmark, apply consistent rules to all partitions, and must not silently bridge gaps in discharge-cycle numbering.

### NASA: development-to-validation distribution shift

The train-only StandardScaler is fitted on B0005+B0006. In the inspected `voltage_std` feature, B0007 (the entire validation battery) lies outside ±3 training standard deviations for approximately **66.7%** of its cycle rows, with mean validation z-score **+3.23**. The held-out B0018 test cell has a mean z-score approximately **+1.36** and around 2.3% of its cycles beyond ±3. This materially challenges validation-based checkpoint selection; it is a *possible explanation*, not proof, of the poor NASA GRU validation and test results.

Further work should assess train/validation distribution differences on all features, use grouped cross-validation only among **development** cells B0005/B0006/B0007, and keep B0018 unseen during model choice. If previous-cycle SOH is added as an autoregressive input, that constitutes a **separate experimental protocol** requiring explicitly documented information availability and its own run identifier. It must not be confused with same-cycle capacity leakage.

### Current status

The user executed initial 30-epoch trials (NASA GRU and CALCE LSTM) in Colab, with the numerical results retained in their Google Drive. These trials are **not yet published as versioned result files in this repository** and should not be misrepresented as repository-verified scores.
