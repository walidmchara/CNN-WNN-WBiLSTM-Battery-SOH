# Source-data suitability for the simple SOH benchmark

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
