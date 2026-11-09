# CNN–WNN–WBiLSTM/Attention: battery data protocol

## Dataset inventory

Cycle-level data are versioned under `data/processed/`:

| Source | Included series | Cycle-level records |
| --- | --- | ---: |
| NASA PCoE | 12 cells: B0005, B0006, B0007, B0018, B0029, B0030, B0031, B0032, B0053, B0054, B0055, B0056 | 1,157 |
| CALCE CS2 | CS2_35, CS2_36, CS2_37, CS2_38 | 3,921 |

All 106 CALCE source workbooks were processed into the four cycle-level CSVs. Input archive SHA-256 hashes and output file digests are recorded in [data/processed/manifest.json](../data/processed/manifest.json).

**Original data sources:** [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [CALCE, University of Maryland](https://calce.umd.edu/data). Original MATLAB files, full Excel traces and ZIP archives are maintained separately from the committed cycle-level tables.

## Data interpretation

NASA measurements provide discharge-cycle electrical and temperature summaries. CALCE CS2 measurements include current, voltage and cumulative discharge-capacity fields; per-cycle capacity was calculated from the cumulative measurements. A CALCE temperature channel should not be assumed when the source workbook does not provide one.

A capacity-based SOH ratio is not a remaining-useful-life target. The data tables include separate fields for health and an observed-end cycle count. Predictors must not include the same-cycle capacity-derived label, and normalization must be fitted on training observations.

## Experimental command

```bash
python train.py --csv data/processed/nasa/B0005.csv --target target --time-col time --features voltage_mean,voltage_std,current_mean,current_std,temperature_mean,discharge_duration_s --epochs 2 --out outputs/nasa_trial
```

This command evaluates a single NASA cell chronologically. The article's cross-battery assessment requires independent held-out cells and verified architectural parameters. The current code retains past targets in autoregressive windows, which is a modeling assumption rather than confirmed reproduction of the paper's feature protocol.

## Regenerating the cycle-level tables

The original archives supplied for this study are named `5.+Battery+Data+Set.zip`, `CS2_35.zip`, `CS2_36.zip`, `CS2_37.zip` and `CS2_38.zip`. The converter supports re-extraction from `data/raw/`:

```bash
python scripts/prepare_battery_data.py --nasa 'data/raw/5.+Battery+Data+Set.zip' --calce data/raw/CS2_35.zip data/raw/CS2_36.zip data/raw/CS2_37.zip data/raw/CS2_38.zip --out data/processed
```

The versioned CSVs are derived from measured battery-aging data. They should be cited through their original data providers, and experimental metrics must be independently verified before comparison with published results.
