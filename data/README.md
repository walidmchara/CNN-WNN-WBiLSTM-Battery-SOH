# Experimental battery data

This directory contains cycle-level NASA PCoE and CALCE CS2 records prepared from their respective battery-aging archives.

## Versioned datasets

- [NASA discharge cycles](processed/nasa/): B0005, B0006, B0007, B0018, B0029, B0030, B0031, B0032, B0053, B0054, B0055, B0056.
- [CALCE CS2 discharge cycles](processed/calce/): CS2_35, CS2_36, CS2_37, CS2_38.
- [Data provenance](processed/manifest.json): source checksums, conversion counts and output file hashes.

The tables include measurements aggregated by discharge cycle and derived health indicators. NASA and CALCE records have different measurement channels; CALCE data should not be assumed to contain temperature. The original time-sample traces are not part of these CSVs.

## Running a cycle-level experiment

```bash
python train.py --csv data/processed/nasa/B0005.csv --target target --time-col time --features voltage_mean,voltage_std,current_mean,current_std,temperature_mean,discharge_duration_s --epochs 2 --out outputs/nasa_trial
```

The one-cell chronological experiment checks the implementation; an independent battery-level split is required for cross-cell generalization claims. Health labels are derived from capacity and must not be exposed as same-cycle predictors.

Dataset sources: [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [CALCE/UMD](https://calce.umd.edu/data). Full data-processing notes are in [../docs/REAL_DATA_SETUP.md](../docs/REAL_DATA_SETUP.md).
