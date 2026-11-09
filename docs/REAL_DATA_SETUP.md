# Real datasets installed: NASA PCoE and CALCE CS2

## Files checked into the repository
- `data/processed/nasa/B0005.csv` through the 12 selected batteries listed in `data/processed/manifest.json` (1,157 discharge-cycle rows across 12 cells).
- `data/processed/calce/CS2_35.csv`, `CS2_36.csv`, `CS2_37.csv`, `CS2_38.csv` (3,921 cycle-level rows across **all 106 CALCE XLSX workbooks**).
- `data/processed/manifest.json`: input archive SHA-256 fingerprints, cell IDs, output record counts.
- `scripts/prepare_battery_data.py`: the code used to regenerate derived CSVs.

The original `.mat` and Excel/ZIP archives are **not** tracked because they are very large; derived CSVs are tracked. Source links: [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [CALCE/UMD](https://calce.umd.edu/data). NASA asks researchers to acknowledge data donors; CALCE asks publications to cite the relevant experimental studies. Verify conditions for subsequent redistribution.

## Derived columns and warnings

Each CSV has `cycle`, `time`, `capacity_ah`, `soh_first_observed`, `target`, `cell_id`, diagnostic statistics and `remaining_observed_cycles`. SOH here is relative to the **first observed** discharge capacity, not necessarily the published nominal rating. `remaining_observed_cycles` counts records until *the end of the observed file* and is **not** confirmed EOL-based RUL.

NASA: one row per valid discharge cycle. Some cells have `target > 1` because the first measured capacity is lower than later measurements; do not clip these silently. CALCE: discharge capacity is cumulative **within each workbook**; conversion uses capacity maximum minus minimum per valid cycle, corrects identical-time duplicates, and flags workbook boundary cycles. CALCE measurement sheets lack an observed temperature channel, so `temperature_mean` is missing.

**Avoid target leakage.** Do not feed the same-cycle `capacity_ah`, `soh_first_observed`, `soh_nominal`, `target`, or `remaining_observed_cycles` as predictor features for an SOH model. The current reconstruction's `train.py` appends *past* target values to historical windows; this is an autoregressive research assumption, not an independently validated paper implementation. Split by cell for cross-battery benchmarks; the default one-CSV chronology is a pipeline test only.

## Using the committed data

From repository root:

```bash
python -m pip install -r requirements.txt
# NASA, one cell: historical voltage/current/temperature features
python train.py --csv data/processed/nasa/B0005.csv --target target --time-col time --features voltage_mean,voltage_std,current_mean,current_std,temperature_mean,discharge_duration_s --epochs 2
# CALCE, one cell: exclude missing temperature and all target-proxy features
python train.py --csv data/processed/calce/CS2_35.csv --target target --time-col time --features voltage_mean,voltage_std,current_mean,current_std,discharge_duration_s --epochs 2
```

For the public WTX-CBO project, use `python main.py --csv ... --target-col remaining_observed_cycles` only as an illustrative observed-end countdown target; obtaining a scientifically valid RUL requires an EOL event definition and censoring protocol. For the legacy TensorFlow notebooks in the CNN-BiLSTM project, the notebooks still need original NASA `.mat` files in `data/`; these compact CSV aggregates cannot replace sample-level traces.

## Regenerating from original archives

Put the downloaded `5.+Battery+Data+Set.zip` and `CS2_35.zip`–`CS2_38.zip` into `data/raw/`, then:

```bash
python scripts/prepare_battery_data.py --nasa 'data/raw/5.+Battery+Data+Set.zip' --calce data/raw/CS2_35.zip data/raw/CS2_36.zip data/raw/CS2_37.zip data/raw/CS2_38.zip --out data/processed --extract-nasa-mat-to data/raw/nasa
```

CSV data are derived from real measurements, **not** synthetic demonstration data. They have not been used here to verify published paper-level accuracy.
