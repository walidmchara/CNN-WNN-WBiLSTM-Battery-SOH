# Dataset contract

No copyrighted, restricted, or unavailable raw experimental datasets are included.

Use a chronological CSV with one time or cycle column (`time`), numeric observations, and the target column (`target`), either irradiance (W/m^2), PV power (site units), or battery SOH (typically 0–1), as appropriate.

The default input automatically includes the historical target as an autoregressive feature. For a strictly exogenous model, adapt `prepare` after reviewing the paper.

Typical solar columns: `time,temperature,cloud_cover,solar_elevation,target`. Typical battery columns: `time,capacity,temperature,current,target`.

Preclean or train-only impute missing features. Do not randomly shuffle chronological data. To reproduce multi-site/cell experiments, run per site or extend the split with grouped holdout; this scaffold currently supports a single chronology.

`python train.py --smoke` generates synthetic signals for plumbing validation only; it NEVER reproduces published results.


## User-supplied NASA + CALCE source archives

The NASA outer ZIP contains nested archives with the `.mat` files. CALCE CS2_35–38 contain 25/26/27/28 `.xlsx` files; the original CALCE discharge capacity column is cumulative, so it must be differenced per cycle. See [`../docs/REAL_DATA_SETUP.md`](../docs/REAL_DATA_SETUP.md). Run `python scripts/prepare_battery_data.py --help` from the project root. Never commit the original archives or processed CSVs.
