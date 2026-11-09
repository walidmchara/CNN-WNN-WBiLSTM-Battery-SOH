# Dataset contract

No copyrighted, restricted, or unavailable raw experimental datasets are included.

Use a chronological CSV with one time or cycle column (`time`), numeric observations, and the target column (`target`), either irradiance (W/m^2), PV power (site units), or battery SOH (typically 0–1), as appropriate.

The default input automatically includes the historical target as an autoregressive feature. For a strictly exogenous model, adapt `prepare` after reviewing the paper.

Typical solar columns: `time,temperature,cloud_cover,solar_elevation,target`. Typical battery columns: `time,capacity,temperature,current,target`.

Preclean or train-only impute missing features. Do not randomly shuffle chronological data. To reproduce multi-site/cell experiments, run per site or extend the split with grouped holdout; this scaffold currently supports a single chronology.

`python train.py --smoke` generates synthetic signals for plumbing validation only; it NEVER reproduces published results.
