# Reproducibility audit — Automatika 2025

- Article DOI: https://doi.org/10.1080/00051144.2025.2461821
- Main code family: `cnn_wavelet_bilstm`
- Raw research data: **NOT bundled**. Supply actual dataset(s), correct train/test dates, units, target definitions and validation splits.
- Architecture: **component-level implementation** from article's named blocks, *not* byte-identical author source. Confirm numbers of layers, kernels, Morlet equations, attention masks, model dimensionality and all table-level hyperparameters.
- Paper-reported numerical results: **not copied into output and not asserted reproduced**.
- Optimization: `none`. `WOA` is a bounded search implementation; `Optuna` uses a TPE sampler; `CBO` is an explicitly **illustrative search**, not an exact verified billiards optimizer. Verify the original equations and search budgets before paper-level comparison.
- Residual model: `none`. Fits residuals ONLY on an independent chronological calibration slice.
- Informer-type modules: sampled-key sparse-query attention prototype, not full original Informer (e.g., no generative decoder).
- SOH vs RUL vs irradiance: **set and verify target separately**. In `--smoke`, target is generated synthetically and has no experimental meaning.
- Batteries: this initial package accepts cycle-level CSV; extraction from NASA/CALCE `.mat`, group-wise cell-level holdouts, SOH formula and any capacity feature leakage need paper-specific scrutiny.
- Multi-step forecasting: current package produces *one* target per input window; forecasting horizon and multivariate multi-step protocol must be implemented exactly for WHDMNet and any multi-step studies.
- Before scientific release: verify all preprocessing, datasets, chronological/group splits, no leakage, baselines, ablations, uncertainty, confidence intervals, compute budget, seeds, optimizer details, deterministic settings.

## Data governance

Data is excluded from Git by `.gitignore`; no credentials stored. This code is prepared for a **private repository**, not yet published.
