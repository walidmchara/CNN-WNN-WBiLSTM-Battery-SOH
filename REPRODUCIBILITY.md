# Reproducibility and evaluation notes

**Publication:** Advanced health state intelligent diagnosis of lithium-ion batteries based on CNN-WNN-WBiLSTM model with attention mechanism (Automatika (2025))  
**DOI:** https://doi.org/10.1080/00051144.2025.2461821

## Implementation scope

The processing chain combines convolutional feature extraction, a wavelet-neural representation, bidirectional recurrent modeling and an attention-based regression head. Input windows encode past discharge-cycle measurements; the prediction target is a capacity-based SOH indicator.

The current configuration uses a compact implementation of the reported component families. Architectural details, battery-level splits and tuning parameters require alignment with the published protocol before comparative accuracy claims.

## Data provenance

- NASA PCoE: 12 cell-level CSV files under `data/processed/nasa/`.
- CALCE CS2: four cell-level CSV files under `data/processed/calce/`.
- Source file checksums, record counts and conversion provenance: `data/processed/manifest.json`.
- Source MATLAB and Excel time-series archives are maintained separately; the committed CSVs contain cycle-level aggregates.

## Evaluation protocol

The default `train.py` command performs a chronological split within a single CSV. This is useful for checking preprocessing, training and evaluation, but it is not a substitute for the held-out-cell experiments described in the publication.

The SOH label is capacity-derived. For leakage-aware results, same-cycle capacity or any direct SOH transformation should not be used as a predictor. Verify the treatment of historical target values, outlier removal, normalization statistics, and battery-specific nominal capacities before comparing experiments.

## Results and remaining checks

Automated tests confirm execution of individual software components. Reproduction of the published numerical results has not yet been established. Further checks include architecture hyperparameters, source features, train/validation/test partitions, random seeds, baseline implementations, and uncertainty across repeated runs.
