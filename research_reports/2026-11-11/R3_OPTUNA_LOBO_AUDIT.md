# R3 Optuna LOBO audit — 11 November 2026

**Basis:** Executed notebook `Automatika_2025_R3_Optuna_LOBO_Checkpoints (1).ipynb` and previously prepared static audit; no direct Drive checkpoint access in this archive step.

| Group | Completed | Pruned | Research-best trial | Research score | Confirmed trial | 3-seed mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| NASA 43°C | 38 | 22 | 56 | 0.01980310 | 1 | 0.020104 | 0.000166 |
| NASA 4°C | 60 | 0 | 44 | 0.03220767 | 44 | 0.032498 | 0.000319 |

Objective (not test RMSE): `0.7 * mean(LOBO fold RMSE) + 0.3 * max(LOBO fold RMSE)`. Each group has 3 development batteries, two fit / one validate. The excluded test batteries B0030 and B0056 are not used for this R3 tuning.

**Confirmed 43°C trial 1:** CNN 128/kernel 5, WNN 127, standard BiLSTM 94, attention 128/sigmoid, dropout 0.4, 128 points, batch 64, RMSprop LR 0.00193773642, weight decay 1.325477e-7, clip 3, MSE+variance lambda 0.0014349256.

**Confirmed 4°C trial 44:** CNN 64/kernel 3, WNN 192, standard BiLSTM 128, attention 188/tanh, dropout 0.5, 128 points, batch 32, RMSprop LR 0.0003621276, weight decay 0.0009695816, clip 0.5, MSE.

## Audit findings

1. **High distribution shift:** B0029 as held-out LOBO fold had 24.15% normalized feature values outside the training range. Subsequent channel-wise audit located 95.96% out-of-range in `Current_measured` on that fold. These are different aggregation levels; do not conflate them.
2. **Duplicate trials:** 43°C trials 1 and 2 had identical hyperparameters and the same cached score `0.020084109691612766`; not independent confirmation observations.
3. **Pruning:** 22 trials in 43°C were pruned after partial fold results; fold ordering can bias selection.
4. **Uncertainty:** Three seeds give an empirical stability check, not proof of full determinism.
5. **Checkpoint provenance:** Notebook implements epoch checkpoints and metadata fingerprints; static inspection alone cannot certify their future availability.
6. **Architecture identity:** R3 uses a standard BiLSTM, not the original Morlet-gated recurrent cell, so it is an *optimized ablation*, not an exact paper-source reproduction.

## Action

Preserve studies, deduplicate configuration rankings, document fold-level predictions, investigate the current-signal shift in physical units, and assess independent datasets before asserting external validity.
